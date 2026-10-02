"""Reproducible real-model probes of the frozen Stage2A.3.1 action path.

Run with the PointerCAD interpreter. The prepared root is produced offline by
cadquery2crs/scripts/prepare_stage2a3_native.py in the CadQuery environment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import yaml
from transformers import AutoTokenizer

from models.crs_pointercad.brep_bridge import validate_candidate_alignment
from models.crs_pointercad.contracts import PointerType
from models.crs_pointercad.sequence import Stage2QwenCollator, frozen_grammar_vocabulary
from models.crs_pointercad.training import PreparedStage2Corpus, training_action_loss, _target_key
from models.model_factory import from_config


FROZEN = Path('/mnt/afs/L202500475/cadquery2crs-corpus/stage2-clean-validation-v1')
CHECKPOINT = Path('/mnt/afs/L202500475/hf-data/models/Qwen2.5-0.5B-Instruct')


class GateInputs:
    def __init__(self, prepared_root: Path):
        self.prepared = PreparedStage2Corpus(FROZEN, prepared_root)
        manifest = json.loads((FROZEN / 'manifest.json').read_text())
        self.records = {}
        for entry in manifest['entries']:
            identity = tuple(entry[key] for key in ('dataset', 'sample_id', 'source_variant_id', 'approved_variant_id'))
            supervision = json.loads((FROZEN / entry['supervision_artifact']).read_text())
            self.records[entry['sample_id']] = SimpleNamespace(identity=identity, supervision=supervision)
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        tokenizer.chat_template = Path('config/chat_template.jinja').read_text()
        self.vocabulary = frozen_grammar_vocabulary([record.supervision for record in self.records.values()])
        self.collator = Stage2QwenCollator(tokenizer, grammar_vocabulary=self.vocabulary)
        self.sequences = {}

    def sequence(self, sample_id: str):
        if sample_id not in self.sequences:
            self.sequences[sample_id] = self.prepared.collate_record(self.records[sample_id], self.collator)
        return self.sequences[sample_id]

    def state(self, sample_id: str, action_index: int):
        return self.prepared.state_for(self.records[sample_id].identity, action_index)

    def model(self):
        config = yaml.safe_load(Path('config/crs_expanded_9k.yaml').read_text())
        config['model']['base_model'] = str(CHECKPOINT)
        config['model']['grammar_vocab_size'] = len(self.vocabulary)
        return from_config(config).to('cuda')


def inspect(inputs: GateInputs) -> dict:
    counts = Counter()
    types = Counter()
    failures = []
    lengths = []
    prepared_steps = 0
    native = Counter()
    resolved_targets = Counter()
    aligned_candidates = Counter()
    topology_edges = 0
    for sample_id, record in inputs.records.items():
        try:
            sequence = inputs.sequence(sample_id)
            condition = inputs.prepared.conditioning_for(record.identity)
            assert condition == {'text': 'Construct the CAD model.', 'source': 'legacy_pointercad_adapter_constant'}
            assert sequence.conditioning_text == condition['text']
            assert sequence.conditioning_source == condition['source']
            counts.update(grammar=len(sequence.grammar_positions), pointer=len(sequence.pointer_positions),
                          scalar=len(sequence.scalar_positions), record=len(sequence.structured_record_positions))
            types.update(target['ref_kind'] for target in record.supervision['pointer_targets'])
            lengths.append(int(sequence.input_ids.numel()))
            entry = inputs.prepared.entries[record.identity]
            prepared_steps += len(entry['steps'])
            for step in entry['steps']:
                native.update(face_rows=step['face_rows'], edge_rows=step['edge_rows'])
                if step['candidate_counts'].get('FACE', 0) > step['face_rows']:
                    raise ValueError('FACE candidate count exceeds native rows')
                if step['candidate_counts'].get('EDGE', 0) > step['edge_rows'] + step.get('loose_edge_rows', 0):
                    raise ValueError('EDGE candidate count exceeds native rows')
                step_root = inputs.prepared.prepared_root / step['path']
                graph = json.loads((step_root / 'metadata.json').read_text())
                state = json.loads((step_root / 'state.json').read_text())
                if len(graph['face_adjacency']) != graph['edge_count']:
                    raise ValueError('face adjacency and graph edge rows differ')
                if any(len(pair) != 2 or min(pair) < 0 or max(pair) >= graph['face_count']
                       for pair in graph['face_adjacency']):
                    raise ValueError('face adjacency has an invalid row')
                topology_edges += len(graph['face_adjacency'])
                for kind in ('FACE', 'EDGE'):
                    candidate_keys = {json.dumps(item['key'], sort_keys=True) for item in state['candidates'][kind]}
                    native_keys = graph['face_keys'] if kind == 'FACE' else graph['edge_keys'] + graph['loose_edge_keys']
                    native_set = {json.dumps(item, sort_keys=True) for item in native_keys}
                    if not candidate_keys <= native_set:
                        raise ValueError(f'{kind} candidate lacks native owner/key row')
                    aligned_candidates[kind] += len(candidate_keys)
            for target in record.supervision['pointer_targets']:
                step = entry['steps'][target['action_index']]
                state = json.loads((inputs.prepared.prepared_root / step['path'] / 'state.json').read_text())
                kind = target['ref_kind']
                key = target['target_candidate']
                if key not in [item['key'] for item in state['candidates'][kind]]:
                    raise ValueError(f'{kind} target candidate not in causal bank at action {target["action_index"]}')
                resolved_targets[kind] += 1
        except Exception as exc:
            failures.append({'sample_id': sample_id, 'error': f'{type(exc).__name__}: {exc}'})
    return {'records': len(inputs.records), 'prepared_records': len(inputs.prepared.entries),
            'prepared_steps': prepared_steps, 'fixed_conditioning_count': len(inputs.records)-len(failures),
            'mapped_targets': dict(counts), 'pointer_types': dict(types),
            'native_rows_total': dict(native), 'qwen_sequence_tokens_min_max': [min(lengths), max(lengths)],
            'resolved_pointer_targets': dict(resolved_targets),
            'native_aligned_face_edge_candidates': dict(aligned_candidates),
            'validated_graph_edges': topology_edges,
            'failures': failures}


def action_specs(sequence, supervision, action_index):
    action = next(item for item in sequence.action_boundaries if item.action_index == action_index)
    prefix = sequence.input_ids[:sequence.conditioning_end]
    ids = torch.cat((prefix, sequence.input_ids[action.start:action.end])).unsqueeze(0)
    def local(position):
        return position - action.start + sequence.conditioning_end
    specs = []
    for index, target in enumerate(sequence.pointer_positions):
        if target.action_index != action_index:
            continue
        source = next(item for item in supervision['pointer_targets']
                      if item['action_index'] == action_index and item['position'] == target.command_position)
        specs.append({'position': local(target.model_position),
                      'feedback_position': local(sequence.feedback_positions[index]),
                      'pointer_type': PointerType('PTR_' + target.target_candidate['kind']),
                      'target_key': _target_key(target.target_candidate),
                      'decoder_substate': source.get('decoder_substate', {})})
    return ids, specs


def gradient_norms(model) -> dict:
    prefixes = {'qwen_lora': 'base_model', 'pointer_projection': 'query_heads.queries',
                'pointer_temperature': 'query_heads.log_tau', 'native_uvnet': 'brep.',
                'scalar_head': 'numeric_heads.scalar', 'record_heads': 'numeric_heads.records',
                'registry_context': 'registry_context.', 'context_projection': 'context_projection.'}
    result = {}
    for label, prefix in prefixes.items():
        gradients = [parameter.grad.float().norm().item() for name, parameter in model.named_parameters()
                     if name.startswith(prefix) and parameter.grad is not None]
        result[label] = {'parameters_with_grad': len(gradients), 'max_norm': max(gradients, default=0.0)}
    return result


def step(inputs: GateInputs, model, sample_id: str, action_index: int, *, optimize=True) -> dict:
    record = inputs.records[sample_id]
    sequence = inputs.sequence(sample_id)
    state, brep = inputs.state(sample_id, action_index)
    action_ids, specs = action_specs(sequence, record.supervision, action_index)
    call_lengths = []
    captured = {}
    original_forward = model.forward_ragged
    def capture_forward(**kwargs):
        output = original_forward(**kwargs)
        captured['banks'] = output.candidate_banks_by_example[0]
        return output
    model.forward_ragged = capture_forward
    hook = model.base_model.register_forward_pre_hook(
        lambda module, args, kwargs: call_lengths.append(
            int(kwargs['inputs_embeds'].shape[1]) if kwargs.get('inputs_embeds') is not None else -1), with_kwargs=True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        loss, parts = training_action_loss(model, sequence, record.supervision, action_index, state, brep)
        torch.cuda.synchronize()
        forward = time.perf_counter() - start
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - start
        gradients = gradient_norms(model)
        start = time.perf_counter()
        if optimize:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        optimizer_time = time.perf_counter() - start
        return {'status': 'PASS', 'sample_id': sample_id, 'action_index': action_index,
                'pointer_slots': len(specs), 'qwen_tokens': int(action_ids.numel()),
                'candidate_counts': [len(bank) for bank in captured['banks']],
                'full_backbone_calls': len(call_lengths), 'backbone_token_lengths': call_lengths,
                'loss': float(loss.detach()),
                'components': {key: float(parts[key].detach()) if present else None for key, present in {
                    'grammar': any(target.action_index == action_index for target in sequence.grammar_positions),
                    'pointer': bool(specs),
                    'scalar': any(target.action_index == action_index for target in sequence.scalar_positions),
                    'record': any(target.action_index == action_index for target in sequence.structured_record_positions),
                }.items()},
                'gradients': gradients, 'forward_seconds': forward, 'backward_seconds': backward,
                'optimizer_seconds': optimizer_time, 'total_seconds': forward + backward + optimizer_time,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'native_face_rows': len(brep.face_keys), 'native_edge_rows': len(brep.edge_keys),
                'native_alignment': validate_candidate_alignment(brep)}
    finally:
        hook.remove()
        model.forward_ragged = original_forward


@torch.no_grad()
def feedback_probe(inputs: GateInputs, model, sample_id: str = '00002', action_index: int = 2) -> dict:
    model.eval()
    record = inputs.records[sample_id]
    sequence = inputs.sequence(sample_id)
    state, brep = inputs.state(sample_id, action_index)
    ids, specs = action_specs(sequence, record.supervision, action_index)
    if len(specs) < 2:
        raise ValueError('feedback probe requires at least two natural pointer slots')
    ids = ids.to('cuda')
    mask = torch.ones_like(ids)
    kwargs = dict(input_ids=ids, attention_mask=mask, states=[state], pointer_specs=[specs], breps=[brep])
    normal = model.forward_ragged(**kwargs)
    original = model.feedback_for
    try:
        model.feedback_for = lambda pointer_type, bank, index: torch.zeros(model.hidden_dim, device=ids.device)
        ablated = model.forward_ragged(**kwargs)
    finally:
        model.feedback_for = original
    first = specs[0]
    span = next(item for item in sequence.token_spans
                if item.start + sequence.conditioning_end - next(action.start for action in sequence.action_boundaries
                if action.action_index == action_index) == first['position'])
    first_logit_delta = float((normal.pointer_logits_by_example[0][0] - ablated.pointer_logits_by_example[0][0]).abs().max())
    second_logit_delta = float((normal.pointer_logits_by_example[0][1] - ablated.pointer_logits_by_example[0][1]).abs().max())
    own_hidden_delta = float((normal.pointer_hidden_by_example[0][0] - ablated.pointer_hidden_by_example[0][0]).abs().max())
    later_hidden_delta = float((normal.pointer_hidden_by_example[0][1] - ablated.pointer_hidden_by_example[0][1]).abs().max())
    return {'sample_id': sample_id, 'action_index': action_index,
            'pointer_slots': len(specs), 'first_atom_qwen_span_length': span.end-span.start,
            'first_feedback_position': first['feedback_position'],
            'first_pointer_position': first['position'],
            'first_pointer_logit_max_delta': first_logit_delta,
            'first_pointer_hidden_max_delta': own_hidden_delta,
            'second_pointer_logit_max_delta': second_logit_delta,
            'second_pointer_hidden_max_delta': later_hidden_delta,
            'pass': first_logit_delta < 1e-5 and own_hidden_delta < 1e-5 and
                    second_logit_delta > 1e-5 and later_hidden_delta > 1e-5}


@torch.no_grad()
def inference_probe(inputs: GateInputs, model, sample_id: str = '00002', action_index: int = 2) -> dict:
    model.eval()
    record = inputs.records[sample_id]
    sequence = inputs.sequence(sample_id)
    state, brep = inputs.state(sample_id, action_index)
    ids, specs = action_specs(sequence, record.supervision, action_index)
    ids = ids.to('cuda')
    mask = torch.ones_like(ids)
    kwargs = dict(pointer_positions=[item['position'] for item in specs],
                  pointer_types=[item['pointer_type'] for item in specs],
                  decoder_substates=[item['decoder_substate'] for item in specs],
                  feedback_positions=[item['feedback_position'] for item in specs],
                  breps=[brep], return_logits=True)
    cached, cached_logits = model.inference_pointer_decode(ids, mask, state, use_cache=True, **kwargs)
    uncached, reference_logits = model.inference_pointer_decode(ids, mask, state, use_cache=False, **kwargs)
    max_delta = max(float((a-b).abs().max()) for a,b in zip(cached_logits, reference_logits))
    logit_scale = max(float(b.abs().max()) for b in reference_logits)
    relative_scale_delta = max_delta / max(logit_scale, 1.0)
    same_keys = [item[2] for item in cached] == [item[2] for item in uncached]
    return {'sample_id': sample_id, 'action_index': action_index, 'pointer_slots': len(specs),
            'cached_predicted_keys': [str(item[2]) for item in cached],
            'reference_predicted_keys': [str(item[2]) for item in uncached],
            'max_logit_delta': max_delta, 'max_reference_logit_magnitude': logit_scale,
            'max_delta_as_fraction_of_logit_scale': relative_scale_delta,
            'same_candidates': same_keys,
            'pass': same_keys and max_delta < 0.1 and relative_scale_delta < 0.02,
            'tolerance_basis': 'bf16 pointer logits are scaled by 1/0.07; require same candidate and <=0.1 absolute, <=2% of observed scale'}


def ragged_batch_probe(inputs: GateInputs, model) -> dict:
    """Compare a natural heterogeneous batch with its isolated action forwards."""
    examples = [('00002', 2), ('00008', 4)]
    model.train()
    # Keep cuDNN GRUs in training mode for backward while making independent
    # reference forwards deterministic and avoiding BatchNorm state drift.
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
    rows = []
    for sample_id, action_index in examples:
        record = inputs.records[sample_id]
        sequence = inputs.sequence(sample_id)
        state, brep = inputs.state(sample_id, action_index)
        ids, specs = action_specs(sequence, record.supervision, action_index)
        rows.append((sample_id, action_index, sequence, record, state, brep, ids.squeeze(0), specs))
    width = max(len(row[6]) for row in rows)
    pad_id = inputs.collator.tokenizer.pad_token_id or 0
    ids = torch.full((2, width), pad_id, dtype=torch.long, device='cuda')
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        length = len(row[6])
        ids[index, :length] = row[6].to('cuda')
        mask[index, :length] = 1
    output = model.forward_ragged(input_ids=ids, attention_mask=mask,
                                  states=[row[4] for row in rows],
                                  pointer_specs=[row[7] for row in rows],
                                  breps=[row[5] for row in rows])
    grammar_logits, grammar_targets, scalar_predictions, scalar_targets = [], [], [], []
    positives = []
    row_results = []
    max_isolation_delta = 0.0
    max_isolation_scale = 0.0
    same_isolated_rankings = True
    for batch_index, (sample_id, action_index, sequence, record, state, brep, row_ids, specs) in enumerate(rows):
        action = next(item for item in sequence.action_boundaries if item.action_index == action_index)
        def local(position):
            return position - action.start + sequence.conditioning_end
        for index, target in enumerate(sequence.grammar_positions):
            if target.action_index == action_index:
                grammar_logits.append(output.grammar_logits[batch_index, local(target.model_position)])
                grammar_targets.append(sequence.grammar_targets[index])
        for index, target in enumerate(sequence.scalar_positions):
            if target.action_index == action_index:
                scalar_predictions.append(output.scalar_predictions[batch_index, local(target.model_position)])
                scalar_targets.append(sequence.scalar_targets[index])
        banks = output.candidate_banks_by_example[batch_index]
        if len(banks) != len(specs):
            raise AssertionError('ragged bank isolation failed')
        positives.append([bank.positive_mask((spec['target_key'],), device='cuda')
                          for bank, spec in zip(banks, specs)])
        with torch.no_grad():
            isolated = model.forward_ragged(input_ids=row_ids.unsqueeze(0).to('cuda'),
                                            attention_mask=torch.ones((1, len(row_ids)), dtype=torch.long, device='cuda'),
                                            states=[state], pointer_specs=[specs], breps=[brep])
        for actual, reference in zip(output.pointer_logits_by_example[batch_index], isolated.pointer_logits_by_example[0]):
            max_isolation_delta = max(max_isolation_delta, float((actual-reference).abs().max()))
            max_isolation_scale = max(max_isolation_scale, float(reference.abs().max()))
            same_isolated_rankings &= int(actual.argmax()) == int(reference.argmax())
        row_results.append({'sample_id': sample_id, 'action_index': action_index,
                            'qwen_tokens': len(row_ids), 'pointer_slots': len(specs),
                            'pointer_types': [spec['pointer_type'].value for spec in specs],
                            'candidate_counts': [len(bank) for bank in banks]})
    loss, parts = model.loss(grammar_logits=torch.stack(grammar_logits),
                             grammar_targets=torch.stack(grammar_targets).to('cuda'),
                             pointer_logits=output.pointer_logits_by_example,
                             pointer_positive=positives,
                             scalar_predictions=torch.stack(scalar_predictions),
                             scalar_targets=torch.tensor(scalar_targets, device='cuda'))
    loss.backward()
    gradients = gradient_norms(model)
    with torch.no_grad():
        changed_ids = ids.clone()
        changed_ids[1, 1] = (changed_ids[1, 1] + 1) % model.base_model.config.vocab_size
        changed = model.forward_ragged(input_ids=changed_ids, attention_mask=mask,
                                       states=[row[4] for row in rows],
                                       pointer_specs=[row[7] for row in rows],
                                       breps=[row[5] for row in rows])
        cross_delta = max(float((actual-reference).abs().max())
                          for actual, reference in zip(output.pointer_logits_by_example[0],
                                                       changed.pointer_logits_by_example[0]))
    return {'examples': row_results, 'finite_loss': bool(torch.isfinite(loss)),
            'components': {key: float(parts[key].detach()) for key in ('grammar', 'pointer', 'scalar')},
            'valid_pointer_slots': parts['valid_pointer_slots'],
            'max_pointer_logit_isolation_delta': max_isolation_delta,
            'max_isolated_logit_magnitude': max_isolation_scale,
            'same_isolated_argmax': same_isolated_rankings,
            'cross_sample_input_perturbation_delta': cross_delta,
            'gradients': gradients,
            'pass': bool(torch.isfinite(loss)) and same_isolated_rankings and
                    max_isolation_delta/max(max_isolation_scale, 1.0) < 0.05 and
                    cross_delta < 1e-4 and
                    parts['valid_pointer_slots'] == sum(len(row[7]) for row in rows)}


@torch.no_grad()
def legacy_parity_probe(inputs: GateInputs, model) -> dict:
    """Compare Stage2's keyed rows with the same native UVNet forward."""
    model.eval()
    rows = []
    for sample_id, action_index in (('00002', 9), ('00013', 3)):
        state, brep = inputs.state(sample_id, action_index)
        graph = brep.graph.to('cuda')
        direct_edge, direct_face, _, _ = model.brep(graph)
        keyed = model._native_maps([state], [brep])[0]
        face_delta = max((float((keyed[0][key]-direct_face[0][index]).abs().max())
                          for index, key in enumerate(brep.face_keys)), default=0.0)
        edge_delta = max((float((keyed[1][key]-direct_edge[0][index]).abs().max())
                          for index, key in enumerate(brep.edge_keys)), default=0.0)
        rows.append({'sample_id': sample_id, 'action_index': action_index,
                     'face_feature_shape': list(brep.face_features.shape),
                     'edge_feature_shape': list(brep.edge_features.shape),
                     'loose_edge_feature_shape': list(brep.loose_edge_features.shape),
                     'graph_nodes': graph.num_nodes(), 'graph_directed_edges': graph.num_edges(),
                     'face_keys': len(brep.face_keys), 'edge_keys': len(brep.edge_keys),
                     'loose_edge_keys': len(brep.loose_edge_keys),
                     'face_embedding_shape': list(direct_face[0].shape),
                     'edge_embedding_shape': list(direct_edge[0].shape),
                     'max_keyed_vs_direct_face_delta': face_delta,
                     'max_keyed_vs_direct_edge_delta': edge_delta,
                     'alignment': validate_candidate_alignment(brep),
                     'pass': face_delta < 1e-6 and edge_delta < 1e-6})
    return {'examples': rows, 'pass': all(row['pass'] for row in rows),
            'checkpoint_level_comparison': 'unavailable: no original Pointer-CAD checkpoint in workspace'}


def micro_overfit(inputs: GateInputs, model, epochs: int = 12) -> dict:
    """Train a fixed action subset drawn from four frozen natural trajectories."""
    torch.manual_seed(20261002)
    actions = [('00002', 0), ('00002', 2), ('00002', 5), ('00002', 9),
               ('00008', 2), ('00008', 4), ('00013', 3), ('00111', 1)]
    model.train()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    curves = []
    for epoch in range(epochs):
        sums, counts = Counter(), Counter()
        correct = pointer_count = grammar_correct = grammar_count = 0
        for sample_id, action_index in actions:
            record = inputs.records[sample_id]
            sequence = inputs.sequence(sample_id)
            state, brep = inputs.state(sample_id, action_index)
            captured = {}
            original_forward = model.forward_ragged
            def capture(**kwargs):
                output = original_forward(**kwargs)
                captured['output'] = output
                return output
            model.forward_ragged = capture
            try:
                loss, parts = training_action_loss(model, sequence, record.supervision, action_index, state, brep)
            finally:
                model.forward_ragged = original_forward
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite micro-overfit loss at epoch {epoch}, {sample_id}:{action_index}')
            output = captured['output']
            for logits, bank, target in zip(output.pointer_logits_by_example[0],
                                            output.candidate_banks_by_example[0],
                                            (target for target in sequence.pointer_positions if target.action_index == action_index)):
                correct += int(int(logits.argmax()) == bank.external_to_index(_target_key(target.target_candidate)))
                pointer_count += 1
            action = next(item for item in sequence.action_boundaries if item.action_index == action_index)
            for index, target in enumerate(sequence.grammar_positions):
                if target.action_index == action_index:
                    position = target.model_position - action.start + sequence.conditioning_end
                    grammar_correct += int(int(output.grammar_logits[0, position].argmax()) == int(sequence.grammar_targets[index]))
                    grammar_count += 1
            present = {'grammar': any(target.action_index == action_index for target in sequence.grammar_positions),
                       'pointer': bool(output.pointer_logits_by_example[0]),
                       'scalar': any(target.action_index == action_index for target in sequence.scalar_positions),
                       'record': any(target.action_index == action_index for target in sequence.structured_record_positions)}
            sums['total'] += float(loss.detach())
            counts['total'] += 1
            for key, exists in present.items():
                if exists:
                    sums[key] += float(parts[key].detach())
                    counts[key] += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del output, loss
        curves.append({'epoch': epoch, 'means': {key: sums[key]/counts[key] if counts[key] else None
                                                  for key in ('total','grammar','pointer','scalar','record')},
                       'present_action_counts': dict(counts),
                       'pointer_any_positive_at_1': correct/pointer_count if pointer_count else None,
                       'grammar_accuracy': grammar_correct/grammar_count if grammar_count else None})
    first, last = curves[0]['means'], curves[-1]['means']
    return {'trajectory_ids': sorted({item[0] for item in actions}), 'actions': actions,
            'epochs': epochs, 'curves': curves,
            'total_relative_change': (last['total']-first['total'])/first['total'],
            'component_relative_changes': {key: (last[key]-first[key])/first[key] if first[key] else None
                                           for key in ('grammar','pointer','scalar','record')},
            'material_total_decrease': last['total'] < 0.8*first['total']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--mode', choices=('inspect', 'step', 'feedback', 'inference', 'ragged', 'coverage', 'parity', 'overfit'), required=True)
    parser.add_argument('--sample-id', default='00002')
    parser.add_argument('--action-index', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    inputs = GateInputs(args.prepared)
    if args.mode == 'inspect':
        result = inspect(inputs)
    else:
        model = inputs.model()
        if args.mode == 'step':
            result = {'model': model.training_config(), 'step': step(inputs, model.train(), args.sample_id, args.action_index)}
        elif args.mode == 'feedback':
            result = feedback_probe(inputs, model, args.sample_id, args.action_index)
        elif args.mode == 'ragged':
            result = ragged_batch_probe(inputs, model)
        elif args.mode == 'parity':
            result = legacy_parity_probe(inputs, model)
        elif args.mode == 'coverage':
            model.train()
            result = {'sample_id': '00002', 'actions': [
                step(inputs, model, '00002', index, optimize=False)
                for index in (1, 2, 4, 5, 7, 9)
            ]}
        elif args.mode == 'overfit':
            result = micro_overfit(inputs, model, args.epochs)
        else:
            result = inference_probe(inputs, model, args.sample_id, args.action_index)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'status': result.get('status', result.get('pass', 'RECORDED'))}))
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
