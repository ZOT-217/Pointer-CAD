"""Eight-view processor, span, manifest, and causal injection regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from models.crs_pointercad.images import Stage2ImageManifest
from models.crs_pointercad.model import CRSExpandedPointerCAD
from models.crs_pointercad.sequence import Stage2QwenCollator
from models.crs_pointercad.contracts import BodyRecord, BodyVersionRegistry, ExecutionState, ExternalKey, PointerType


@pytest.fixture
def images(tmp_path):
    paths = []
    for index in range(8):
        path = tmp_path / f"view_{index}.png"
        Image.new("RGB", (56, 56), (index * 29, 7, 255 - index * 29)).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def processor():
    model = os.environ.get("STAGE2_QWEN25VL_MODEL")
    if not model or not Path(model).exists():
        pytest.skip("set STAGE2_QWEN25VL_MODEL to real Qwen2.5-VL processor assets")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model, local_files_only=True, use_fast=False)


def record(paths, *, pointers=False):
    atoms = ["START", "<pointer_identifier_alpha>", "MIDDLE", "<pointer_identifier_beta>", "END"]
    return {
        "conditioning": {"text": "Construct the CAD model.", "source": "legacy_pointercad_adapter_constant"},
        "image_paths": paths,
        "command": {"tokens": atoms, "action_boundaries": [{"action_index": 0, "start": 0, "end": len(atoms)}]},
        "pointer_targets": ([{"action_index": 0, "position": i, "slot": str(i),
                             "target_candidate": {"kind": "BODY", "owner": "body_a", "local_index": 0}}
                            for i in (1, 3)] if pointers else []),
        "parameter_targets": [], "structured_record_targets": [],
    }


def collator(processor):
    return Stage2QwenCollator(processor, grammar_vocabulary={
        "START": 0, "MIDDLE": 1, "END": 2,
        "<pointer_identifier_alpha>": 3, "<pointer_identifier_beta>": 4,
    })


def test_real_processor_exact_order_and_command_spans(processor, images):
    first = collator(processor)(record(images, pointers=True))
    reversed_views = collator(processor)(record(list(reversed(images)), pointers=True))
    assert first.image_grid_thw.shape == (8, 3)
    assert first.pixel_values is not None and not torch.equal(first.pixel_values, reversed_views.pixel_values)
    patch_counts = [int(row.prod()) for row in first.image_grid_thw]
    assert len(set(patch_counts)) == 1
    patches_per_view = patch_counts[0]
    for index in range(8):
        original = first.pixel_values[index * patches_per_view:(index + 1) * patches_per_view]
        reversed_position = reversed_views.pixel_values[(7 - index) * patches_per_view:(8 - index) * patches_per_view]
        assert torch.equal(original, reversed_position)
    assert first.visual_token_positions == reversed_views.visual_token_positions
    assert max(first.visual_token_positions) < first.context_position < first.action_boundaries[0].start
    assert all(span.start >= first.conditioning_end for span in first.token_spans)
    for target, feedback in zip(first.pointer_positions, first.feedback_positions):
        span = first.token_spans[target.command_position]
        assert span.end - span.start > 1
        assert target.model_position == span.start and feedback == span.end
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    config = AutoConfig.from_pretrained(os.environ["STAGE2_QWEN25VL_MODEL"], local_files_only=True)
    action = first.action_boundaries[0]
    ids = torch.cat((first.input_ids[:first.conditioning_end],
                     first.input_ids[action.start:action.end])).unsqueeze(0)
    mrope, _ = Qwen2_5_VLForConditionalGeneration.get_rope_index(
        SimpleNamespace(config=config), ids, first.image_grid_thw, None, None, torch.ones_like(ids))
    assert mrope.shape == (3, 1, ids.shape[1])
    assert not torch.equal(mrope[0], mrope[1])


def test_missing_view_fails(processor, images):
    with pytest.raises(ValueError, match="exactly eight"):
        collator(processor)(record(images[:7]))
    images[4].unlink()
    with pytest.raises(FileNotFoundError):
        collator(processor)(record(images))


def test_manifest_order_and_missing_view(tmp_path, images):
    identity = ["zero2cad100k-train", "00002", "000", "v000"]
    views = [{"view_index": i, "locator": path.name, "width": 56, "height": 56,
              "source": "upstream_original"} for i, path in enumerate(images)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "stage2-eight-view-images-v1",
                                    "entries": [{"corpus_identity": identity, "views": views}]}))
    assert Stage2ImageManifest(manifest).paths_for(identity) == tuple(images)
    views.pop()
    manifest.write_text(json.dumps({"format": "stage2-eight-view-images-v1",
                                    "entries": [{"corpus_identity": identity, "views": views}]}))
    with pytest.raises(ValueError, match="exactly eight"):
        Stage2ImageManifest(manifest)


class FakeVision(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dtype = torch.float32

    def forward(self, pixel_values, grid_thw):
        count = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum()) // 4
        return self.weight.expand(count, -1)


class FakeDecoder(nn.Module):
    def forward(self, *, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


class FakeVL(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim, image_token_id=151655, use_cache=False)
        self.embedding = nn.Embedding(152000, dim)
        self.visual = FakeVision(dim)
        self.model = FakeDecoder()

    def get_input_embeddings(self):
        return self.embedding

    def get_rope_index(self, input_ids, image_grid_thw, *args):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1)
        return positions.expand(3, input_ids.shape[0], -1), None


def test_visual_tokens_preserved_and_post_span_feedback_causal(processor, images):
    sequence = collator(processor)(record(images, pointers=True))
    model = CRSExpandedPointerCAD(base_model=FakeVL(), hidden_dim=16, pointer_dim=128,
                                  grammar_vocab_size=5, use_native_brep=False)
    ids = sequence.input_ids.unsqueeze(0)
    state = ExecutionState()
    embeddings, _ = model._prepare_embeddings(
        ids, state, pixel_values=sequence.pixel_values, image_grid_thw=sequence.image_grid_thw,
        visual_token_positions=sequence.visual_token_positions,
        context_position=sequence.context_position)
    visual = embeddings[0, list(sequence.visual_token_positions)]
    assert torch.equal(visual, torch.ones_like(visual))
    # A later teacher selection cannot alter the hidden state that produced it.
    start, end = sequence.token_spans[1].start, sequence.token_spans[1].end
    altered = embeddings.clone()
    altered[:, end:, :] += 3.0
    before = model._backbone_forward(embeddings, None, model._vlm_position_ids(ids, None, sequence.image_grid_thw)).last_hidden_state
    after = model._backbone_forward(altered, None, model._vlm_position_ids(ids, None, sequence.image_grid_thw)).last_hidden_state
    assert torch.equal(before[:, start, :], after[:, start, :])
    assert not torch.equal(before[:, sequence.token_spans[3].start, :], after[:, sequence.token_spans[3].start, :])
    assert torch.equal(embeddings[0, list(sequence.visual_token_positions)],
                       altered[0, list(sequence.visual_token_positions)])
    a = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"area": 1.0}]})
    b = BodyRecord(ExternalKey("body_b"), 0, "Sketch", geometry={"faces": [{"area": 3.0}]})
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({a.key: a, b.key: b}))
    specs = [[(target.model_position, PointerType.BODY) for target in sequence.pointer_positions]]
    common = dict(input_ids=ids, attention_mask=torch.ones_like(ids), states=[state],
                  pointer_specs=specs, feedback_positions=[sequence.feedback_positions],
                  pixel_values=sequence.pixel_values, image_grid_thw=sequence.image_grid_thw,
                  context_positions=[sequence.context_position],
                  visual_token_positions=[sequence.visual_token_positions])
    first = model.forward_ragged(**common, teacher_indices=[[0, 1]])
    second = model.forward_ragged(**common, teacher_indices=[[1, 0]])
    assert torch.equal(first.pointer_hidden_by_example[0][0], second.pointer_hidden_by_example[0][0])
    assert not torch.equal(first.pointer_hidden_by_example[0][1], second.pointer_hidden_by_example[0][1])


def test_multiview_ragged_zero_and_multi_pointer_batch(processor, images):
    sequence = collator(processor)(record(images, pointers=True))
    model = CRSExpandedPointerCAD(base_model=FakeVL(), hidden_dim=16, pointer_dim=128,
                                  grammar_vocab_size=5, use_native_brep=False).eval()
    a = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"area": 1.0}]})
    b = BodyRecord(ExternalKey("body_b"), 0, "Sketch", geometry={"faces": [{"area": 3.0}]})
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({a.key: a, b.key: b}))
    slots = [(target.model_position, PointerType.BODY) for target in sequence.pointer_positions]
    ids = torch.stack((sequence.input_ids, sequence.input_ids))
    result = model.forward_ragged(
        input_ids=ids, attention_mask=torch.ones_like(ids), states=[ExecutionState(), state],
        pointer_specs=[[], slots], teacher_indices=[[], [0, 1]],
        feedback_positions=[[], sequence.feedback_positions],
        pixel_values=[sequence.pixel_values, sequence.pixel_values],
        image_grid_thw=[sequence.image_grid_thw, sequence.image_grid_thw],
        context_positions=[sequence.context_position, sequence.context_position],
        visual_token_positions=[sequence.visual_token_positions, sequence.visual_token_positions])
    assert len(result.pointer_logits_by_example[0]) == 0
    assert len(result.pointer_logits_by_example[1]) == 2
    assert [len(bank) for bank in result.candidate_banks_by_example[1]] == [2, 2]
    assert model.vision_forward_count == 2


def test_official_qwen25vl_decoder_receives_visuals_and_mrope(processor, images):
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    config = AutoConfig.from_pretrained(os.environ["STAGE2_QWEN25VL_MODEL"], local_files_only=True)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.rope_scaling["mrope_section"] = [2, 3, 3]
    config.vision_config.depth = 2
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 128
    config.vision_config.num_heads = 4
    config.vision_config.out_hidden_size = 64
    config.vision_config.fullatt_block_indexes = [1]
    sequence = collator(processor)(record(images))
    model = CRSExpandedPointerCAD(base_model=Qwen2_5_VLForConditionalGeneration(config),
                                  grammar_vocab_size=5, use_native_brep=False,
                                  registry_context_mode="TYPE_POOLED").eval()
    ids = sequence.input_ids.unsqueeze(0)
    body = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"area": 1.0}]})
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body.key: body}))
    output = model.forward_ragged(
        input_ids=ids, attention_mask=torch.ones_like(ids), states=[state], pointer_specs=[[]],
        pixel_values=sequence.pixel_values, image_grid_thw=sequence.image_grid_thw,
        context_positions=[sequence.context_position],
        visual_token_positions=[sequence.visual_token_positions])
    assert output.hidden_states.shape == (1, ids.shape[1], 64)
    assert torch.isfinite(output.grammar_logits).all()
    assert model.vision_forward_count == 1
    features = model.encode_visual(sequence.pixel_values, sequence.image_grid_thw)
    reused = model.forward_ragged(
        input_ids=ids, attention_mask=torch.ones_like(ids), states=[state], pointer_specs=[[]],
        pixel_values=sequence.pixel_values, image_grid_thw=sequence.image_grid_thw,
        context_positions=[sequence.context_position],
        visual_token_positions=[sequence.visual_token_positions], visual_features=features)
    assert model.vision_forward_count == 2
    assert torch.allclose(output.hidden_states, reused.hidden_states)
    output.grammar_logits[0, sequence.action_boundaries[0].start - 1].sum().backward()
    assert model.context_projection.weight.grad is not None
