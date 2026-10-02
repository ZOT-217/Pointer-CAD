"""Stage2A.3.1 text conditioning and pointer span regressions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest
import torch

from models.crs_pointercad import (
    BodyRecord, BodyVersionRegistry, CRSExpandedPointerCAD, ExecutionState,
    ExternalKey, PointerType, Stage2QwenCollator,
)
from models.crs_pointercad.training import training_action_loss
from models.crs_pointercad.training import PreparedStage2Corpus
from models.crs_pointercad.frozen_corpus import FrozenStage2Record


class SplitPointerTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        assert [item["role"] for item in messages] == ["system", "user"]
        text = next(item["text"] for item in messages[1]["content"] if item["type"] == "text")
        return f"<|im_start|>system\n{messages[0]['content']}<|im_end|><|im_start|>user\n{text}<|im_end|><|im_start|>assistant\n<|cad_start|>"

    def __call__(self, value, *, add_special_tokens=False):
        assert add_special_tokens is False
        if value == "<pe>":
            return {"input_ids": [31, 32, 33]}
        return {"input_ids": [64 + ord(character) % 64 for character in value]}


def _record(text="Make a bracket"):
    return {
        "conditioning": {"text": text, "source": "dataset_annotation"},
        "command": {
            "tokens": ["START", "<pe>", "MIDDLE", "<pe>", "END", "FUTURE"],
            "action_boundaries": [
                {"action_index": 0, "start": 0, "end": 5},
                {"action_index": 1, "start": 5, "end": 6},
            ],
        },
        "pointer_targets": [
            {"action_index": 0, "position": 1, "slot": "first", "target_index": 0,
             "target_candidate": {"kind": "BODY", "owner": "body_a", "local_index": 0}},
            {"action_index": 0, "position": 3, "slot": "second", "target_index": 1,
             "target_candidate": {"kind": "BODY", "owner": "body_b", "local_index": 0}},
        ],
        "parameter_targets": [],
        "structured_record_targets": [],
    }


def _collator():
    return Stage2QwenCollator(SplitPointerTokenizer(), grammar_vocabulary={
        "START": 0, "MIDDLE": 1, "END": 2, "FUTURE": 3,
    })


def _state():
    a = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"area": 1.0}]})
    b = BodyRecord(ExternalKey("body_b"), 0, "Sketch", geometry={"faces": [{"area": 3.0}]})
    return ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({a.key: a, b.key: b}))


def test_conditioning_text_is_in_loss_bearing_action_and_future_targets_do_not_leak():
    torch.manual_seed(7)
    one, two = _collator()(_record("Make a bracket")), _collator()(_record("Make a shaft"))
    assert not torch.equal(one.input_ids[:one.conditioning_end], two.input_ids[:two.conditioning_end])
    assert one.action_boundaries[0].start == one.conditioning_end
    assert one.grammar_positions[0].model_position == one.conditioning_end - 1
    assert one.pointer_positions[0].model_position == one.token_spans[1].start
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=4, use_native_brep=False).eval()
    inputs = []
    original = model.forward_ragged

    def capture(**kwargs):
        inputs.append(kwargs["input_ids"].clone())
        return original(**kwargs)

    model.forward_ragged = capture
    training_action_loss(model, one, _record("Make a bracket"), 0, _state(), None)
    training_action_loss(model, two, _record("Make a shaft"), 0, _state(), None)
    assert not torch.equal(inputs[0], inputs[1])

    changed_future = deepcopy(_record("Make a bracket"))
    changed_future["command"]["tokens"][5] = "ALTERED_FUTURE"
    changed_future["pointer_targets"].append({
        "action_index": 1, "position": 5, "slot": "future", "target_index": 0,
        "target_candidate": {"kind": "BODY", "owner": "body_b", "local_index": 0},
    })
    future = _collator()(changed_future)
    training_action_loss(model, future, changed_future, 0, _state(), None)
    assert torch.equal(inputs[0], inputs[2])
    assert one.conditioning_text == future.conditioning_text


def test_multitoken_pointer_feedback_starts_after_complete_span():
    torch.manual_seed(11)
    sequence = _collator()(_record())
    first, second = sequence.pointer_positions
    span = sequence.token_spans[first.command_position]
    assert span.end - span.start == 3
    assert first.model_position == span.start
    assert sequence.feedback_positions[0] == span.end

    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=4, use_native_brep=False).eval()
    ids = sequence.input_ids[:sequence.action_boundaries[0].end].unsqueeze(0)
    mask = torch.ones_like(ids)

    def run(first_index):
        return model.forward_ragged(
            input_ids=ids, attention_mask=mask, states=[_state()],
            pointer_specs=[[{
                "position": first.model_position, "pointer_type": PointerType.BODY,
                "target_index": first_index, "feedback_position": sequence.feedback_positions[0],
            }, {
                "position": second.model_position, "pointer_type": PointerType.BODY,
                "target_index": 0, "feedback_position": sequence.feedback_positions[1],
            }]],
        )

    left, right = run(0), run(1)
    for position in range(span.start, span.end):
        torch.testing.assert_close(left.hidden_states[0, position], right.hidden_states[0, position])
    torch.testing.assert_close(left.pointer_logits_by_example[0][0], right.pointer_logits_by_example[0][0])
    assert not torch.allclose(left.hidden_states[0, span.end], right.hidden_states[0, span.end])
    assert not torch.allclose(left.pointer_logits_by_example[0][1], right.pointer_logits_by_example[0][1])


def test_inference_waits_for_span_end_in_cached_and_uncached_paths():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    torch.manual_seed(17)
    sequence = _collator()(_record())
    state = _state()
    base = Qwen2Model(Qwen2Config(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
    ))
    model = CRSExpandedPointerCAD(base_model=base, grammar_vocab_size=4, use_native_brep=False).eval()
    ids = sequence.input_ids[:sequence.action_boundaries[0].end].unsqueeze(0)
    mask = torch.ones_like(ids)
    positions = [item.model_position for item in sequence.pointer_positions]
    kwargs = dict(pointer_positions=positions, pointer_types=[PointerType.BODY] * 2,
                  feedback_positions=sequence.feedback_positions, return_logits=True)
    cached, cached_logits = model.inference_pointer_decode(ids, mask, state, use_cache=True, **kwargs)
    reference, reference_logits = model.inference_pointer_decode(ids, mask, state, use_cache=False, **kwargs)
    assert [item[2] for item in cached] == [item[2] for item in reference]
    for actual, expected in zip(cached_logits, reference_logits):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_prepared_frozen_record_exposes_exact_source_text(tmp_path):
    frozen, prepared = tmp_path / "frozen", tmp_path / "prepared"
    frozen.mkdir()
    prepared.mkdir()
    manifest = b'{"format":"stage2-clean-validation-v1","entries":[]}'
    (frozen / "manifest.json").write_bytes(manifest)
    identity = ["zero2cad100k-train", "00002", "000", "v000"]
    text = "Construct the CAD model."
    (prepared / "manifest.json").write_text(json.dumps({
        "format": "stage2a3-native-input-v1",
        "frozen_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "entries": [{"identity": identity, "steps": [], "conditioning": {
            "text": text, "source": "legacy_pointercad_adapter_constant",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }}],
    }))
    corpus = PreparedStage2Corpus(frozen, prepared)
    record = FrozenStage2Record(tuple(identity), {}, _record(text))
    sequence = corpus.collate_record(record, _collator())
    assert sequence.conditioning_text == text
    assert sequence.conditioning_source == "legacy_pointercad_adapter_constant"
    assert sequence.conditioning_end > 0

    stale = json.loads((prepared / "manifest.json").read_text())
    stale["entries"][0]["conditioning"]["sha256"] = "0" * 64
    (prepared / "manifest.json").write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="conditioning"):
        PreparedStage2Corpus(frozen, prepared).collate_record(record, _collator())


def test_missing_conditioning_fails_instead_of_training_without_x():
    record = _record()
    del record["conditioning"]
    with pytest.raises(ValueError, match="conditioning X"):
        _collator()(record)
