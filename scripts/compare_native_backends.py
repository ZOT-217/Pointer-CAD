"""Compare V1 and V2 at the prepared action and candidate-bank boundary."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from collections.abc import Mapping
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from models.crs_pointercad.candidates import CandidateView
from models.crs_pointercad.contracts import PointerType
from models.crs_pointercad.training import PreparedStage2Corpus, _target_key
from models.crs_pointercad.sequence import Stage2QwenCollator, frozen_grammar_vocabulary


def _plain(value):
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in vars(value).items()}
    return value


def compare_action(v1, v2, identity, action_index, supervision):
    """Return first semantic mismatch or per-action counts."""
    state_a, brep_a = v1.state_for(identity, action_index)
    state_b, brep_b = v2.state_for(identity, action_index)

    def mismatch(field, left, right):
        return {"identity": list(identity), "action_index": action_index, "field": field,
                "v1": _plain(left), "v2": _plain(right)}

    step_a = next(item for item in v1.entries[identity]["steps"] if item["action_index"] == action_index)
    step_b = next(item for item in v2.entries[identity]["steps"] if item["action_index"] == action_index)
    data_a = json.loads((v1.prepared_root / step_a["path"] / "state.json").read_text())
    data_b = json.loads((v2.prepared_root / step_b["path"] / "state.json").read_text())
    if data_a != data_b:
        return mismatch("state", data_a, data_b)
    if v1.conditioning_for(identity) != v2.conditioning_for(identity):
        return mismatch("conditioning", v1.conditioning_for(identity), v2.conditioning_for(identity))
    for field in ("face_keys", "edge_keys", "loose_edge_keys", "face_adjacency"):
        if getattr(brep_a, field) != getattr(brep_b, field):
            return mismatch(field, getattr(brep_a, field), getattr(brep_b, field))
    for field in ("face_features", "edge_features", "loose_edge_features"):
        left, right = getattr(brep_a, field), getattr(brep_b, field)
        if left.shape != right.shape or left.dtype != right.dtype:
            return mismatch(field + ".shape_dtype", [left.shape, str(left.dtype)], [right.shape, str(right.dtype)])
        if not np.allclose(left, right, rtol=0, atol=1e-7, equal_nan=False):
            difference = np.abs(left - right)
            index = np.unravel_index(np.nanargmax(difference), difference.shape)
            return mismatch(field + f"{index}", float(left[index]), float(right[index]))
    for field in ("num_nodes", "edges"):
        left = brep_a.graph.num_nodes() if field == "num_nodes" else tuple(x.tolist() for x in brep_a.graph.edges())
        right = brep_b.graph.num_nodes() if field == "num_nodes" else tuple(x.tolist() for x in brep_b.graph.edges())
        if left != right:
            return mismatch("graph." + field, left, right)
    if not np.array_equal(brep_a.graph.edata["x"].numpy(), brep_b.graph.edata["x"].numpy()):
        return mismatch("graph.reverse_edge_features", "different", "different")
    slots = [item for item in supervision["pointer_targets"] if item["action_index"] == action_index]
    rows = 0
    for pointer_type in PointerType:
        left = CandidateView(state_a).bank(pointer_type)
        right = CandidateView(state_b).bank(pointer_type)
        left_keys = [str(item.external_key) for item in left]
        right_keys = [str(item.external_key) for item in right]
        rows += len(left)
        if left_keys != right_keys:
            return mismatch(f"bank.{pointer_type.value}.keys", left_keys, right_keys)
        for item_a, item_b in zip(left, right):
            if _plain(item_a.semantic) != _plain(item_b.semantic) or _plain(dict(item_a.legal_metadata)) != _plain(dict(item_b.legal_metadata)):
                return mismatch(f"bank.{pointer_type.value}.metadata", str(item_a), str(item_b))
    for slot in slots:
        pointer_type = PointerType("PTR_" + slot["target_candidate"]["kind"])
        frozen_bank = supervision["candidate_banks"][slot["candidate_bank"]]
        expected_keys = [_target_key(key) for key in frozen_bank["candidates"]]
        if slot["target_index"] >= len(expected_keys) or expected_keys[slot["target_index"]] != _target_key(slot["target_candidate"]):
            return mismatch("slot.frozen_target_index", slot["target_index"],
                            [str(key) for key in expected_keys])
        left = CandidateView(state_a).bank(pointer_type, slot.get("decoder_substate", {}))
        right = CandidateView(state_b).bank(pointer_type, slot.get("decoder_substate", {}))
        keys_a = [str(item.external_key) for item in left]
        keys_b = [str(item.external_key) for item in right]
        if keys_a != keys_b:
            return mismatch("slot.masked_bank", keys_a, keys_b)
        target = _target_key(slot["target_candidate"])
        positive_a = left.positive_mask((target,)).tolist()
        positive_b = right.positive_mask((target,)).tolist()
        if positive_a != positive_b or sum(positive_a) != 1:
            return mismatch("slot.positive_mask", positive_a, positive_b)
    return {"pointer_slots": len(slots), "candidate_rows": rows}


_worker_corpora = None


class _ParityFixtureTokenizer:
    """Deterministic tokenizer for structural collator parity, not production IDs."""

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        if tokenize or not add_generation_prompt:
            raise ValueError("unexpected parity fixture chat-template call")
        text = next(item["text"] for item in messages[1]["content"] if item["type"] == "text")
        return f"<|im_start|>system\n{messages[0]['content']}<|im_end|><|im_start|>user\n{text}<|im_end|><|im_start|>assistant\n<|cad_start|>"

    def __call__(self, value, *, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("fixture tokenizer expects no special tokens")
        return {"input_ids": [64 + ord(character) % 64 for character in value]}


def _init_worker(frozen_root, v1_root, v2_root, tokenizer_path=None, fixture_tokenizer=False):
    global _worker_corpora
    collator = None
    if tokenizer_path is not None or fixture_tokenizer:
        entries = json.loads((Path(frozen_root) / "manifest.json").read_text())["entries"]
        records = [json.loads((Path(frozen_root) / item["supervision_artifact"]).read_text()) for item in entries]
        grammar = frozen_grammar_vocabulary(records)
        if fixture_tokenizer:
            tokenizer = _ParityFixtureTokenizer()
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        collator = Stage2QwenCollator(tokenizer, grammar_vocabulary=grammar)
    _worker_corpora = (PreparedStage2Corpus(frozen_root, v1_root),
                       PreparedStage2Corpus(frozen_root, v2_root, native_backend="v2"), Path(frozen_root), collator)


def _compare_record(identity):
    v1, v2, frozen_root, collator = _worker_corpora
    entry = v1.entries[identity]
    frozen = json.loads((frozen_root / "manifest.json").read_text())
    source = next(item for item in frozen["entries"] if identity == tuple(item[key] for key in
                  ("dataset", "sample_id", "source_variant_id", "approved_variant_id")))
    supervision = json.loads((frozen_root / source["supervision_artifact"]).read_text())
    result = {"actions": 0, "pointer_slots": 0, "candidate_rows": 0, "first_mismatch": None}
    if collator is not None:
        record = SimpleNamespace(identity=identity, supervision=supervision)
        try:
            sequence_a = _plain(v1.collate_record(record, collator))
            sequence_b = _plain(v2.collate_record(record, collator))
        except Exception as exc:
            result["first_mismatch"] = {"identity": list(identity), "field": "sequence.materialization",
                                        "error_type": type(exc).__name__, "message": str(exc),
                                        "comparison_status": "BLOCKED"}
            return result
        if sequence_a != sequence_b:
            for field in sequence_a:
                if sequence_a[field] != sequence_b[field]:
                    result["first_mismatch"] = {"identity": list(identity), "field": f"sequence.{field}",
                                                "v1": sequence_a[field], "v2": sequence_b[field]}
                    return result
    for step in entry["steps"]:
        compared = compare_action(v1, v2, identity, step["action_index"], supervision)
        if "field" in compared:
            result["first_mismatch"] = compared
            break
        result["actions"] += 1
        result["pointer_slots"] += compared["pointer_slots"]
        result["candidate_rows"] += compared["candidate_rows"]
    return result


def compare_corpora(frozen_root, v1_root, v2_root, *, workers=1, tokenizer_path=None, fixture_tokenizer=False):
    v1 = PreparedStage2Corpus(frozen_root, v1_root)
    v2 = PreparedStage2Corpus(frozen_root, v2_root, native_backend="v2")
    report = {"records": 0, "actions": 0, "pointer_slots": 0, "candidate_rows": 0,
              "parity_mismatches": 0, "first_mismatch": None, "comparison_blockers": 0, "first_blocker": None,
              "sequence_parity_mode": "fixture" if fixture_tokenizer else "production" if tokenizer_path else "artifact_equivalence"}
    if set(v1.entries) != set(v2.entries):
        report.update(parity_mismatches=1, first_mismatch={"field": "record_identities",
                                                       "v1_only": [list(x) for x in set(v1.entries) - set(v2.entries)],
                                                       "v2_only": [list(x) for x in set(v2.entries) - set(v1.entries)]})
        return report
    tasks = []
    for identity in sorted(v1.entries):
        a, b = v1.entries[identity], v2.entries[identity]
        if a["crs_sha256"] != b["crs_sha256"] or a["supervision_sha256"] != b["supervision_sha256"]:
            report.update(parity_mismatches=1, first_mismatch={"identity": list(identity), "field": "frozen_artifact_sha256"})
            return report
        actions_a = [step["action_index"] for step in a["steps"]]
        actions_b = [step["action_index"] for step in b["steps"]]
        if actions_a != actions_b:
            report.update(parity_mismatches=1, first_mismatch={"identity": list(identity), "field": "action_indices",
                                                         "v1": actions_a, "v2": actions_b})
            return report
        tasks.append(identity)
    if workers == 1:
        _init_worker(frozen_root, v1_root, v2_root, tokenizer_path, fixture_tokenizer)
        results = map(_compare_record, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                   initializer=_init_worker, initargs=(frozen_root, v1_root, v2_root,
                                                                        tokenizer_path, fixture_tokenizer))
        results = pool.map(_compare_record, tasks)
    try:
        for result in results:
            report["records"] += 1
            if result["first_mismatch"] is not None:
                if result["first_mismatch"].get("comparison_status") == "BLOCKED":
                    report.update(comparison_blockers=1, first_blocker=result["first_mismatch"])
                else:
                    report.update(parity_mismatches=1, first_mismatch=result["first_mismatch"])
                return report
            for key in ("actions", "pointer_slots", "candidate_rows"):
                report[key] += result[key]
    finally:
        if workers != 1:
            pool.shutdown(cancel_futures=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--fixture-tokenizer", action="store_true")
    args = parser.parse_args()
    if args.fixture_tokenizer and args.tokenizer:
        parser.error("choose either --fixture-tokenizer or --tokenizer")
    report = compare_corpora(args.frozen, args.v1, args.v2, workers=args.workers,
                             tokenizer_path=args.tokenizer, fixture_tokenizer=args.fixture_tokenizer)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(bool(report["parity_mismatches"] or report["comparison_blockers"]))


if __name__ == "__main__":
    main()
