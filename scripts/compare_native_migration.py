"""Compare unreleased V1 and local V2 migration records at the Stage2 boundary.

Normal PreparedStage2Corpus loading stays release-only. Both migration inputs
must be explicitly supplied, and every selected action is compared in order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from models.crs_pointercad.sequence import Stage2QwenCollator, frozen_grammar_vocabulary
from models.crs_pointercad.training import PreparedStage2Corpus
from scripts.compare_native_backends import _ParityFixtureTokenizer, _plain, compare_action


FIELDS = ("dataset", "sample_id", "source_variant_id", "approved_variant_id")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def compare(frozen_root: Path, v1_root: Path, v2_root: Path, v1_manifest: Path,
            v2_manifest: Path, selection: Path, output: Path, *, resume: bool = False) -> dict:
    frozen_root, v1_root, v2_root = (path.resolve() for path in (frozen_root, v1_root, v2_root))
    v1_manifest, v2_manifest, selection, output = (path.resolve() for path in
                                                   (v1_manifest, v2_manifest, selection, output))
    if not output.is_relative_to(Path("/tmp").resolve()):
        raise ValueError("migration comparison output must be local /tmp scratch")
    selected = json.loads(selection.read_text())
    identities = [tuple(row["identity"]) for row in selected["entries"]]
    if selected.get("selected_count") != len(identities) or len(set(identities)) != len(identities):
        raise ValueError("Level C selection has duplicate or miscounted identities")
    identities.sort()
    v1 = PreparedStage2Corpus(frozen_root, v1_root, migration_manifest=v1_manifest)
    v2 = PreparedStage2Corpus(frozen_root, v2_root, native_backend="v2", migration_manifest=v2_manifest)
    if set(v1.entries) != set(identities) or set(v2.entries) != set(identities):
        raise ValueError("migration manifests must contain exactly the Level C identities")
    source_manifest = json.loads((frozen_root / "manifest.json").read_text())
    source = {tuple(row[field] for field in FIELDS): row["supervision_artifact"]
              for row in source_manifest["entries"] if tuple(row[field] for field in FIELDS) in v1.entries}
    del source_manifest
    if set(source) != set(identities):
        raise ValueError("selected identity is absent from frozen corpus")
    supervisions = {identity: json.loads((frozen_root / source[identity]).read_text()) for identity in identities}
    tokens = set()
    for supervision in supervisions.values():
        tokens.update(frozen_grammar_vocabulary([supervision]))
    collator = Stage2QwenCollator(_ParityFixtureTokenizer(),
                                 grammar_vocabulary={token: index for index, token in enumerate(sorted(tokens))})
    config = {"frozen_manifest_sha256": _digest(frozen_root / "manifest.json"),
              "v1_migration_manifest_sha256": _digest(v1_manifest),
              "v2_migration_manifest_sha256": _digest(v2_manifest),
              "selection_sha256": _digest(selection),
              "comparator_sha256": _digest(Path(__file__)),
              "mode": "fixture_tokenizer_plus_exact_prepared_action",
              "expected_records": len(identities)}
    config_path = output.with_suffix(".config.json")
    progress_path = output.with_suffix(".progress.jsonl")
    if resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise ValueError("parity resume config/source differs from its completed records")
    else:
        if output.exists() or progress_path.exists() or config_path.exists():
            raise FileExistsError("parity output exists; pass --resume only for the same contract")
        _write_json_atomic(config_path, config)
    completed = {}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            item = json.loads(line)
            identity = tuple(item["identity"])
            if identity in completed or identity not in identities:
                raise ValueError("parity progress contains duplicate or foreign identity")
            completed[identity] = item
    report = {"verdict": "IN_PROGRESS", "expected_records": len(identities),
              "records": len(completed), "actions": sum(item["actions"] for item in completed.values()),
              "pointer_slots": sum(item["pointer_slots"] for item in completed.values()),
              "candidate_rows": sum(item["candidate_rows"] for item in completed.values()),
              "parity_mismatches": 0, "first_mismatch": None, "comparison_blockers": 0,
              "first_blocker": None, "tokenizer_mode": "fixture",
              "tensor_policy": "np.allclose(rtol=0, atol=1e-7, equal_nan=False)",
              "source_backend": "v1_unreleased_individually_validated", "migration_mode": True}
    for identity in identities:
        if identity in completed:
            continue
        left, right = v1.entries[identity], v2.entries[identity]
        if (left["crs_sha256"] != right["crs_sha256"] or
                left["supervision_sha256"] != right["supervision_sha256"]):
            report["first_mismatch"] = {"identity": list(identity), "field": "frozen_artifact_sha256"}
            break
        actions_a = [step["action_index"] for step in left["steps"]]
        actions_b = [step["action_index"] for step in right["steps"]]
        if actions_a != actions_b:
            report["first_mismatch"] = {"identity": list(identity), "field": "action_indices",
                                        "v1": actions_a, "v2": actions_b}
            break
        supervision = supervisions[identity]
        record = SimpleNamespace(identity=identity, supervision=supervision)
        try:
            sequence_a = _plain(v1.collate_record(record, collator))
            sequence_b = _plain(v2.collate_record(record, collator))
        except Exception as exc:
            report["first_blocker"] = {"identity": list(identity), "field": "sequence.materialization",
                                       "error_type": type(exc).__name__, "message": str(exc)}
            break
        if sequence_a != sequence_b:
            field = next((key for key in sequence_a if sequence_a[key] != sequence_b.get(key)), "sequence")
            report["first_mismatch"] = {"identity": list(identity), "field": f"sequence.{field}"}
            break
        record_result = {"identity": list(identity), "actions": 0, "pointer_slots": 0, "candidate_rows": 0}
        for action_index in actions_a:
            try:
                result = compare_action(v1, v2, identity, action_index, supervision)
            except Exception as exc:
                report["first_blocker"] = {"identity": list(identity), "action_index": action_index,
                                           "field": "state_for", "error_type": type(exc).__name__,
                                           "message": str(exc)}
                break
            if "field" in result:
                report["first_mismatch"] = result
                break
            record_result["actions"] += 1
            record_result["pointer_slots"] += result["pointer_slots"]
            record_result["candidate_rows"] += result["candidate_rows"]
        if report["first_mismatch"] is not None or report["first_blocker"] is not None:
            break
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record_result, sort_keys=True) + "\n")
        completed[identity] = record_result
        report["records"] += 1
        report["actions"] += record_result["actions"]
        report["pointer_slots"] += record_result["pointer_slots"]
        report["candidate_rows"] += record_result["candidate_rows"]
        if report["records"] % 10 == 0 or report["records"] == len(identities):
            print(json.dumps({"compared_records": report["records"],
                              "expected_records": len(identities), "actions": report["actions"]}), flush=True)
    report["parity_mismatches"] = int(report["first_mismatch"] is not None)
    report["comparison_blockers"] = int(report["first_blocker"] is not None)
    report["verdict"] = ("PASS" if report["records"] == len(identities) and
                         not report["parity_mismatches"] and not report["comparison_blockers"] else
                         "MISMATCH" if report["parity_mismatches"] else "BLOCKED" if report["comparison_blockers"] else
                         "INCOMPLETE")
    _write_json_atomic(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = compare(args.frozen, args.v1, args.v2, args.v1_manifest, args.v2_manifest,
                     args.selection, args.output, resume=args.resume)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(report["verdict"] != "PASS")


if __name__ == "__main__":
    main()
