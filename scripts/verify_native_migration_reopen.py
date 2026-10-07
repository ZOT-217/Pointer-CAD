"""Deterministically reopen every local Native V2 migration record.

This is a migration check, not a corpus release check. The first, middle, and
last action of each record are loaded through PreparedStage2Corpus.state_for.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback

from models.crs_pointercad.training import PreparedStage2Corpus


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def sampled_actions(steps: list[dict]) -> tuple[int, ...]:
    indices = [step["action_index"] for step in steps]
    if indices != list(range(len(indices))) or not indices:
        raise ValueError("migration entry has noncontiguous or empty actions")
    return tuple(sorted({indices[0], indices[len(indices) // 2], indices[-1]}))


def verify(frozen: Path, v2: Path, migration_manifest: Path, output: Path, *, resume: bool = False) -> dict:
    frozen, v2, migration_manifest, output = (path.resolve() for path in
                                              (frozen, v2, migration_manifest, output))
    if not output.is_relative_to(Path("/tmp").resolve()):
        raise ValueError("migration reopen report must be local /tmp scratch")
    manifest = json.loads(migration_manifest.read_text())
    if (manifest.get("format") != "stage2a3-native-migration-v2" or
            manifest.get("source_backend") != "v1_unreleased_individually_validated" or
            manifest.get("migration_mode") is not True):
        raise ValueError("reopen requires an unreleased V1-to-V2 migration manifest")
    corpus = PreparedStage2Corpus(frozen, v2, native_backend="v2",
                                  migration_manifest=migration_manifest)
    entries = corpus.entries
    config = {"frozen_manifest_sha256": _digest(frozen / "manifest.json"),
              "migration_manifest_sha256": _digest(migration_manifest),
              "verifier_sha256": _digest(Path(__file__)),
              "sampling": "first_middle_last_per_record", "expected_successes": len(entries)}
    config_path = output.with_suffix(".config.json")
    progress_path = output.with_suffix(".progress.jsonl")
    if resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise ValueError("reopen resume contract differs from completed records")
    else:
        if output.exists() or config_path.exists() or progress_path.exists():
            raise FileExistsError("reopen report exists; pass --resume for the same migration")
        _write_atomic(config_path, config)
    completed = {}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            item = json.loads(line)
            identity = tuple(item["identity"])
            if identity in completed or identity not in entries:
                raise ValueError("reopen progress contains duplicate or foreign identity")
            completed[identity] = item
    for ordinal, identity in enumerate(sorted(entries), 1):
        if identity in completed:
            continue
        result = {"identity": list(identity), "sampled_action_indices": (),
                  "actions_reopened": 0, "status": "PASS"}
        try:
            action_indices = sampled_actions(entries[identity]["steps"])
            result["sampled_action_indices"] = action_indices
            for index in action_indices:
                state, brep = corpus.state_for(identity, index)
                if state.action_index != index or brep.graph.num_nodes() != len(brep.face_keys):
                    raise ValueError("reopened action index or FACE graph node count differs")
                result["actions_reopened"] += 1
                del state, brep
        except Exception as exc:
            result.update(status="FAILED", error_type=type(exc).__name__, message=str(exc),
                          traceback=traceback.format_exc())
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        completed[identity] = result
        if ordinal % 100 == 0 or ordinal == len(entries):
            print(json.dumps({"processed": ordinal, "expected": len(entries),
                              "failed": sum(item["status"] != "PASS" for item in completed.values())}),
                  flush=True)
        if ordinal % 100 == 0:
            gc.collect()
    failures = [item for item in completed.values() if item["status"] != "PASS"]
    report = {"verdict": "PASS" if not failures and len(completed) == len(entries) and
              manifest.get("status") == "COMPLETE" else "BLOCKED",
              "mode": "migration_reopen_not_corpus_release",
              "sample_policy": "first_middle_last_per_record", "expected_records": len(entries),
              "records_reopened": len(completed) - len(failures),
              "actions_reopened": sum(item["actions_reopened"] for item in completed.values()),
              "failures": failures, "migration_status": manifest.get("status")}
    _write_atomic(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--migration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = verify(args.frozen, args.v2, args.migration_manifest, args.output, resume=args.resume)
    print(json.dumps({"verdict": report["verdict"], "records_reopened": report["records_reopened"],
                      "actions_reopened": report["actions_reopened"],
                      "failures": len(report["failures"])}, sort_keys=True))
    raise SystemExit(report["verdict"] != "PASS")


if __name__ == "__main__":
    main()
