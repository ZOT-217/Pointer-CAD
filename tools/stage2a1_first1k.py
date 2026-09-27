"""Run the Stage 2A.1 causal supervision adapter over CRS artifacts.

Run this from an environment that has both cadquery2crs and Pointer-CAD's
PyTorch dependencies installed. The script never reads old sidecar banks.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cadquery2crs.crs import document_from_dict
from cadquery2crs.runtime import ExecutionTrace
from cadquery2crs.supervision import compile_causal_supervision
from cadquery2crs.toolkit import CRSProgram

from models.crs_pointercad.materialization import materialize_causal_supervision


def audit(root: Path, *, limit: int = 0) -> dict:
    files = sorted((root / "crs").glob("*/000.json"))
    if limit:
        files = files[:limit]
    totals = Counter()
    failures = []
    for path in files:
        try:
            document = document_from_dict(json.loads(path.read_text()))
            program = CRSProgram.from_json(document.to_dict())
            supervision = compile_causal_supervision(program.to_command(), document)
            trace = ExecutionTrace.from_document(document, causal_candidate_banks=True)
            result = materialize_causal_supervision(supervision, trace)
            totals["pointer_positions"] += result["pointer_positions"]
            totals["candidate_banks"] += result["candidate_banks"]
            totals["unresolved_targets"] += result["unresolved_targets"]
            if result["failures"]:
                failures.append({"sample_id": path.parent.name, "failures": result["failures"]})
        except Exception as exc:
            failures.append({"sample_id": path.parent.name, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "population": "FIRST_1K_DISCOVERY_ONLY",
        "samples_attempted": len(files),
        **dict(totals),
        "model_materialization_failures": len(failures),
        "failures": failures,
        "old_sidecar_banks_used": False,
        "status": "PASS" if not failures else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("../cadquery2crs/outputs/current-1k-certification"))
    parser.add_argument("--output", type=Path, default=Path("docs/audits/2026-09-27-pointercad-stage2a1/first1k_stage2_materialization.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    result = audit(args.root, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
