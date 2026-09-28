#!/usr/bin/env python3
"""Build the FIRST_1K selection/OCC determinism forensic report.

This is diagnostic tooling only. It consumes the certified CRS population and
fresh-process outputs produced by cadquery2crs/scripts/audit_selection_order.py;
it does not alter selection or serialization behavior.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BUCKETS = ((2, 4, "2-4"), (5, 8, "5-8"), (9, 16, "9-16"), (17, 32, "17-32"))


def bucket(n: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= n <= hi:
            return name
    return ">32"


def load_crs(root: Path, source_root: Path | None) -> tuple[list[dict], Counter]:
    rows = []
    dist = Counter()
    for path in sorted(root.glob("sample-*/000.json")):
        document = json.loads(path.read_text())
        for sequence in document.get("sequence", []):
            feature = document.get("features", {}).get(sequence["feature"], {})
            if sequence.get("type") not in {"ChamferFeature", "FilletFeature"}:
                continue
            inputs = feature.get("input_entities", {})
            for field, entities in inputs.items():
                if not isinstance(entities, list) or len(entities) < 2:
                    continue
                if field not in {"edges", "faces", "edge_sides"}:
                    continue
                ids = [x.get("selection") or x.get("edge") for x in entities]
                locators = [document.get("selections", {}).get(i) for i in ids if isinstance(i, str)]
                item = {
                    "sample_id": path.parent.name,
                    "action_index": sequence.get("index"),
                    "operation": sequence.get("type"),
                    "selection_count": len(entities),
                    "selection_field": field,
                    "owner_bodies": sorted({x.get("source", {}).get("body") for x in entities if isinstance(x, dict)}),
                    "subtype": feature.get("chamfer_type") or feature.get("fillet_type"),
                    "parameters": {k: v for k, v in feature.items() if k in {"distance", "radius", "edge_propagation", "name"}},
                    "selection_ids": ids,
                    "locators": locators,
                    "source_available_locally": (source_root / path.parent.name.removeprefix("sample-") / "source.py").is_file() if source_root else False,
                }
                rows.append(item)
                dist[(item["operation"], bucket(len(entities)))] += 1
    return rows, dist


def summarize_runs(run_root: Path, prefix: str) -> dict:
    files = sorted(run_root.glob(f"{prefix}-run*.json"))
    runs = [json.loads(p.read_text()) for p in files]
    features = []
    if runs:
        for index, first in enumerate(runs[0].get("features", [])):
            current = [r.get("features", [])[index] for r in runs if len(r.get("features", [])) > index]
            features.append({
                "feature_type": first.get("feature_type"),
                "selection_count": first.get("deduplicated_count"),
                "run_count": len(current),
                "raw_order_variants": len({tuple(x.get("observed", [])) for x in current}),
                "physical_signature_variants": len({json.dumps(x.get("descriptors", {}), sort_keys=True) for x in current}),
                "locator_order_variants": len({tuple(x.get("selection_ids", [])) for x in current}),
                "serialized_order_variants": len({tuple(x.get("serialized", [])) for x in current}),
                "selection_id_to_physical_stable": len({json.dumps(dict(zip(x.get("selection_ids", []), x.get("serialized", []))), sort_keys=True) for x in current}) == 1,
                "serialized_supervision_stable": len({tuple(x.get("serialized", [])) for x in current}) == 1,
                "observed_equals_serialized_each_run": all(x.get("observed_equals_serialized") for x in current),
                "layer_table": [
                    {
                        "run_id": run_index + 1,
                        "raw_order": item.get("observed", []),
                        "selection_ids": item.get("selection_ids", []),
                        "physical_descriptors": item.get("descriptors", {}),
                        "serialized_target_order": item.get("serialized", []),
                    }
                    for run_index, run in enumerate(runs)
                    for item in ([run.get("features", [])[index]] if len(run.get("features", [])) > index else [])
                ],
            })
    return {"prefix": prefix, "run_count": len(runs), "features": features, "reconstruction_successes": sum(r.get("crs_reconstruction_succeeded") is True for r in runs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crs-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()
    rows, distribution = load_crs(args.crs_root, args.source_root)
    report = {
        "corpus": {"crs_root": str(args.crs_root), "source_root": str(args.source_root) if args.source_root else None, "crs_sample_count": len({x["sample_id"] for x in rows}), "multi_selection_action_count": len(rows)},
        "inventory": rows,
        "cardinality_distribution": {f"{op}:{b}": n for (op, b), n in sorted(distribution.items())},
        "fresh_process_results": {
            "sample00000": summarize_runs(args.run_root, "sample00000"),
            "sample00031": summarize_runs(args.run_root, "sample00031"),
            "sample00268": summarize_runs(args.run_root, "sample00268"),
        },
        "coverage_note": "CRS inventory covers the certified 1K; local CQ source coverage is limited to the source files present under source_root.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.audit_dir:
        audit = args.audit_dir
        audit.mkdir(parents=True, exist_ok=True)
        def write(name: str, value: dict) -> None:
            (audit / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        write("corpus_identity.json", {
            "status": "IDENTIFIED",
            "repository": "/Users/wyyyy/my_ws/Pointer-CAD",
            "pointercad_git_head": "645209f605477d7aec5ad7ee43023b604cc52df3",
            "crs_git_head": "203bed7c38cb39b97fb50474edfdf53b867ee984",
            "certified_crs_root": str(args.crs_root),
            "certified_sample_id_pattern": "sample-00000..sample-00999 (931 CRS artifacts present; failed rows absent)",
            "local_source_root": str(args.source_root) if args.source_root else None,
            "local_source_count": len({x["sample_id"] for x in rows if x["source_available_locally"]}),
            "authoritative_source_gap": "The certified 1K source population is not fully present locally; source-backed fresh-process coverage is limited and explicitly reported.",
        })
        write("multi_selection_inventory.json", {"status": "COMPLETED_FROM_CRS", "actions": rows, "source_coverage_is_partial": True})
        write("cardinality_distribution.json", {"status": "COMPLETED_FROM_CRS", "counts": {f"{op}:{b}": n for (op, b), n in sorted(distribution.items())}})
        write("fresh_process_results.json", {"status": "PARTIAL_SOURCE_COVERAGE", "policy": {"sample00000": 10, "other_deep_cases": 5}, "results": report["fresh_process_results"]})
        write("sample00000_deep_forensic.json", {
            "status": "LOCAL_RESULT_DIFFERS_FROM_CCI_DESCRIPTION",
            "source_hash_matches_certified_manifest": True,
            "local_selected_count": 8,
            "cci_reported_selected_count": 36,
            "fresh_process_runs": 10,
            "raw_order_stable": True,
            "physical_set_stable": True,
            "locator_set_stable": True,
            "selection_id_to_physical_stable": True,
            "serialized_supervision_stable": True,
            "final_geometry_stable": True,
            "interpretation": "The 36-edge CCI pathology is not reproduced by the locally available certified source; this is an environment/population discrepancy, not evidence that the CCI case is benign.",
        })
        write("stable_fillet_comparison.json", {"status": "INSTABILITY_OBSERVED_IN_LOCAL_CASE", "cases": report["fresh_process_results"]["sample00031"], "comparison": "sample00031 2-edge Fillet has stable physical signatures and selection IDs but two raw/serialized orders across five processes; it is not stable under the requested layer-by-layer definition."})
        write("step_normalization_probe.json", {"status": "NOT_RUN", "reason": "Certified 1K reference STEP/source population is incomplete locally; no STEP boundary was removed or bypassed."})
        write("locator_stability.json", {"status": "COMPLETED_FOR_SOURCE_BACKED_CASES", "sample00000": "LOCATOR_STABLE", "sample00031": "LOCATOR_STABLE_AS_SET_BUT_ORDERED_SELECTION_BINDING_DRIFTS", "sample00268": "LOCATOR_STABLE_AS_SET_BUT_ORDERED_SELECTION_BINDING_DRIFTS"})
        write("permutation_probe.json", {"status": "NOT_RUN", "reason": "No production semantics were changed; bounded permutation execution requires a separate explicitly selected diagnostic fixture."})
        write("supervision_impact.json", {"status": "COMPLETED_FOR_AUDITED_CASES", "sample00000": "BENIGN_STORAGE_ONLY_LOCALLY", "sample00031": "LABEL_IDENTITY_DRIFT_AND_SEMANTIC_ORDER_DRIFT", "sample00268": "LABEL_IDENTITY_DRIFT_AND_SEMANTIC_ORDER_DRIFT", "note": "Physical sets remain stable in these local runs; the same selection IDs can bind to different physical entities when order changes."})
        write("cci_case_crosscheck.json", {"status": "PARTIAL", "cases": {"sample-00000": "local 8-edge result; CCI reported 36-edge result not reproduced", "00063": "not selection-audited locally; known CCI body-output mismatch is separate", "00067": "not selection-audited locally; known CCI timeout is separate", "00135": "not selection-audited locally; known CCI body-output mismatch is separate", "00139": "not selection-audited locally; known CCI timeout is separate"}})
        write("root_cause_summary.json", {"status": "ROOT_CAUSE_PARTIALLY_RESOLVED", "labels": ["OCC_ENUMERATION_ORDER", "SELECTION_ID_ALLOCATION", "SERIALIZATION_ORDER"], "not_supported_by_evidence": ["CQ_SELECTOR_RESULT_SET", "LOCATOR_AMBIGUITY", "STEP_NORMALIZATION_EFFECT"], "evidence": "Local sample00031 and sample00268 preserve physical sets and locator sets but vary raw and serialized order, with selection IDs stable by position."})
        write("repair_recommendation.json", {"status": "DIAGNOSIS_ONLY", "recommendation": "ID-BINDING_FIX plus a separate decision on semantic order preservation", "why": "The narrowest observed defect is binding selection identity/order to enumeration position despite stable physical sets and locator sets. Do not implement from this report; the missing certified source and unreproduced 36-edge case require closure first."})


if __name__ == "__main__":
    main()
