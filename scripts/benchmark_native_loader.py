"""Measure one backend's prepared action loader in an isolated process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from time import perf_counter

from models.crs_pointercad.training import PreparedStage2Corpus


def benchmark(frozen: Path, prepared: Path, backend: str):
    corpus = PreparedStage2Corpus(frozen, prepared, native_backend=backend)
    started = perf_counter()
    actions = 0
    for identity, entry in sorted(corpus.entries.items()):
        for step in entry["steps"]:
            corpus.state_for(identity, step["action_index"])
            actions += 1
    seconds = perf_counter() - started
    return {"backend": backend, "records": len(corpus.entries), "actions": actions,
            "loader_wall_seconds": seconds, "actions_per_second": actions / seconds,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backend", choices=("v1", "v2"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.frozen, args.prepared, args.backend)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
