#!/usr/bin/env python3
"""Summarize structured expanded-9K training logs as JSON and Markdown."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--json', type=Path); p.add_argument('--markdown', type=Path)
    a = p.parse_args(); rows = [json.loads(line) for line in (a.run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    train = [r for r in rows if r.get('split') == 'train']; failures = [r for r in rows if r.get('status') == 'failure']
    result = {'records': len(rows), 'train_records': len(train), 'failures': failures,
              'final_loss': train[-1].get('total_loss') if train else None,
              'best_loss': min((r.get('total_loss') for r in train if isinstance(r.get('total_loss'), (int,float))), default=None),
              'final_components': {k: train[-1].get(k) for k in ('L_g','L_p','L_s','L_r')} if train else {},
              'throughput': {'examples_per_second': train[-1].get('examples_per_second') if train else None, 'optimizer_steps_per_second': train[-1].get('optimizer_steps_per_second') if train else None},
              'peak_vram': {'allocated': max((r.get('peak_allocated_vram', 0) for r in train), default=0), 'reserved': max((r.get('peak_reserved_vram', 0) for r in train), default=0)},
              'pointer_metrics': {'overall': train[-1].get('pointer', {}) if train else {}, 'by_type': (train[-1].get('pointer', {}) if train else {}).get('by_type', {}), 'by_bank_bucket': (train[-1].get('pointer', {}) if train else {}).get('by_bank_bucket', {})}}
    out_json = a.json or a.run / 'summary.json'; out_md = a.markdown or a.run / 'summary.md'
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    out_md.write_text('# Expanded 9K Stage2 Summary\n\n' + '\n'.join([f'- Training records: {result["train_records"]}', f'- Final loss: {result["final_loss"]}', f'- Best loss: {result["best_loss"]}', f'- Examples/s: {result["throughput"]["examples_per_second"]}', f'- Peak allocated VRAM: {result["peak_vram"]["allocated"]}', f'- Failures: {len(failures)}']) + '\n')
    print(json.dumps(result, sort_keys=True)); return 0
if __name__ == '__main__': raise SystemExit(main())
