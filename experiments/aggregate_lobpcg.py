"""Aggregate LOBPCG-QDC diagnostics over matched saved artifacts."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('files', nargs='+')
p.add_argument('--out', required=True)
a = p.parse_args()
rows = [json.loads(Path(f).read_text()) for f in a.files]
out = {
    'resolution': rows[0]['resolution'],
    'split': rows[0].get('split', 'unspecified'),
    'k': rows[0]['k'],
    'runs': len(rows),
    'dense_fallback_any': any(bool(r.get('dense_fallback', False)) for r in rows),
}
for key in [
    'mean_reference_rel_l2_recovery',
    'mean_operational_rel_l2_recovery',
    'mean_gain_retention',
    'mean_fp32_gap_closure',
    'mean_residual_fp32_gap_closure',
]:
    vals = [float(r[key]) for r in rows]
    out[key] = {
        'values': vals,
        'mean': statistics.fmean(vals),
        'sample_sd': statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
