"""Select the LOBPCG rank on validation and freeze it before holdout evaluation.

Selection rule from the paper/config:
  choose the smallest candidate rank whose *high-accuracy same-rank reference*
  validation Rel-L2 recovery reaches at least `fraction` of the best candidate's
  validation reference recovery.

The operational LOBPCG diagnostics are retained in the candidate files, but the rank
budget is selected from the high-accuracy reference recovery curve so that approximation
quality and budget choice are not conflated.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from qdc_fno.lobpcg import select_smallest_rank_by_reference_recovery

p = argparse.ArgumentParser()
p.add_argument('files', nargs='+', help='validation aggregate JSON files, one per K')
p.add_argument('--fraction', type=float, default=0.95)
p.add_argument('--expected-k', type=int, default=None)
p.add_argument('--out', required=True)
a = p.parse_args()

rows = []
for f in a.files:
    z = json.loads(Path(f).read_text())
    if z.get('split') != 'validation':
        raise SystemExit(f'{f}: expected split=validation, got {z.get("split")!r}')
    rec = float(z['mean_reference_rel_l2_recovery']['mean'])
    rows.append({
        'file': str(f),
        'resolution': int(z['resolution']),
        'k': int(z['k']),
        'mean_reference_rel_l2_recovery': rec,
        'mean_operational_rel_l2_recovery': float(z['mean_operational_rel_l2_recovery']['mean']),
        'mean_gain_retention': float(z['mean_gain_retention']['mean']),
        'dense_fallback_any': bool(z.get('dense_fallback_any', False)),
    })

resolutions = {r['resolution'] for r in rows}
if len(resolutions) != 1:
    raise SystemExit(f'candidate files span multiple resolutions: {sorted(resolutions)}')
rows.sort(key=lambda r: r['k'])
try:
    decision = select_smallest_rank_by_reference_recovery(rows, a.fraction)
except ValueError as e:
    raise SystemExit(str(e))
selected_k = int(decision['selected_k'])

out = {
    'resolution': rows[0]['resolution'],
    'selection_split': 'validation',
    'criterion': 'smallest_rank_with_at_least_fraction_of_best_validation_same_rank_reference_rel_l2_recovery',
    'fraction': a.fraction,
    'best_validation_reference_rel_l2_recovery': decision['best_recovery'],
    'threshold': decision['threshold'],
    'selected_k': selected_k,
    'expected_k': a.expected_k,
    'matches_expected_k': (a.expected_k is None or selected_k == a.expected_k),
    'candidate_results': rows,
    'holdout_policy': 'evaluate_only_selected_k_on_disjoint_holdout_after_selection',
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
if a.expected_k is not None and selected_k != a.expected_k:
    print(f'ERROR: selected K={selected_k}, expected historical K={a.expected_k}', file=sys.stderr)
    raise SystemExit(3)
