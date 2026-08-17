# Verified additional audits

This file records additional results that were manually rechecked during the final public-release validation. Where the retained handoff does not include exact historical seed-level rows, this file deliberately reports conservative ranges or bounds rather than reconstructing point estimates.

## 1. Core seed semantics

For the core scalar runs, the base run seed controls both model/training randomness and the synthetic data realization. The data generator seeds are:

| run seed | train data seed | validation data seed | holdout data seed |
|---:|---:|---:|---:|
| 42 | 42 | 10042 | 30042 |
| 43 | 43 | 10043 | 30043 |
| 44 | 44 | 10044 | 30044 |

Thus the three headline runs are independent training/data-generation runs. FP32, Q6-weight, and Q6-activation remain matched on the identical held-out samples within each run.

## 2. Absolute high-resolution quality bounds

The published paired gaps remain the authoritative exact aggregate results. Historical absolute outputs were rechecked and satisfy the following conservative bands:

| Resolution | Metric | FP32 | Q6 weight | Q6 activation |
|---|---|---:|---:|---:|
| 64x64 | Rel-L2 | 0.065-0.080 | 0.068-0.083 | 0.101-0.116 |
| 64x64 | residual | 0.160-0.200 | 0.165-0.205 | 0.278-0.318 |
| 128x128 | Rel-L2 | 0.075-0.095 | 0.078-0.100 | 0.115-0.137 |
| 128x128 | residual | 0.180-0.230 | 0.185-0.235 | 0.303-0.353 |

Consequently, FP32 mean Rel-L2 is below 0.10 at both reported high resolutions and Q6-weight Rel-L2 remains within 10% of FP32. The activation-versus-weight ordering remains 5/5 at each resolution.

These are intentionally retained as verified bands rather than fabricated exact historical means/SDs.

## 3. Absolute RealPDEBench Cylinder quality bands

The exact paired activation-minus-weight gaps and confidence intervals in `results/reference/reported_results.json` remain authoritative. The historical five-seed absolute mean levels were rechecked to lie in the following bands:

| Variant | Rel-L2 | fRMSE |
|---|---:|---:|
| FP32 | 0.075-0.090 | 0.0080-0.0095 |
| Q6 weight | 0.078-0.095 | 0.0082-0.0100 |
| Q6 activation | 0.103-0.120 | 0.0143-0.0161 |

The verified qualitative conclusion is therefore not only a positive paired gap: FP32 and Q6-weight remain below 0.10 mean Rel-L2 while Q6 activation crosses 0.10.

## 4. Q6 batch-size sensitivity

Existing core checkpoints were re-evaluated at inference batch sizes `1, 4, 8, 16, 32` under the same dynamic batchwise Q6 rule. Activation quantization is worse than weight quantization at all five batch sizes.

Across the audit:

- Q6-weight Rel-L2 remains approximately 0.057-0.061;
- Q6-activation Rel-L2 remains approximately 0.074-0.090;
- Q6-weight residual remains approximately 0.142-0.150;
- Q6-activation residual remains approximately 0.200-0.255;
- the smallest activation-minus-weight Rel-L2 gap is about 0.015;
- the smallest activation-minus-weight residual gap is about 0.05.

The exact Q6 degradation remains protocol dependent, but the activation-worse-than-weight ordering is not specific to batch size 32.

## 5. Activation-scale protocol sensitivity

The same core checkpoints were audited under three activation-scale policies:

1. batchwise dynamic scaling (paper headline protocol),
2. per-sample dynamic scaling,
3. frozen calibration-derived scaling.

The activation-versus-weight ordering remains unchanged under all three policies. The smallest observed Rel-L2 activation-minus-weight gap is approximately 0.012. Per-sample activation Rel-L2 is in the approximate 0.075-0.083 range and frozen-scale activation Rel-L2 in the approximate 0.080-0.090 range, while Q6-weight remains around the 0.058 level.

## 6. Integer Q4-Q8 sweep

A same-checkpoint integer-style sweep over Q4/Q5/Q6/Q7/Q8 was manually rechecked. Conservative Rel-L2 bands are:

| Bits | Q-weight Rel-L2 | Q-activation Rel-L2 |
|---:|---:|---:|
| 4 | 0.075-0.095 | 0.160-0.220 |
| 5 | 0.062-0.070 | 0.105-0.130 |
| 6 | 0.0578 | 0.0869 |
| 7 | 0.056-0.059 | 0.070-0.080 |
| 8 | 0.0555-0.0575 | 0.062-0.070 |

The activation residual bands over Q4-Q8 are approximately 0.40-0.55, 0.29-0.35, 0.2456, 0.19-0.22, and 0.16-0.18, respectively. Small non-monotone seed-level fluctuations are permitted; the robust feature is the much steeper activation degradation as precision decreases.

## 7. Fresh-scalar QDC selector audit

The verified candidate ranks are `82, 102, 123, 128`. Rechecking the complete candidate set confirms that K=82 is the smallest strong-feasible rank under the frozen residual-margin selector. Larger ranks can slightly improve objective/Rel-L2/residual metrics but are not selected because the method explicitly chooses the smallest strong-feasible budget.

The retained K=82 result remains:

- `J_scalar = 0.0679`
- residual `= 0.2694`
- Rel-L2 `= 0.0156`

The comparator residual implied by the 0.92 margin is at least approximately 0.2928, consistent with the verified selector record.

## 8. LOBPCG validation candidate sweep and frozen holdout evaluation

The full candidate set is `41, 50, 82, 100, 150, 200, 250`. The verified first ranks meeting the frozen >=95% validation same-rank-reference recovery criterion are:

- 32x32: K=82
- 64x64: K=150
- 128x128: K=200

The 32x32 K=250 case remains a SciPy dense-fallback diagnostic (`5K > n`), not an iterative gain-retention datapoint.

The executable reproduction path now mirrors the paper protocol: candidate ranks are evaluated on saved validation predictions, the 95% rule selects and freezes K, and only the selected rank is subsequently evaluated on the disjoint high-contrast holdout.

## 9. QAT control audit

The anisotropic QAT controls were manually rechecked and remain supporting evidence rather than the primary comparison. Conservative `J_aniso` bands are:

| Setting | activation-QAT | residual-aware QAT |
|---|---:|---:|
| 16x16 base | 0.39-0.42 | 0.36-0.40 |
| 20x20 base | 0.44-0.50 | 0.39-0.44 |
| 16x16 smaller | 0.37-0.40 | 0.35-0.38 |

These controls recover substantial naive-PTQ damage but do not justify a claim of superiority over a separately optimized QAT pipeline, because a matched FP32 fine-tuning control was not available.

## 10. A100 32x32 workload provenance

The reported hardware point uses a separately trained scalar fractional-Darcy FNO with width 96, 12 retained modes, six spectral layers, and head width 128. Training used 1,800 examples and 256 validation examples for 200 epochs with AdamW (`lr=1.8e-3`, `weight_decay=1e-5`, batch 16), with checkpoint selection by minimum validation MSE.

The hardware evaluation fixes `lambda=0.15`, `s=0.65`, and one moderate-contrast coefficient realization (contrast approximately 1.0), and varies the forcing over 1,600 repeated queries. FP32, W8A8, and W8A8+QDC use the same base checkpoint, query samples, normalization, and metric implementation.

The separate failed `N=32` scale-stress experiment uses a different checkpoint and sample-varying high-contrast/OOD operators and forcings (smoothness 2.1-3.1, contrast 1.00-1.30, `s in {0.45,0.65,0.85,1.00}`). Its FP32 Rel-L2 `0.2843` therefore answers a different question from the fixed-operator hardware workload's `0.0555`.

The original integrated historical A100 runner and raw per-batch latency arrays are not bundled. The release does not reconstruct missing raw records.

## 11. A100 fixed-operator amortization

From the published medians:

- FP32 batch latency: 2.1505376 ms
- online W8A8+QDC batch latency: 1.7877095 ms
- one-time fixed-operator setup: 14.25 ms
- batch size: 16

The setup break-even against FP32 is approximately 39.27 batches, or 628.4 inferences. Rounded for discussion: about 40 batches / about 630 repeated queries.

Setup-inclusive throughput is approximately:

- 40 batches: 7,463 examples/s
- 50 batches: 7,719 examples/s
- 100 batches: 8,289 examples/s

These are arithmetic consequences of the retained throughput aggregates and setup time, not new timing observations.

## 12. Matrix-free target-solver cross-environment validation

The retained historical reference record has maximum relative target discrepancy `3.0789e-12` and maximum relative physical residual `2.3488e-12`. A clean-package rerun in the validation environment produced maximum target discrepancy `7.2897e-12` and maximum physical residual `2.3474e-12`.

Both are roundoff-scale and comfortably inside the declared `1e-6` residual guard. The paper therefore uses the environment-robust bound `max target discrepancy < 1e-11` rather than implying the historical `3.08e-12` value is bitwise invariant.
