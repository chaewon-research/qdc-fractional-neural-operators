# Quantized Defect Correction for Fractional Neural Operators - Reproducibility Artifact
[![DOI](https://zenodo.org/badge/1336958141.svg)](https://doi.org/10.5281/zenodo.21977847)

[![DOI](https://zenodo.org/badge/1336958141.svg)](https://doi.org/10.5281/zenodo.21977847)

This archive accompanies **Conditional Precision Bottlenecks and Quantized Defect Correction in Fractional Neural Operators**. It is a cleaned reproducibility implementation organized from the verified experimental protocol rather than a verbatim dump of exploratory notebooks.

Historical training/fine-tuning used FP32 parameters and optimization without AMP/autocast, GradScaler, BF16, or FP16 model training. The FP16-FFT/INT8 path is confined to the specialized A100 inference audit. Historical environment and compute provenance are documented in `ENVIRONMENTS.md` and `COMPUTE_ACCOUNTING.md`; retrospectively reconstructed compute is reported as ranges rather than invented exact totals.

The artifact prioritizes four claims that carry the paper's main scientific weight:

1. the corrected 16x16 high-contrast/OOD Q6 activation-versus-weight diagnosis;
2. fresh-scalar QDC at the independently frozen residual-margin rank;
3. the higher-resolution matrix-free target and selected-rank LOBPCG diagnostics; and
4. the five-run RealPDEBench Cylinder activation-versus-weight validation.

Supporting mixed Q4/Q8, QAT, E4M3FN, anisotropic, subspace, and hardware components are provided where they materially help audit the method. The archive does **not** claim that every secondary ablation is a one-command rerun.


## Paper

The current public preprint is available at [`paper/QDC_Fractional_Neural_Operators_Preprint_2026.pdf`](paper/QDC_Fractional_Neural_Operators_Preprint_2026.pdf).

**Research contact:** Chaewon Yoon - `chaewon.yoon.ds@gmail.com`

## 1. Installation

Use the environment appropriate to the workflow. Exact validation metadata and the recommended synthetic-GPU/RealPDEBench environments are documented in `ENVIRONMENTS.md`; `environment_base.yml` contains only shared non-PyTorch dependencies. For the base synthetic workflow, install a PyTorch build matched to the intended CPU/CUDA environment first, then install the remaining dependencies and this package:

```bash
python -m venv .venv
source .venv/bin/activate
# Install PyTorch separately using the official command for the target CUDA/CPU environment.
pip install -r requirements.txt
pip install -e . --no-deps
```

The same source tree is usable on workstations/HPC systems and in Google Colab (`colab/quickstart.ipynb`).

## 2. Quick package validation

```bash
bash scripts/validate_package.sh
```

This idempotent quick check verifies the closed-world manifest and hashes, runs unit/sanity tests without writing pytest/bytecode caches, performs a one-sample $16\times16$ exact-dense numerical sanity check, regenerates the matched core-diagnosis figure, and runs the execution-only scalar smoke test. Generated validation outputs are written under git-ignored `results/generated/`. It is a package check, not a substitute for the full training runs below.

For the complete $16\times16$/$32\times32$ exact-dense target-solver audit used for the stronger numerical validation record, run:

```bash
bash scripts/validate_numerics_full.sh
```

The full numerical audit can take several minutes on CPU-only machines.

## 3. Central reproduction workflows

### Core 16x16 Q6 diagnosis

```bash
bash scripts/reproduce_core.sh
```

This trains seeds 42-44, selects each FP32 checkpoint by minimum validation MSE, evaluates FP32/Q6-weight/Q6-activation on the disjoint high-contrast/OOD holdout, and writes both seed-level JSON files and `results/core_scalar/aggregate.json`.

### Fresh-scalar QDC

```bash
bash scripts/reproduce_qdc.sh
```

The paper's independently frozen residual-margin selector chooses `K=82` for each fresh scalar run. The fresh-scalar candidate ranks are `[82, 102, 123, 128]`, corresponding to cost fractions `[0.32, 0.40, 0.48, 0.50]`; this grid is distinct from the LOBPCG candidate grid. The frozen selector record (`results/reference/qdc_selection_record.json`) documents the budget grid, margin, and selected rank without fabricating unavailable per-candidate calibration measurements. This focused workflow reruns the held-out QDC application for seeds 45-47 at that frozen rank and aggregates the resulting metrics. `qdc_fno/qdc.py` contains the residual-margin selector, including the strong-feasible and weak-feasible one-rank-promotion rules; it reports no feasible budget when neither guard set is satisfied. The supporting mixed/tie search that established the frozen comparator is documented by the paper and reference results rather than rerun as part of this focused artifact.

Calibration freezes the rank budget; residual-energy top-K physical modes are selected per sample from the low-bit residual defect.

### Higher-resolution diagnosis and selected-rank LOBPCG

First validate the target solver:

```bash
python experiments/validate_highres_targets.py
```

To rerun all five 64x64/128x128 activation-versus-weight experiments and aggregate their paired gaps:

```bash
bash scripts/reproduce_highres.sh
```

The runs cache matrix-free targets under filenames containing a SHA-256 configuration fingerprint and also embed the full experiment/solver specification inside each cache. A mismatched cache is rejected rather than silently reused.

To run the focused selected-rank matrix-free eigenspace diagnostics from a fresh checkout (training/generating the required prediction artifacts first):

```bash
RUN_TRAIN=1 bash scripts/reproduce_lobpcg.sh
```

If the high-resolution prediction artifacts have already been generated, the cheaper reuse path is `RUN_TRAIN=0 bash scripts/reproduce_lobpcg.sh`.

To reproduce the paper's **validation-selection -> frozen-rank -> disjoint-holdout** protocol from a fresh checkout, use:

```bash
RUN_TRAIN=1 bash scripts/reproduce_lobpcg_sweep.sh
```

The compact archive does not bundle the multi-gigabyte `*.validation_qdc.pt` / `*.qdc.pt` prediction artifacts. If they have already been generated by the updated high-resolution trainer, reuse them with `RUN_TRAIN=0 bash scripts/reproduce_lobpcg_sweep.sh`.

The updated high-resolution trainer saves both `*.validation_qdc.pt` and disjoint holdout `*.qdc.pt` artifacts. `reproduce_lobpcg_sweep.sh` evaluates all declared candidate ranks on the validation artifacts, applies the frozen 95% high-accuracy same-rank-reference recovery rule, verifies the historical selected ranks (`K=82/150/200` at 32/64/128), and then evaluates **only the selected rank** on the holdout artifacts. Set `RUN_TRAIN=1` to train/generate the required prediction artifacts first. The 32x32 `K=250` entry is explicitly recorded as the SciPy dense-fallback diagnostic (`5K > n`) rather than an iterative LOBPCG result.


### Historical A100 32x32 workload identity

The retained A100 accuracy/throughput point is not the failed `N=32` scale-stress experiment. It uses a separately trained scalar fractional-Darcy FNO with width 96, 12 retained modes, six spectral layers, and head width 128; training used 1,800 examples plus 256 validation examples for 200 epochs with AdamW (learning rate `1.8e-3`, weight decay `1e-5`, batch 16) and minimum-validation-MSE checkpoint selection. The hardware workload fixes `lambda=0.15`, `s=0.65`, and one moderate-contrast coefficient realization (contrast about 1.0), then varies the forcing over 1,600 repeated queries. FP32, W8A8, and W8A8+QDC use the same base checkpoint and the same query stream.

The separate `N=32` scale-stress record uses a different checkpoint and sample-varying high-contrast/OOD operators and forcings. This is why its FP32 Rel-L2 `0.2843` is not comparable to the A100 workload's `0.0555`. The historical integrated hardware runner and raw per-batch latency arrays are not bundled; this archive preserves the verified configuration, aggregate measurement record, timing protocol, and low-level reference components without reconstructing missing raw records.

### Release hygiene and local runtime paths

The release archive is manifest-closed: `scripts/validate_package.sh` checks that every shipped file is either listed in `MANIFEST.txt` or is one of the two manifest bookkeeping files. Python/test caches and generated artifacts are excluded. Local RealPDEBench runtime YAML files under `work/` may contain machine-specific absolute dataset/checkpoint paths and are intentionally git-ignored; do not commit or redistribute those generated configs. Persistent checkpoint-selection metadata uses portable identifiers and SHA-256 hashes rather than absolute local paths.


### Supporting controls not fully replayed

The 64-mask random equal-budget null audit and the Fourier-diffusion/fractional-Burgers negative controls are historical supporting experiments named in the manuscript. Complete historical per-mask/raw quantitative records and turnkey reruns for those controls are not bundled in this compact archive and are not reconstructed. Their artifact status is documented explicitly in `SUPPORTING_CONTROLS.md` and `REPRODUCIBILITY_MATRIX.md`.

### RealPDEBench Cylinder

RealPDEBench is an external dependency whose source/data are CC BY-NC 4.0 and whose public model-checkpoint repository is CC BY 4.0. The source, dataset, model snapshots, and numerical checkpoint used to initialize fine-tuning are fixed in `external/realpdebench_assets.lock.json`.

Prepare the pinned dependency and numerical initialization checkpoint:

```bash
bash scripts/prepare_realpdebench.sh
```

Then fine-tune five independent FNOs (seeds 0-4) on the real Cylinder split. The script passes the locked dataset revision to the upstream Hugging Face loader and automatically copies the upstream validation-RMSE-best checkpoint to a stable seed path:

```bash
export DATASET_ROOT=/path/to/writable/realpdebench-data
bash scripts/run_realpdebench_finetune.sh
```

Finally run paired Q6 activation/weight evaluation and aggregate the five paired Student-t intervals:

```bash
export DATASET_ROOT=/path/to/writable/realpdebench-data
bash scripts/reproduce_realpdebench.sh
```

The evaluator uses the pinned upstream loader, Gaussian normalization, ten-step autoregressive rollout, `test_mode=all`, and upstream `eval_metrics`; only the spectral activation or spectral-weight Q6 intervention is adapted.


### Synthetic FNO used by the cleaned reproduction

The synthetic workflows use one explicitly documented architecture across matched comparisons: spatial coordinate channels are appended to the PDE inputs; each block combines two-sided `rfft2/irfft2` spectral convolution with a pointwise branch and a circular local 3x3 branch, followed by GroupNorm, GELU, and a 0.15 residual connection. Width/mode/layer budgets vary only where declared by configuration. The forcing generator combines periodic Gaussian sources with low-frequency sinusoidal structure; integer count ranges in YAML are inclusive. Spatial coordinate channels use the non-duplicated periodic grid $j/N$, $j=0,...,N-1$. The high-contrast/OOD holdout also uses a harder forcing distribution.

Q6 PTQ uses ordinary QDQ. QAT uses a separate straight-through fake-quantization path (`x + (Q(x)-x).detach()`), so gradients remain defined through quantized spectral activations.

## 4. Quantization definitions

The core Q6 experiments use a symmetric signed six-bit QDQ convention with 63 representable integer levels `[-31, 31]`, zero point 0, and round-to-even. Fourier real and imaginary components use independent scales in the core Q6 diagnostic. Activation scales are recomputed dynamically for each retained signed-frequency block by reducing over the inference batch, channel, and retained-mode axes, so the absolute Q6 degradation is protocol- and batch-dependent; the declared batch size is therefore part of every matched comparison. Spectral-weight scales reduce over the corresponding input/output-channel and retained-mode axes. QDQ outputs are multiplied/accumulated in floating point.

The separate mixed Q4/Q8 condition assigns Q8 to the calibrated top 25% of retained modes and Q4 to the remainder (nominal 5.0-bit value payload, excluding masks/metadata). The calibration artifact freezes both the mode mask and numerical activation scales from the fixed 500-example training calibration set; no test-time absmax recomputation is used.

## 5. Metrics, anisotropic convention, and fixed objectives

For anisotropic fields, tensor axis 0 is x and axis 1 is y. The anisotropic event mask uses the gradient magnitude of `0.5 * (log(kx) + log(ky))`.

- anisotropic audit score: `event + 0.50*Rel-L2 + 0.25*residual`;
- fresh-scalar QDC score: `Rel-L2 + event + 0.10*residual + 0.25*HF-log`;
- static-mask development score: `event + 0.50*HF-log + 0.50*Rel-L2 + 0.20*residual_proxy`.

Composite weights are frozen before the held-out comparisons for which they are used.

## 6. Supporting implementations

- `qdc_fno/quantization.py`: Q6, frozen mixed Q4/Q8, E4M3FN.
- `qdc_fno/training.py`: FP32 training and differentiable residual-aware scalar QAT utilities. The reported anisotropic QAT control uses the lowest validation $J_{\mathrm{aniso}}$ checkpoint; QAT is supporting evidence rather than one of the four end-to-end reproduction targets.
- `qdc_fno/qdc.py`: QDC rankings, correction, and residual-margin selector.
- `experiments/train_anisotropic.py`: anisotropic family evaluation; `configs/anisotropic_n16_smaller.yaml` records the width-48/modes-6/layers-4 smaller-capacity control.
- `hardware/`: specialized A100/CUTLASS reference source and timing protocol for the fixed-operator repeated-query operating point. The historical measurement used A100-SXM4 40 GB / SM80, CUDA 12.8, PyTorch 2.7.0+cu128, NVIDIA driver 570.133.20, and CUTLASS 4.5.3. CUTLASS 4.7.0 is documented only as a later validated reproduction target. The W8A8 hardware measurement is separate from the uniform-Q6 activation-versus-weight diagnosis and does not constitute an on-hardware replication of that object-level asymmetry. It is intentionally not represented as a commodity-hardware one-command reproduction of the reported throughput.
- `COMPUTE_ACCOUNTING.md`: per-family A100 ranges, development compute, the separate T4 audit, CPU matrix-free target-generation/LOBPCG accounting, and high-resolution storage footprint.

## 7. Figures and reference results

`results/reference/reported_results.json` and `qdc_tables.json` contain the authoritative aggregate values reported in the paper. They are reference targets, not reconstructed historical per-seed files, and should not be interpreted as newly generated raw measurements. The archive does not synthesize missing seed-level results.

```bash
python scripts/make_core_diagnosis_figure.py
python scripts/make_reference_tables.py
```

The source artifact for the mode-alignment figure is under `results/reference/figures/`; the original raw per-mode ranking table was not retained in this cleaned artifact. After running central workflows, `scripts/report_against_reference.py` can report fresh aggregate deviations from the authoritative paper values without overwriting either set of results.

## 8. Licensing and third-party assets

Original code in this archive is MIT licensed except where a file explicitly carries upstream terms. RealPDEBench data/checkpoints are not redistributed. See `THIRD_PARTY_NOTICES.md`, `external/REALPDEBENCH.md`, and `external/realpdebench_assets.lock.json` for precise provenance and license information.

## 9. Reported-result consistency checks

`results/reference/derived_consistency_checks.json` contains quantities derived mathematically from the verified aggregate results (for example, the sample SD implied by each five-seed Student-t interval and the batch latencies implied by the reported A100 throughputs). They are consistency checks, not invented raw measurements. Regenerate them with `python scripts/derive_consistency_checks.py`.
