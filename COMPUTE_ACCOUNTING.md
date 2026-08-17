# Compute accounting and resource provenance

Compute values in this document are **retrospectively reconstructed from archived configurations and runtime/environment records**. Ranges are reported where complete historical scheduler accounting was not retained. Unavailable precision is not retrospectively fabricated.

CPU scientific compute is reported separately from accelerator GPU-hours. Tesla T4 work is also kept separate without cross-device normalization.

## A. Reported A100 accelerator work

Ordinary A100 training/evaluation used one full NVIDIA A100-SXM4 40-GB GPU per job, without distributed training and with MIG disabled. The DGX host contained eight A100 GPUs, but reported neural-operator jobs used one accelerator per ordinary run.

| Experiment family | Reconstructed A100 compute |
|---|---:|
| Core 16x16 scalar / core diagnostic work | ~5.5-6.5 GPU-h |
| Anisotropic 16x16/20x20, smaller-capacity, independent replication | ~9-10 GPU-h |
| Fresh-scalar QDC training/evaluation | ~5.5-6.5 GPU-h |
| 32x32 scale/E4M3FN and associated checks | ~4.5-5.5 GPU-h |
| 64x64 five-seed experiment | ~13-14 GPU-h |
| 128x128 five-seed experiment | ~40-44 GPU-h |
| RealPDEBench Cylinder five-seed work | ~8.5-9.5 GPU-h |
| QAT, mixed/tie, random-mask, and other secondary controls | ~13-14 GPU-h |
| A100 hardware timing/audit | ~0.4-0.6 GPU-h |
| **Reported A100 experiments overall** | **~100-110 GPU-h** |

The central retrospective estimate is approximately **105 A100 GPU-hours**. The component ranges are intentionally not forced to sum to an exact point value.

### Representative per-seed A100 timing

| Family | Approximate historical A100 GPU-h per seed |
|---|---:|
| Core scalar | ~1.8-2.2 |
| Fresh scalar | ~1.8-2.2 |
| 64x64 | ~2.6-2.8 |
| 128x128 | ~8.0-8.8 |
| RealPDEBench Cylinder | ~1.7-1.9 |

For RealPDEBench, the per-seed range includes fine-tuning, validation/checkpoint selection, and final matched Q6 activation-versus-weight evaluation. A useful approximate decomposition is ~1.5-1.7 GPU-h for fine-tuning and ~0.1-0.3 GPU-h for checkpoint selection/final evaluation, for ~1.7-1.9 GPU-h total per seed.

## B. Preliminary, unsuccessful, and development A100 work

Approximately **35 additional A100 GPU-hours** were used for project work that did not map directly to the final headline experiments.

| Development category | Reconstructed range |
|---|---:|
| architecture/data-generation pilots | ~10-12 GPU-h |
| mask/QDC-selector development | ~7-9 GPU-h |
| failed scale/precision/negative-control attempts | ~5-7 GPU-h |
| debugging, reruns, environment validation | ~5-7 GPU-h |
| hardware kernel/build tuning | ~3-4 GPU-h |
| **Additional development work overall** | **~35 GPU-h** |

These component ranges are likewise not forced to sum exactly to 35.

Combining reported and development work gives approximately **140 A100 GPU-hours of reconstructed A100 accelerator work across the project**.

## C. Separate Tesla T4 runtime audit

The small-grid Tesla T4 runtime audit used:

- NVIDIA Tesla T4
- one GPU
- no distributed execution
- ~0.2-0.5 T4 GPU-hours total

This covers the runtime-audit workflow including warm-up, timing reruns, and associated setup. It uses simulated quantization and precomputed eigensystems to measure correction overhead rather than native low-bit speedup. It is **not** converted into A100 GPU-hours.

## D. CPU scientific compute

### D.1 Matrix-free 64x64 and 128x128 target generation

High-resolution target generation was CPU-side on the same DGX host used for the A100 work:

- host CPUs: 2 x AMD EPYC 7742
- 128 physical CPU cores total
- 1 TB system RAM
- five seed jobs independently scheduled
- ~16 CPU threads per generation job

The timing below is aggregate job time summed across the independently scheduled seed jobs, not campaign wall time when jobs overlapped.

| Resolution | Samples per seed | Samples across five seeds | Aggregate ~16-thread job-hours |
|---|---:|---:|---:|
| 64x64 | 1,800 train + 256 val + 256 holdout = 2,312 | 11,560 | ~4-7 |
| 128x128 | 2,200 train + 256 val + 256 holdout = 2,712 | 13,560 | ~35-60 |
| **Combined** | - | 25,120 | **~40-70** |

The 64x64 workflow uses roughly 0.8-1.4 hours per seed job and the 128x128 workflow roughly 7-12 hours per seed job as implied typical ranges. If one mechanically multiplies by 16 threads/job, this corresponds to approximately 64-112 thread-hours at 64x64 and 560-960 thread-hours at 128x128. We prefer the less ambiguous **aggregate 16-thread job-hours** terminology because physical-core occupancy was not separately metered.

Seed-specific target caches were not automatically shared across all five seeds. Each resolution has three caches per seed (train, validation, holdout), for 15 cache files across five seeds.

### D.2 Matrix-free LOBPCG diagnostics

LOBPCG diagnostics were also CPU-side on the dual-EPYC host, using approximately 16 CPU threads per job and float64 `scipy.sparse.linalg.lobpcg` / `LinearOperator` execution.

Operational settings:

- `largest=False`
- tolerance `1e-4`
- maximum 100 iterations
- no preconditioner
- candidate ranks: 41, 50, 82, 100, 150, 200, 250
- selected ranks: K=82 (32x32), K=150 (64x64), K=200 (128x128)
- tighter larger-grid same-rank references: tolerance `1e-8`, maximum 300 iterations

Approximate selected-rank solve time per seed:

- 32x32, K=82: tens of seconds to ~1 minute
- 64x64, K=150: ~2-10 minutes
- 128x128, K=200: ~10-40 minutes

Approximate full candidate/reference workflow job-time scales:

| Resolution | Approximate ~16-thread job-hours |
|---|---:|
| 32x32 | ~0.5-2 |
| 64x64 | ~2-6 |
| 128x128 | ~8-20 |

Historical peak process RSS was not retained and is not reconstructed.

## E. High-resolution storage footprint

The high-resolution cache payload consists of three float32 input channels plus one float32 target field per sample. QDC artifacts additionally store the holdout input, target, activation-quantized prediction, and FP32 prediction. The deterministic raw tensor-size calculation is approximately:

| Artifact | Approximate raw tensor payload |
|---|---:|
| 64x64 dataset caches, five seeds | ~0.706 GiB |
| 128x128 dataset caches, five seeds | ~3.31 GiB |
| 64x64 QDC prediction artifacts | ~0.12 GiB |
| 128x128 QDC prediction artifacts | ~0.47 GiB |
| **Raw tensor payload overall** | **~4.61 GiB** |

The checked serialized/on-disk footprint for these high-resolution cache + QDC artifacts is approximately **5-6 GiB** after filesystem/serialization overhead. This is not the total repository size and excludes external RealPDEBench assets and unrelated checkpoints.

## F. Historical environment and training precision

See `ENVIRONMENTS.md` for the full historical host/software record. In summary:

- A100-SXM4 40 GB / SM80, one full GPU per ordinary run
- no distributed training; MIG disabled
- Linux x86_64
- Python 3.11.x
- PyTorch 2.7.0+cu128
- CUDA 12.8
- NVIDIA driver 570.133.20
- historical hardware benchmark: CUTLASS 4.5.3
- later validated hardware reproduction target: CUTLASS 4.7.0, clearly separated from the historical measurement
- synthetic training and RealPDEBench fine-tuning: FP32 parameters/optimization, no AMP/autocast, GradScaler, BF16, or FP16 model training

The specialized FP16-cuFFT / INT8-CUTLASS path is an inference/hardware condition only.
