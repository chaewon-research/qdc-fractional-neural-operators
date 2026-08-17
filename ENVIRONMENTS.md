# Environments and historical execution record

This file separates the **historical environments that produced reported measurements** from **later/current reproduction targets**. Where an exact patch-level library version was not preserved historically, the coarser verified description is retained rather than inferred retrospectively.

## Historical main training host

Reported A100 training/evaluation runs used one full physical GPU per ordinary job on an NVIDIA DGX A100 320GB host:

- GPU per run: NVIDIA A100-SXM4 40 GB (SM80), one GPU per job
- GPUs physically installed in host: 8 x A100-SXM4 40 GB
- distributed training: no
- MIG: disabled
- host CPUs: 2 x AMD EPYC 7742, 64 physical cores each
- host memory: 1 TB RAM
- application clocks: not manually locked; default DVFS
- power limit: default; not manually modified
- TensorRT: not used
- `torch.compile`: not used
- CUDA stream behavior: default stream / no unusual custom stream scheduling

The presence of eight GPUs in the DGX host should not be interpreted as eight-GPU training.

## Historical model-training precision

Synthetic model training and RealPDEBench Cylinder fine-tuning used:

- FP32 parameters and optimization
- no AMP/autocast
- no `GradScaler`
- no BF16 training
- no FP16 model training

The FP16 FFT / INT8 spectral path applies only to the specialized A100 inference/hardware experiment. TF32 being disabled is a condition of the FP32 reference in that hardware timing audit, not a generalized statement about every historical training run.

## Historical A100 software environment

The environment below was manually cross-checked against the original experiment/environment records and should be treated as the historical environment for the reported A100 work:

| Field | Historical verified value |
|---|---|
| GPU | NVIDIA A100-SXM4 40 GB |
| architecture / compile target | SM80 / `sm_80` |
| operating system | Linux x86_64 |
| Python | 3.11.x |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime/toolkit | 12.8 |
| NVIDIA driver | 570.133.20 |
| CUTLASS used for reported Table 7 hardware measurement | 4.5.3 |
| cuFFT | CUDA 12.8 release family |
| cuSOLVER | CUDA 12.8 release family |

Exact cuFFT/cuSOLVER patch versions were not historically retained and are intentionally not inferred.

## Historical A100 inference/hardware measurement

The reported fixed-operator A100 operating point used:

- one full A100-SXM4 40-GB GPU, MIG disabled
- FP16 cuFFT
- four real INT8 CUTLASS GEMMs per complex spectral product
- INT32 GEMM accumulation
- FP32 reduction
- TF32 disabled for the FP32 reference
- batch size 16
- 50 warm-up batches
- 100 timed batches
- synchronized CUDA events
- median timed-batch latency
- host-to-device transfers excluded
- online quantization/dequantization and QDC application included
- default DVFS; application clocks not manually locked
- `sm_80` compilation target

The historical CUTLASS version for this measurement is **4.5.3**.


The verified `32x32` hardware workload uses a separately trained scalar fractional-Darcy FNO (width 96, 12 retained modes, six spectral layers, head width 128; 1,800 training and 256 validation examples; 200 epochs; AdamW learning rate `1.8e-3`, weight decay `1e-5`, batch 16; checkpoint selected by minimum validation MSE). The stream fixes `lambda=0.15`, `s=0.65`, and one moderate-contrast coefficient realization (contrast approximately 1.0) while varying the forcing over 1,600 repeated queries. FP32/W8A8/W8A8+QDC use the same checkpoint and query samples. This is a different workload and checkpoint from the sample-varying high-contrast/OOD `N=32` scale-stress record.

## Synthetic GPU reproduction environment

For a clean rerun of the synthetic GPU workflows, use Python 3.11 and the published PyTorch 2.7.0 + CUDA 12.8 wheel combination:

```bash
python3.11 -m venv .venv-qdc
source .venv-qdc/bin/activate
python -m pip install --upgrade pip
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

Record the concrete runtime with `python scripts/capture_environment.py` before archival reproduction.

## RealPDEBench Cylinder reproduction environment

Keep the external benchmark separate from the synthetic-QDC environment:

```bash
python3.11 -m venv .venv-realpde
source .venv-realpde/bin/activate
python -m pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_realpdebench_fno.txt
bash scripts/prepare_realpdebench.sh
```

`prepare_realpdebench.sh` checks out the immutable source commit and installs it with `--no-deps`. This prevents the upstream project's unpinned dependency list from modifying the environment after the explicit FNO/Hugging Face dependencies are installed. After a full run, capture `pip freeze` and `results/environment.json` with the result bundle.

## Later validated hardware reproduction environment

A separately validated later/current hardware reproduction target may use Python 3.11, PyTorch 2.7.0 + CUDA 12.8, NVIDIA A100-SXM4 (SM80), and CUTLASS 4.7.0. **CUTLASS 4.7.0 did not generate the historical reported Table 7 measurement.** Any fresh hardware rerun should record the exact driver, CUDA runtime/toolkit, cuFFT/cuSOLVER versions, GPU state, and CUTLASS commit actually used.

## Local package validation snapshot

The packaged unit/sanity tests and small-grid matrix-free validation were last audited in the build environment recorded in `environment_validation.json`. This snapshot documents the package-validation runtime and is not a claim about the historical A100 measurement environment.

## Determinism and expected agreement

The experiment scripts seed Python, NumPy, PyTorch, and CUDA. GPU kernels and library implementations can still vary across driver/CUDA/PyTorch stacks, so the reproduction target is statistical agreement with the reported seed-level ordering and aggregate metrics rather than bit-for-bit identity. Record the concrete runtime with `scripts/capture_environment.py` for any archival rerun.

`environment_base.yml` contains only the shared non-PyTorch dependencies; install the workflow-specific PyTorch build first using the commands above.
