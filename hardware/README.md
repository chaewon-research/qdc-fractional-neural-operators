# A100 FP16-FFT / INT8 spectral path

The hardware experiment in the paper uses one full NVIDIA A100-SXM4 40-GB GPU (SM80) at 32x32, batch 16, with 50 warm-up and 100 measured batches. MIG is disabled. FFTs remain FP16; each complex spectral product is represented by four real INT8 GEMMs, accumulated in INT32 and reduced in FP32. Timing uses synchronized CUDA events and reports the median. Host-device transfers are excluded; online quantization/dequantization and QDC application are included. Application clocks were not manually locked; the run used default DVFS.

The historical measurement environment was Linux x86_64, Python 3.11.x, PyTorch 2.7.0+cu128, CUDA 12.8, NVIDIA driver 570.133.20, and CUTLASS 4.5.3, with compilation target `sm_80`. cuFFT and cuSOLVER are retained at the verified CUDA 12.8 release-family level rather than assigned unpreserved patch versions. TF32 was disabled for the FP32 reference condition.

`cutlass_complex_int8_gemm.cu` is the bundled four-GEMM CUTLASS reference primitive. Integrate it with the FNO spectral blocks after FP16 cuFFT and before the FP32 reduction/QDC path. `benchmark_a100.py` provides the synchronized-CUDA-event timing harness and reference FFT path used to validate the measurement procedure; `cutlass_complex_int8_gemm.cu` provides the low-level four-GEMM primitive. The archive does not claim that these reference components alone reconstruct the complete reported A100 pipeline. Exact fresh-run performance depends on the CUDA/CUTLASS build and GPU state; capture the run environment with `scripts/capture_environment.py`.

The reported hardware point is a fixed-operator repeated-query amortization. Rank 41 is the illustrative lower-rank hardware operating point used to expose the accuracy-throughput tradeoff; it is distinct from the primary calibration-selected fresh-scalar budget. One rank-41 physical operator/eigenbasis is prepared once and reused across 100 batches of 16 inferences. The one-time QDC setup uses cuSOLVER and includes the eigensolve, basis-memory formatting, and residual-operator preparation; its reported time is 14.25 ms. The corresponding reported throughputs imply median batch latencies of about 2.1505 ms (FP32), 1.4286 ms (W8A8), and 1.7877 ms (online W8A8+QDC). If the physical coefficient/operator changes between examples, setup would need a different amortization; this package does not report a variable-operator throughput figure. The W8A8 experiment is also separate from the paper's uniform-Q6 activation-versus-weight diagnosis: it demonstrates a hardware accuracy-throughput operating point, not an on-hardware proof of the activation/weight asymmetry.


## Verified 32x32 model and workload provenance

The reported Table 7 accuracy/throughput point uses a separately trained scalar fractional-Darcy FNO:

- width 96;
- 12 retained Fourier modes;
- 6 spectral layers;
- head width 128;
- 1,800 training and 256 validation examples;
- 200 epochs;
- AdamW, learning rate `1.8e-3`, weight decay `1e-5`;
- training batch size 16;
- checkpoint selected by minimum validation MSE.

For the hardware workload, `lambda=0.15`, `s=0.65`, and one moderate-contrast coefficient realization (contrast approximately 1.0) are fixed. The forcing/right-hand side varies over 1,600 repeated queries. FP32, W8A8, and W8A8+QDC are matched conditions on the same base checkpoint and query stream.

This workload is deliberately different from the paper's separate one-split `N=32` scale-stress test (`FP32 Rel-L2 = 0.2843`), which uses a different checkpoint and sample-varying high-contrast/OOD coefficient/operator and forcing realizations. The hardware workload's `FP32 Rel-L2 = 0.0555` should therefore not be interpreted as a successful rerun of that scale-stress experiment.

The original integrated historical A100 runner and raw per-batch latency arrays are not bundled. The archive instead preserves the verified model/workload configuration, retained aggregate measurements, timing protocol, environment record, timing harness, and four-real-GEMM CUTLASS primitive. A fresh end-to-end reconstruction requires integrating those components into the specialized FNO/CUDA/CUTLASS path; the archive does not claim that access to an A100 alone reproduces Table 7.

## Historical versus later reproduction environment

- **Historical reported measurement:** CUTLASS 4.5.3, CUDA 12.8, PyTorch 2.7.0+cu128, NVIDIA driver 570.133.20, A100-SXM4 40 GB / SM80.
- **Later/current validated reproduction target:** CUTLASS 4.7.0 may be used with Python 3.11, PyTorch 2.7.0 + CUDA 12.8 on A100/SM80, provided the concrete run environment and exact CUTLASS revision are captured.

CUTLASS 4.7.0 should not be attributed retrospectively to the historical Table 7 measurement.

See `../COMPUTE_ACCOUNTING.md` for the accelerator and CPU resource accounting.

## Fixed-operator break-even

The retained throughput aggregates imply batch latencies whose difference shows that online rank-41 QDC saves approximately 0.3628 ms/batch relative to FP32. Therefore the 14.25 ms one-time setup is amortized after approximately 39.27 batches, or approximately 628.4 inferences at batch 16. Rounded operationally, the setup breaks even after about 40 batches / 630 repeated queries. Setup-inclusive throughput is approximately 7,463 examples/s at 40 batches and 7,719 examples/s at 50 batches. These are arithmetic consequences of the retained historical throughput aggregates and setup time, not additional timing samples. See `../results/reference/a100_historical_measurement.json`.
