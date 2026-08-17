# Third-party notices

This archive contains original QDC reproduction code under the MIT License, except where a file explicitly carries a different notice. External projects and adapted interoperability code retain their upstream terms.

## RealPDEBench

- Project: **RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data**
- Upstream: `AI4Science-WestlakeU/RealPDEBench`
- Reproduction source commit: `62f4c80ab17f78933d046f2b038531dbc6a478a0`
- Dataset snapshot: Hugging Face `AI4Science-WestlakeU/RealPDEBench` at revision `5c2bc33eaf79460824a54abfe547198faebd190f`
- Model snapshot: Hugging Face `AI4Science-WestlakeU/RealPDEBench-models` at revision `3c377d3a3c1cb344d37062b5c60221c06dd82395`
- Source-code license: **CC BY-NC 4.0**
- Dataset license: **CC BY-NC 4.0**
- Public model-checkpoint repository license: **CC BY 4.0**

`qdc_fno/realpdebench_patch.py` is an interoperability adaptation that follows the upstream RealPDEBench FNO spectral-block structure so that the activation-only and weight-only Q6 interventions can be evaluated without vendoring the benchmark. That file is distributed under the RealPDEBench CC BY-NC 4.0 terms, with modifications for the QDQ intervention. RealPDEBench source code, data, and checkpoints are not redistributed in this ZIP. The adapted interoperability file retains the source repository's CC BY-NC 4.0 terms; use of downloaded dataset/model assets remains subject to their respective upstream licenses above.

## NVIDIA CUTLASS

The file `hardware/cutlass_complex_int8_gemm.cu` is integration/reference code intended to be compiled against an external NVIDIA CUTLASS checkout. CUTLASS itself is not bundled. Users must follow the license in the CUTLASS revision they install. The **historical reported A100 measurement used CUTLASS 4.5.3** with CUDA 12.8 on A100/SM80. A separately validated later/current reproduction environment may use CUTLASS 4.7.0, but that later version must not be attributed to the historical Table 7 measurement. Any concrete rerun should record the exact checked-out CUTLASS revision/commit in the captured environment.
