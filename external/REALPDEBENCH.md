# RealPDEBench external asset

The measured Cylinder validation uses the external RealPDEBench project. Source code, data, and checkpoints are not redistributed in this archive.

The public reproducibility snapshot is fixed in `external/realpdebench_assets.lock.json`:

- source repository: `AI4Science-WestlakeU/RealPDEBench`
- source commit: `62f4c80ab17f78933d046f2b038531dbc6a478a0`
- dataset repository: `AI4Science-WestlakeU/RealPDEBench`
- dataset revision: `5c2bc33eaf79460824a54abfe547198faebd190f`
- model repository: `AI4Science-WestlakeU/RealPDEBench-models`
- model revision: `3c377d3a3c1cb344d37062b5c60221c06dd82395`
- numerical checkpoint used to initialize each fine-tuning run: `cylinder/fno/numerical.pth`
- scenario: `cylinder`, PIV-measured velocity fields `(u,v)`
- model: FNO, width 64, four spectral layers, modes `(4,12,16)`
- protocol: Gaussian normalization, 10 autoregressive steps, `test_mode=all`
- five matched runs: independently fine-tune seeds `0,1,2,3,4`, then evaluate Q6 activation and Q6 spectral weights as a paired intervention within each checkpoint
- metrics: upstream `eval_metrics`; Rel-L2 is tuple element 2 and fRMSE/Fourier-space error is tuple element 5 at the pinned source revision
- source license: CC BY-NC 4.0
- dataset license: CC BY-NC 4.0
- model-checkpoint repository license: CC BY 4.0

Run `bash scripts/prepare_realpdebench.sh` to install the pinned source and download the pinned numerical initialization checkpoint. The fine-tuning workflow passes the locked dataset revision to the upstream Hugging Face loader.

The five-run workflow should be executed in its own Python environment. See `ENVIRONMENTS.md`; the pinned RealPDEBench checkout is installed with `--no-deps` after the minimal FNO/HF dependency set is installed, preventing the external project from silently changing the synthetic-QDC environment.
