# Supporting controls and artifact coverage

This file makes the artifact boundary explicit for supporting controls that are named in the paper but are not central end-to-end reproduction targets.

## 64-mask random equal-budget null audit

The paper reports the historical qualitative conclusion that energy/lowband-family masks outperform the random-mask median on physical metrics, while the highband policy is favored only by the high-frequency proxy. The compact release does **not** bundle the complete historical per-mask quantitative table or a one-command rerun of all 64 masks. Those unavailable rows are not reconstructed.

## Fourier-diffusion and fractional-Burgers negative controls

The paper retains these as historical negative controls that delimit the activation-versus-weight claim rather than establishing positive evidence for a universal quantization law. The compact release does **not** bundle complete quantitative raw records or turnkey rerun scripts for these two historical negative-control studies. No replacement numbers are fabricated.

## Tie/mixed and QAT controls

The release retains configurations/components and verified ranges/reference information for the declared tie/mixed and QAT controls. These are supporting controls rather than the primary reproduction pillars; see `REPRODUCIBILITY_MATRIX.md` and `VERIFIED_ADDITIONAL_AUDITS.md`.

The main reproducibility targets remain the core Q6 diagnosis, fresh-scalar QDC held-out evaluation, high-resolution diagnosis, validation-selected LOBPCG workflow, and pinned RealPDEBench workflow. The specialized A100 result is preserved as historical measurement evidence plus reference components, as documented separately in `hardware/README.md`.
