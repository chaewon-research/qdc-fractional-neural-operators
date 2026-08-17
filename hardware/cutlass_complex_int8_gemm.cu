// Four-real-GEMM primitive for complex INT8 spectral products on SM80-class GPUs.
// C = (Ar Br - Ai Bi) + i(Ar Bi + Ai Br), with INT32 accumulation.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cstdint>
#include <stdexcept>

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using GemmI8 = cutlass::gemm::device::Gemm<
    int8_t, LayoutA,
    int8_t, LayoutB,
    int32_t, LayoutC,
    int32_t>;

static void run_gemm(int M, int N, int K,
                     int8_t const* A, int8_t const* B,
                     int32_t* C, int alpha, int beta) {
  GemmI8 gemm;
  GemmI8::Arguments args(
      {M, N, K},
      {A, K}, {B, K}, {C, N}, {C, N},
      {alpha, beta});
  auto status = gemm(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS INT8 GEMM failed");
  }
}

// Temporary buffers T0..T3 are caller-owned device pointers of M*N int32 values.
extern "C" void complex_int8_four_gemm(
    int M, int N, int K,
    int8_t const* Ar, int8_t const* Ai,
    int8_t const* Br, int8_t const* Bi,
    int32_t* T0, int32_t* T1, int32_t* T2, int32_t* T3) {
  run_gemm(M,N,K,Ar,Br,T0, 1,0);  // Ar Br
  run_gemm(M,N,K,Ai,Bi,T1, 1,0);  // Ai Bi
  run_gemm(M,N,K,Ar,Bi,T2, 1,0);  // Ar Bi
  run_gemm(M,N,K,Ai,Br,T3, 1,0);  // Ai Br
  // Combine T0-T1 and T2+T3 with a lightweight CUDA kernel in the calling integration.
  // Keeping combination separate makes the INT32 accumulation path explicit.
}
