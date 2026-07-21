#pragma once

namespace rvnpu::kernels {

inline constexpr const char *kTxInit = "__tx_init";
inline constexpr const char *kMatMulFp32 = "triton_matmul_fp32_kernel";
inline constexpr const char *kAddFp32 = "kernel_add_fp32";
inline constexpr const char *kMulFp32 = "kernel_mul_fp32";
inline constexpr const char *kSigmoidFp32 = "sigmoid_f32";
inline constexpr const char *kIm2ColNchwFp32 = "im2col_nchw_fp32";
inline constexpr const char *kMaxPool2dNchwFp32 = "maxpool2d_nchw_fp32";
inline constexpr const char *kTopKLastDimFp32 = "topk_lastdim_fp32";
inline constexpr const char *kNonMaxSuppressionFp32 =
    "non_max_suppression_fp32";

} // namespace rvnpu::kernels
