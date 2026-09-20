#pragma once

namespace rvnpu::kernels {

inline constexpr const char *kTxInit = "__tx_init";
inline constexpr const char *kMatMulFp32 = "triton_matmul_fp32_kernel";
inline constexpr const char *kAddFp32 = "kernel_add_fp32";
inline constexpr const char *kMulFp32 = "kernel_mul_fp32";
inline constexpr const char *kSubFp32 = "kernel_sub_fp32";
inline constexpr const char *kDivFp32 = "kernel_div_fp32";
inline constexpr const char *kConcatFp32 = "concat_fp32";
inline constexpr const char *kSplitFp32 = "split_fp32";
inline constexpr const char *kSliceFp32 = "slice_fp32";
inline constexpr const char *kSliceInt64 = "slice_int64";
inline constexpr const char *kResizeNearestNchwFp32 =
    "resize_nearest_nchw_fp32";
inline constexpr const char *kTranspose0213Fp32 = "transpose_0213_fp32";
inline constexpr const char *kTranspose021Fp32 = "transpose_021_fp32";
inline constexpr const char *kTranspose10Int64 = "transpose_10_int64";
inline constexpr const char *kSoftmaxAxis1Fp32 = "softmax_axis1_fp32";
inline constexpr const char *kReduceMaxLastDimFp32 =
    "reduce_max_lastdim_fp32";
inline constexpr const char *kArgMaxLastDimFp32 = "argmax_lastdim_fp32";
inline constexpr const char *kSigmoidFp32 = "sigmoid_f32";
inline constexpr const char *kSiluFp32 = "silu_f32";
inline constexpr const char *kIm2ColNchwFp32 = "im2col_nchw_fp32";
inline constexpr const char *kIm2ColNchwFp32Padded =
    "im2col_nchw_fp32_padded";
inline constexpr const char *kMaxPool2dNchwFp32 = "maxpool2d_nchw_fp32";
inline constexpr const char *kTopKLastDimFp32 = "topk_lastdim_fp32";
inline constexpr const char *kNonMaxSuppressionFp32 =
    "non_max_suppression_fp32";
inline constexpr const char *kGatherNdFp32 = "gather_nd_fp32";
inline constexpr const char *kGatherNdInt64 = "gather_nd_int64";

} // namespace rvnpu::kernels
