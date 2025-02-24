// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/common_ops.h"

namespace custom_kernel {
namespace {
std::unordered_map<phi::DataType, phi::DataType> kDataTypeTrans64To32 = {
    {phi::DataType::INT64, phi::DataType::INT32},
    {phi::DataType::FLOAT64, phi::DataType::FLOAT32},
};

inline bool IsValidPermutation(const std::vector<int64_t>& permutation) {
  auto size = permutation.size();
  std::vector<bool> flags(size, false);
  for (size_t i = 0; i < size; ++i) {
    auto k = permutation[i];
    if (k >= 0 && k < size && !flags[k])
      flags[k] = true;
    else
      return false;
  }
  return true;
}

inline std::vector<int64_t> ReorderVector(
    const std::vector<int64_t>& src, const std::vector<int64_t>& permutation) {
  PADDLE_ENFORCE(
      permutation.size() == src.size() && IsValidPermutation(permutation),
      phi::errors::InvalidArgument("Invalid permutation."));

  std::vector<int64_t> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = src[permutation[i]];
  }
  return dst;
}

std::vector<int64_t> InferBroadcastDimMap(
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims) {
  auto src_rank = src_dims.size();
  auto dst_rank = dst_dims.size();
  PADDLE_ENFORCE_LE(src_rank,
                    dst_rank,
                    phi::errors::InvalidArgument(
                        "src_rank should LE dst_rank, but get %zu vs %zu",
                        src_rank,
                        dst_rank));
  if (src_rank == 0) {
    return dst_rank == 0
               ? std::vector<int64_t>{0}
               : std::vector<int64_t>{static_cast<int64_t>(dst_rank - 1)};
  }
  std::vector<int64_t> dim_maps(src_rank);
  for (int64_t i = src_rank - 1; i >= 0; --i) {
    int64_t offset = src_rank - 1 - i;
    int64_t dst_index = dst_rank - 1 - offset;
    int64_t src_dim_val = src_dims[i];
    int64_t dst_dim_val = dst_dims[dst_index];
    PADDLE_ENFORCE(
        src_dim_val == dst_dim_val || src_dim_val == 1,
        phi::errors::InvalidArgument("The dim of src tensor(%ld"
                                     ") must match the dim of dst tensor (%ld"
                                     ") at non-singleton dimension %ld",
                                     src_dim_val,
                                     dst_dim_val,
                                     i));
    dim_maps[i] = dst_index;
  }
  return dim_maps;
}

inline std::string GetDataTypePairKey(const phi::DataType src_type,
                                      const phi::DataType dst_type) {
  return phi::DataTypeToString(src_type) + "_to_" +
         phi::DataTypeToString(dst_type);
}
}  // namespace

phi::DenseTensor MaybeCreateOrTrans(
    const phi::CustomContext& dev_ctx,
    const phi::DenseTensor& src,
    const std::unordered_map<phi::DataType, phi::DataType>& tans_map,
    bool need_cast) {
  auto src_dtype = src.dtype();
  if (tans_map.count(src_dtype) == 0) {
    return src;
  }
  phi::DenseTensor dst;
  if (need_cast) {
    custom_kernel::Cast(dev_ctx, src, tans_map.at(src_dtype), &dst);
  } else {
    auto meta = src.meta();
    meta.dtype = tans_map.at(src_dtype);
    dst.set_meta(meta);
    dev_ctx.Alloc(&dst, dst.dtype());
  }
  return dst;
}

phi::DenseTensor MaybeCreateOrTrans64To32bits(const phi::CustomContext& dev_ctx,
                                              const phi::DenseTensor& src,
                                              bool need_cast) {
  return MaybeCreateOrTrans(dev_ctx, src, kDataTypeTrans64To32, need_cast);
}

phi::DenseTensor MaybeCreateOrTransFp16ToFp32(const phi::CustomContext& dev_ctx,
                                              const phi::DenseTensor& src,
                                              bool need_cast) {
  static const std::unordered_map<phi::DataType, phi::DataType> kFp16ToFp32 = {
      {phi::DataType::FLOAT16, phi::DataType::FLOAT32},
  };
  return MaybeCreateOrTrans(dev_ctx, src, kFp16ToFp32, need_cast);
}

void MaybeTransResult(const phi::CustomContext& dev_ctx,
                      const phi::DenseTensor& result,
                      phi::DenseTensor* dst) {
  auto dst_dtype = dst->dtype();
  if (dst_dtype == result.dtype()) {
    return;
  }
  custom_kernel::Cast(dev_ctx, result, dst_dtype, dst);
}

void Broadcast(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& src,
               phi::DenseTensor* dst) {
  if (src.numel() <= 0) {
    VLOG(1) << "Common op Broadcast, src numel:" << src.numel()
            << ", will do nothing.";
    return;
  }
  auto src_dims = phi::vectorize(src.dims());
  auto dst_dims = phi::vectorize(dst->dims());
  auto dims_map = InferBroadcastDimMap(src_dims, dst_dims);
  auto alpha = phi::Scalar(1.0f);
  auto beta = phi::Scalar(0.0f);
  VLOG(3) << "Broadcast: dim map:" << VectorToStr<int64_t>(dims_map)
          << ", dst:" << dst->dims();
  static std::unordered_map<phi::DataType, phi::DataType> kDataTypeTransMap = {
      {phi::DataType::INT64, phi::DataType::INT32},
      {phi::DataType::FLOAT64, phi::DataType::FLOAT32},
  };
  phi::DenseTensor src_tmp = MaybeCreateOrTrans64To32bits(dev_ctx, src);
  phi::DenseTensor dst_tmp = MaybeCreateOrTrans64To32bits(dev_ctx, *dst, false);
  LAUNCH_TOPSOP(
      topsopBroadcast, dev_ctx, dst_tmp, src_tmp, dims_map, alpha, beta);
  MaybeTransResult(dev_ctx, dst_tmp, dst);
}

phi::DenseTensor Broadcast(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& src,
                           const std::vector<int64_t>& output_shapes) {
  auto meta = phi::DenseTensorMeta(src.dtype(), phi::make_ddim(output_shapes));
  phi::DenseTensor dst = TensorEmpty(dev_ctx, meta);
  Broadcast(dev_ctx, src, &dst);
  return dst;
}

namespace {
bool IsCastSupport(const phi::DataType src_type, const phi::DataType dst_type) {
  static const std::unordered_set<std::string> kSupportedCast = {
      // ******************* bool convert ***************** //
      // bool <--> int32
      GetDataTypePairKey(phi::DataType::BOOL, phi::DataType::INT32),
      GetDataTypePairKey(phi::DataType::INT32, phi::DataType::BOOL),
      // bool <--> float16
      GetDataTypePairKey(phi::DataType::BOOL, phi::DataType::FLOAT16),
      GetDataTypePairKey(phi::DataType::FLOAT16, phi::DataType::BOOL),
      // bool <--> float32
      GetDataTypePairKey(phi::DataType::BOOL, phi::DataType::FLOAT32),
      GetDataTypePairKey(phi::DataType::FLOAT32, phi::DataType::BOOL),

      // ******************** 64 bits ******************** //
      // int32 <--> int64
      GetDataTypePairKey(phi::DataType::INT32, phi::DataType::INT64),
      GetDataTypePairKey(phi::DataType::INT64, phi::DataType::INT32),

      // ***************** int to float16 **************** //
      // int32 <--> float16
      GetDataTypePairKey(phi::DataType::INT32, phi::DataType::FLOAT16),
      GetDataTypePairKey(phi::DataType::FLOAT16, phi::DataType::INT32),

      // ***************** int to float32 *************** //
      // int8 <--> float32
      GetDataTypePairKey(phi::DataType::INT8, phi::DataType::FLOAT32),
      GetDataTypePairKey(phi::DataType::FLOAT32, phi::DataType::INT8),
      // int16 <--> float32
      GetDataTypePairKey(phi::DataType::INT16, phi::DataType::FLOAT32),
      GetDataTypePairKey(phi::DataType::FLOAT32, phi::DataType::INT16),
      // int32 <--> float32
      GetDataTypePairKey(phi::DataType::INT32, phi::DataType::FLOAT32),
      GetDataTypePairKey(phi::DataType::FLOAT32, phi::DataType::INT32),

      // ***************** float convert ***************** //
      // float16 <--> float32
      GetDataTypePairKey(phi::DataType::FLOAT16, phi::DataType::FLOAT32),
      GetDataTypePairKey(phi::DataType::FLOAT32, phi::DataType::FLOAT16),

      // ***************** int convert ****************** //
      // int8 <--> int16
      GetDataTypePairKey(phi::DataType::INT8, phi::DataType::INT16),
      GetDataTypePairKey(phi::DataType::INT16, phi::DataType::INT8),
      // int8 <--> int32
      GetDataTypePairKey(phi::DataType::INT8, phi::DataType::INT32),
      GetDataTypePairKey(phi::DataType::INT32, phi::DataType::INT8),
      // int16 <--> int32
      GetDataTypePairKey(phi::DataType::INT16, phi::DataType::INT32),
      GetDataTypePairKey(phi::DataType::INT32, phi::DataType::INT16),
      // uint8 <--> uint16
      GetDataTypePairKey(phi::DataType::UINT8, phi::DataType::UINT16),
      GetDataTypePairKey(phi::DataType::UINT16, phi::DataType::UINT8),
      // uint8 <--> uint32
      GetDataTypePairKey(phi::DataType::UINT8, phi::DataType::UINT32),
      GetDataTypePairKey(phi::DataType::UINT32, phi::DataType::UINT8),
      // uint16 <--> uint16
      GetDataTypePairKey(phi::DataType::UINT16, phi::DataType::UINT32),
      GetDataTypePairKey(phi::DataType::UINT32, phi::DataType::UINT16),
  };
  return (kSupportedCast.count(GetDataTypePairKey(src_type, dst_type)) > 0);
}

phi::DataType IntermediateDtypeToCast(const phi::DataType src_type,
                                      const phi::DataType dst_type) {
  static const std::unordered_map<std::string, phi::DataType>
      kSupportedIndirectCast = {
          // float32 <--> int64
          {GetDataTypePairKey(phi::DataType::FLOAT32, phi::DataType::INT64),
           phi::DataType::INT32},
          {GetDataTypePairKey(phi::DataType::INT64, phi::DataType::FLOAT32),
           phi::DataType::INT32},

          // float16 <--> int64
          {GetDataTypePairKey(phi::DataType::FLOAT16, phi::DataType::INT64),
           phi::DataType::INT32},
          {GetDataTypePairKey(phi::DataType::INT64, phi::DataType::FLOAT16),
           phi::DataType::INT32},

          // bool <--> int64
          {GetDataTypePairKey(phi::DataType::BOOL, phi::DataType::INT64),
           phi::DataType::INT32},
          {GetDataTypePairKey(phi::DataType::INT64, phi::DataType::BOOL),
           phi::DataType::INT32},

          // int8 <--> int64
          {GetDataTypePairKey(phi::DataType::INT8, phi::DataType::INT64),
           phi::DataType::INT32},
          {GetDataTypePairKey(phi::DataType::INT64, phi::DataType::INT8),
           phi::DataType::INT32},

          // int16 <--> int64
          {GetDataTypePairKey(phi::DataType::INT16, phi::DataType::INT64),
           phi::DataType::INT32},
          {GetDataTypePairKey(phi::DataType::INT64, phi::DataType::INT16),
           phi::DataType::INT32},
      };

  auto key = GetDataTypePairKey(src_type, dst_type);
  return ((kSupportedIndirectCast.count(key) > 0)
              ? kSupportedIndirectCast.at(key)
              : phi::DataType::UNDEFINED);
}

void CastImpl(const phi::CustomContext& dev_ctx,
              const phi::DenseTensor& x,
              const phi::DataType& dtype,
              phi::DenseTensor* out) {
  auto meta = x.meta();
  meta.dtype = dtype;
  out->set_meta(meta);
  dev_ctx.Alloc(out, out->dtype());
  if (x.dtype() == dtype) {
    TensorCopy(dev_ctx, x, false, out);
    return;
  }
  auto x_tensor = CreateTopsatenTensor(x);
  auto out_tensor = CreateTopsatenTensor(*out);
  topsatenDataType_t topsaten_dtype = DataTypeToTopsatenDataType(dtype);
  bool non_blocking = true;
  bool copy = true;
  topsatenMemoryFormat_t topsaten_format = TOPSATEN_MEMORY_PRESERVE;
  auto stream = static_cast<topsStream_t>(dev_ctx.stream());
  VLOG(3) << "CastImpl, use topsatenTo, convert "
          << phi::DataTypeToString(x.dtype()) << " to "
          << phi::DataTypeToString(dtype) << ", stream: " << stream;
  ATEN_OP_CALL_MAYBE_SYNC(topsaten::topsatenTo(out_tensor,
                                               x_tensor,
                                               topsaten_dtype,
                                               non_blocking,
                                               copy,
                                               topsaten_format,
                                               stream),
                          dev_ctx);
  //   LAUNCH_TOPSOP(topsopConvert, dev_ctx, *out, x);
  //   LAUNCH_TOPSATENOP(topsatenTo, dev_ctx, *out, x,
  //                     dtype, non_blocking,
  //                     copy, topsaten_format);
}

#define FOR_EACH_DATA_TYPE(_)                   \
  _(bool, phi::DataType::BOOL)                  \
  _(int8_t, phi::DataType::INT8)                \
  _(uint8_t, phi::DataType::UINT8)              \
  _(int16_t, phi::DataType::INT16)              \
  _(uint16_t, phi::DataType::UINT16)            \
  _(int32_t, phi::DataType::INT32)              \
  _(uint32_t, phi::DataType::UINT32)            \
  _(int64_t, phi::DataType::INT64)              \
  _(uint64_t, phi::DataType::UINT64)            \
  _(phi::bfloat16, phi::DataType::BFLOAT16)     \
  _(phi::float16, phi::DataType::FLOAT16)       \
  _(float, phi::DataType::FLOAT32)              \
  _(double, phi::DataType::FLOAT64)             \
  _(phi::complex64, phi::DataType::COMPLEX64)   \
  _(phi::complex128, phi::DataType::COMPLEX128) \
  _(phi::pstring, phi::DataType::PSTRING)

#define CALL_CPU_CAST_KERNEL(cpp_type, data_type) \
  case data_type:                                 \
    phi::CastKernel<cpp_type, phi::CPUContext>(   \
        dev_ctx_cpu, x_cpu, dtype, out_cpu);      \
    break;

void CallCPUCastImpl(const phi::CPUContext& dev_ctx_cpu,
                     const phi::DenseTensor& x_cpu,
                     const phi::DataType& dtype,
                     phi::DenseTensor* out_cpu) {
  switch (x_cpu.dtype()) { FOR_EACH_DATA_TYPE(CALL_CPU_CAST_KERNEL) }
}

#undef CALL_CPU_CAST_KERNEL
#undef FOR_EACH_DATA_TYPE

void CastCPUImpl(const phi::CustomContext& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DataType& dtype,
                 phi::DenseTensor* out) {
  auto meta = x.meta();
  meta.dtype = dtype;
  out->set_meta(meta);
  dev_ctx.Alloc(out, out->dtype());
  if (x.dtype() == dtype) {
    TensorCopy(dev_ctx, x, false, out);
    return;
  }

  // 1. Copy x to CPU
  ContextPinnedGuard ctx_pinned_guard(dev_ctx);
  phi::DenseTensor x_cpu;
  x_cpu.set_meta(x.meta());
  TensorCopy(dev_ctx, x, false, &x_cpu, phi::CPUPlace());
  dev_ctx.Wait();

  // 2. Call the CPU implementation
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(&(dev_ctx.GetHostAllocator()));
  dev_ctx_cpu.SetHostAllocator(&(dev_ctx.GetHostAllocator()));
  phi::DenseTensor out_cpu;
  out_cpu.set_meta(meta);
  CallCPUCastImpl(dev_ctx_cpu, x_cpu, dtype, &out_cpu);
  dev_ctx.Wait();

  // 3. Copy result to device
  TensorCopy(dev_ctx, out_cpu, false, out);
  dev_ctx.Wait();
}
}  // namespace

void Cast(const phi::CustomContext& dev_ctx,
          const phi::DenseTensor& x,
          const phi::DataType& dtype,
          phi::DenseTensor* out) {
  std::string key = "convert_" + GetDataTypePairKey(x.dtype(), dtype);
  PADDLE_GCU_KERNEL_TRACE(key);
  auto meta = x.meta();
  meta.dtype = dtype;
  out->set_meta(meta);
  dev_ctx.Alloc(out, out->dtype());
  if (x.dtype() == dtype) {
    TensorCopy(dev_ctx, x, false, out);
    return;
  }
  if (IsCastSupport(x.dtype(), dtype)) {
    CastImpl(dev_ctx, x, dtype, out);
    return;
  }
  auto media_type = IntermediateDtypeToCast(x.dtype(), dtype);
  if (media_type != phi::DataType::UNDEFINED) {
    VLOG(3) << "Cast intermediately, convert "
            << phi::DataTypeToString(x.dtype()) << " to "
            << phi::DataTypeToString(media_type) << " to "
            << phi::DataTypeToString(dtype);
    phi::DenseTensor tmp;
    CastImpl(dev_ctx, x, media_type, &tmp);
    CastImpl(dev_ctx, tmp, dtype, out);
  } else {
    VLOG(3) << "[CPU_KERNEL] Use CastCPUImpl, convert "
            << phi::DataTypeToString(x.dtype()) << " to "
            << phi::DataTypeToString(dtype);
    CastCPUImpl(dev_ctx, x, dtype, out);
  }
}

phi::DenseTensor Cast(const phi::CustomContext& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DataType& dtype) {
  phi::DenseTensor out;
  Cast(dev_ctx, x, dtype, &out);
  return out;
}

phi::DenseTensor ReshapeWithoutCopy(const phi::DenseTensor& src,
                                    const std::vector<int64_t>& out_shapes) {
  PADDLE_ENFORCE_EQ(
      src.numel(),
      phi::product(phi::make_ddim(out_shapes)),
      phi::errors::InvalidArgument(
          "The memory size before and after reshape should be the same."));
  phi::DenseTensor dst(src);
  dst.Resize(phi::make_ddim(out_shapes));
  return dst;
}

void Transpose(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& axis,
               phi::DenseTensor* out) {
  // method 1: topsop impl
  // auto alpha = phi::Scalar(1.0f);
  // auto beta = phi::Scalar(0.0f);
  // LAUNCH_TOPSOP(topsopTranspose,
  //               dev_ctx,
  //               *out, x, axis, alpha, beta);
  // method 2: topsaten impl
  auto aten_x = CreateTopsatenTensor(x);
  auto aten_out = CreateTopsatenTensor(*out);
  auto aten_tmp = aten_x;
  auto perm = IntArrayToTopsatenSize(axis);
  LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(
      topsatenPermute, dev_ctx, aten_tmp, aten_x, perm);
  LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenContiguous,
                                      dev_ctx,
                                      aten_out,
                                      aten_tmp,
                                      TOPSATEN_MEMORY_CONTIGUOUS);
}

phi::DenseTensor Transpose(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int64_t>& axis) {
  // infer dst shape
  std::vector<int64_t> src_dims = phi::vectorize(x.dims());
  std::vector<int64_t> dst_dims = ReorderVector(src_dims, axis);

  phi::DenseTensor dst_tensor;
  phi::DenseTensorMeta meta(x.dtype(), phi::make_ddim(dst_dims));
  dst_tensor.set_meta(meta);
  dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
  Transpose(dev_ctx, x, axis, &dst_tensor);
  return dst_tensor;
}

phi::DenseTensor TensorEmpty(const phi::CustomContext& dev_ctx,
                             const phi::DenseTensorMeta& meta) {
  phi::DenseTensor output_tensor;
  output_tensor.set_meta(meta);
  dev_ctx.Alloc(&output_tensor, output_tensor.dtype());
  return output_tensor;
}

phi::DenseTensor TensorOnes(const phi::CustomContext& dev_ctx,
                            const phi::DenseTensorMeta& meta) {
  phi::DenseTensor out = TensorEmpty(dev_ctx, meta);
  auto shape = phi::vectorize(meta.dims);
  LAUNCH_TOPSATENOP(topsatenOnes, dev_ctx, out, shape, meta.dtype);
  return out;
}

phi::DenseTensor TensorZeros(const phi::CustomContext& dev_ctx,
                             const phi::DenseTensorMeta& meta) {
  phi::DenseTensor out = TensorEmpty(dev_ctx, meta);
  auto shape = phi::vectorize(meta.dims);
  LAUNCH_TOPSATENOP(topsatenZeros, dev_ctx, out, shape, meta.dtype);
  return out;
}
}  // namespace custom_kernel
