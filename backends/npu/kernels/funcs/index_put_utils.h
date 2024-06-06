#pragma once

#include <vector>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void NonZeroKernel(const Context& dev_ctx,
                   const phi::DenseTensor& condition,
                   phi::DenseTensor* out);

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        int num,
                        const phi::Scalar& axis_scalar,
                        std::vector<phi::DenseTensor*> outs);

template <typename T, typename Context>
std::vector<const phi::DenseTensor*> DealWithBoolIndices(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& indices_v,
    std::vector<phi::DenseTensor>* tmp_indices_v) {
  auto stream = dev_ctx.stream();
  std::vector<const phi::DenseTensor*> res;

  bool contains_bool_tensor = false;
  for (size_t i = 0; i < indices_v.size(); ++i) {
    if (indices_v[i]->dtype() == phi::DataType::BOOL) {
      contains_bool_tensor = true;
      break;
    }
  }

  if (contains_bool_tensor) {
    for (size_t i = 0; i < indices_v.size(); ++i) {
      if (indices_v[i]->dtype() == phi::DataType::BOOL) {
        int rank = indices_v[i]->dims().size();
        PADDLE_ENFORCE_GE(rank,
                          1UL,
                          phi::errors::InvalidArgument(
                              "the only bool tensor in indices should "
                              "have number of dimension at least 1"));
        phi::DenseTensor nonzero_indices;
        nonzero_indices.set_meta(
            {phi::DataType::INT64, common::make_ddim({-1, rank})});
        dev_ctx.template Alloc<int64_t>(&nonzero_indices);
        custom_kernel::NonZeroKernel<bool, Context>(
            dev_ctx, *indices_v[i], &nonzero_indices);

        if (nonzero_indices.numel() == 0) {
          std::vector<const phi::DenseTensor*> empty_indices;
          return empty_indices;
        }

        std::vector<phi::DenseTensor*> integer_indices(rank, nullptr);
        const int tmp_ix = tmp_indices_v->size();
        for (int i = 0; i < rank; ++i) {
          phi::DenseTensor tmp_tensor;
          tmp_tensor.set_meta({phi::DataType::INT64,
                               common::make_ddim({nonzero_indices.dims()[0]})});
          dev_ctx.template Alloc<int64_t>(&tmp_tensor);
          tmp_indices_v->emplace_back(tmp_tensor);
        }
        for (int i = 0; i < rank; ++i) {
          integer_indices[i] = &((*tmp_indices_v)[i + tmp_ix]);
        }
        custom_kernel::SplitWithNumKernel<int64_t, Context>(
            dev_ctx, nonzero_indices, rank, 1, integer_indices);
      } else if ((indices_v[i]->dtype() == phi::DataType::INT64) ||
                 (indices_v[i]->dtype() == phi::DataType::INT32)) {
        tmp_indices_v->emplace_back(*indices_v[i]);
      } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "data type of tensor in indices must be int32, int64 or bool"));
      }
    }

    res.reserve(tmp_indices_v->size());
    for (size_t i = 0; i < tmp_indices_v->size(); ++i) {
      res.emplace_back(&((*tmp_indices_v)[i]));
    }
  } else {
    res = indices_v;
  }
  return res;
}

}  // namespace custom_kernel