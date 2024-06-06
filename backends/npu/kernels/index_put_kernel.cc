// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/index_put_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void IndexPutNPUKernel(const Context &dev_ctx,
                       const phi::DenseTensor &x,
                       const std::vector<const phi::DenseTensor *> &indices,
                       const phi::DenseTensor &value,
                       bool accumulate,
                       phi::DenseTensor *out) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      phi::errors::InvalidArgument("The data type of tensor value must be same "
                                   "to the data type of tensor x."));

  PADDLE_ENFORCE_EQ(indices.empty(),
                    false,
                    phi::errors::InvalidArgument("Indices cannot be empty."));

  const size_t total_dims = x.dims().size();
  PADDLE_ENFORCE_LE(total_dims,
                    6,
                    phi::errors::InvalidArgument(
                        "Dims of input tensor should be less than 7."));

  auto stream = dev_ctx.stream();

  std::cout << "call index put npu kernel" << std::endl;

  //   for (auto i : indices) {
  //     std::vector<bool> tmp;
  //     TensorToVector(dev_ctx, *i, dev_ctx, &tmp);
  //     std::cout << "index: " << std::endl;
  //     for (auto i : tmp) {
  //       std::cout << i << " ";
  //     }
  //     std::cout << std::endl;
  //   }

  std::vector<phi::DenseTensor> tmp_args;
  std::vector<const phi::DenseTensor *> int_indices_v =
      DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  std::cout << "int_indices_v: " << std::endl;
  for (auto i : int_indices_v) {
    std::cout << i << std::endl;
  }
  std::cout << std::endl;

  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);
  phi::DenseTensorMeta out_meta = {x.dtype(), x.dims(), x.layout()};
  out->set_meta(out_meta);
}

template <typename T, typename Context>
void IndexPutGradNPUKernel(const Context &dev_ctx,
                           const phi::DenseTensor &x,
                           const std::vector<const phi::DenseTensor *> &indices,
                           const phi::DenseTensor &value,
                           const phi::DenseTensor &out_grad,
                           bool accumlate,
                           phi::DenseTensor *x_grad,
                           phi::DenseTensor *value_grad) {
  auto stream = dev_ctx.stream();

  std::cout << "call index put grad npu kernel" << std::endl;
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_put,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexPutNPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(index_put_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexPutGradNPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}