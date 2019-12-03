#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t act(scalar_t z) {
  return 0.5 * z * (tanh(z) + 1);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_act(scalar_t z) {
  return 0.5 * tanh(z) + 0.5 * z * pow((1/cosh(z)),2) + 0.5;
}

template <typename scalar_t>
__global__ void cust_act_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output) {

  const int n = blockIdx.y; //batch index
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // column index
  if (c < input.size(2)){
    output[n][c] = act(input[n][c]);
  }
}

torch::Tensor cust_act_cuda_forward(torch::Tensor input) {
  auto output = torch::zeros_like(input);

  const auto batch_size = input.size(0);
  const auto state_size = input.size(1);
  
  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  
  AT_DISPATCH_FLOATING_TYPES(input.type(), "cust_act_forward_cuda", ([&] {
    cust_act_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
    );
  }));

  return output;
}

template <typename scalar_t>
__global__ void cust_act_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_out,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_inp) {
  
  const int n = blockIdx.y; //batch index
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // column index
  if (c < grad_inp.size(2)){
    grad_inp[n][c] = d_act(input[n][c]) * grad_out[n][c];
  }
}

torch::Tensor cust_act_cuda_backward(torch::Tensor input, torch::Tensor grad_out) {
  auto grad_inp = torch::zeros_like(grad_out);

  const auto batch_size = grad_inp.size(0);
  const auto state_size = grad_inp.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "cust_act_backward_cuda", ([&] {
    cust_act_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_inp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return grad_inp;
}