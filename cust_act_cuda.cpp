#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor cust_act_cuda_forward(torch::Tensor input);

torch::Tensor cust_act_cuda_backward(torch::Tensor input, torch::Tensor grad_out);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor cust_act_forward(torch::Tensor input) {
  CHECK_INPUT(input);

  return cust_act_cuda_forward(input);
}

torch::Tensor cust_act_backward(torch::Tensor input, torch::Tensor grad_out) {
  CHECK_INPUT(input);
  CHECK_INPUT(grad_out);

  return cust_act_cuda_backward(input, grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cust_act_forward, "Custom activation forward (CUDA)");
  m.def("backward", &cust_act_backward, "Custom activation (CUDA)");
}