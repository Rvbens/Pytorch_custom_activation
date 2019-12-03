#include <torch/extension.h>
#include <iostream>

at::Tensor act_forward(torch::Tensor input) {
  auto output = 0.5 * input * (input.tanh() + 1);
  return output;
}

torch::Tensor act_backward(torch::Tensor input, torch::Tensor grad_out) {
  auto d_input = 0.5 * input.tanh() + 0.5 * input * (1/torch::cosh(input)).pow(2) + 0.5;
  return d_input*grad_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &act_forward, "act forward");
  m.def("backward", &act_backward, "act backward");
}