#include <torch/extension.h>

#include <vector>
#include <iostream>

// CUDA forward declarations

std::vector<torch::Tensor> hv_cuda_forward(
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots);

std::vector<torch::Tensor> hv_cuda_backward(
    torch::Tensor grad_grid,
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots);

    
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> hv_forward(
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots) {
  CHECK_INPUT(points);
  CHECK_INPUT(xyz_labels);
  CHECK_INPUT(scale_labels);
  CHECK_INPUT(obj_labels);
  CHECK_INPUT(res);
  CHECK_INPUT(num_rots);

  return hv_cuda_forward(points, xyz_labels, scale_labels, obj_labels, res, num_rots);
}

std::vector<torch::Tensor> hv_backward(
    torch::Tensor grad_grid,
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots) {
  CHECK_INPUT(grad_grid);
  CHECK_INPUT(points);
  CHECK_INPUT(xyz_labels);
  CHECK_INPUT(scale_labels);
  CHECK_INPUT(obj_labels);
  CHECK_INPUT(res);
  CHECK_INPUT(num_rots);

  return hv_cuda_backward(
      grad_grid,
      points,
      xyz_labels,
      scale_labels,
      obj_labels,
      res,
      num_rots);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hv_forward, "hv forward (CUDA)");
  m.def("backward", &hv_backward, "hv backward (CUDA)");
}