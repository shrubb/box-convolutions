#include <torch/extension.h>

at::Tensor integral_image(
    at::Tensor input);

at::Tensor box_convolution_forward(
    at::Tensor input_integrated,
    at::Tensor x_min, at::Tensor x_max,
    at::Tensor y_min, at::Tensor y_max);

std::vector<at::Tensor> box_convolution_backward(
    at::Tensor input_integrated,
    at::Tensor x_min, at::Tensor x_max,
    at::Tensor y_min, at::Tensor y_max,
    at::Tensor grad_output,
    bool input_needs_grad,
    bool x_min_needs_grad, bool x_max_needs_grad,
    bool y_min_needs_grad, bool y_max_needs_grad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image", &integral_image, "Integral image");
    m.def("box_convolution_forward" , &box_convolution_forward , "Box convolution, forward" );
    m.def("box_convolution_backward", &box_convolution_backward, "Box convolution, backward");
}
