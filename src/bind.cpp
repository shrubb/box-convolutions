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
    const bool input_needs_grad,
    const bool x_min_needs_grad, const bool x_max_needs_grad,
    const bool y_min_needs_grad, const bool y_max_needs_grad,
    const float max_input_h, const float max_input_w);

void clip_parameters(
    at::Tensor x_min, at::Tensor x_max,
    at::Tensor y_min, at::Tensor y_max,
    const float max_input_h, const float max_input_w, const bool exact);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image", &integral_image, "Integral image");
    m.def("box_convolution_forward" , &box_convolution_forward , "Box convolution, forward" );
    m.def("box_convolution_backward", &box_convolution_backward, "Box convolution, backward");
    m.def("clip_parameters", &clip_parameters, "Box convolution, clip parameters");
}
