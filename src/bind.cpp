#include <torch/extension.h>

at::Tensor integral_image_forward(
    at::Tensor input);

at::Tensor integral_image_backward(
    at::Tensor grad_output);

at::Tensor box_convolution_forward(
    at::Tensor input_integrated,
    at::Tensor x_min, at::Tensor x_max,
    at::Tensor y_min, at::Tensor y_max, at::Tensor input);

at::Tensor box_convolution_backward(
    at::Tensor input_integrated,
    at::Tensor x_min, at::Tensor x_max,
    at::Tensor y_min, at::Tensor y_max,
    at::Tensor grad_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image_forward"  , &integral_image_forward  , "Integral image, forward"  );
    m.def("integral_image_backward" , &integral_image_backward , "Integral image, backward" );
    m.def("box_convolution_forward" , &box_convolution_forward , "Box convolution, forward" );
    m.def("box_convolution_backward", &box_convolution_backward, "Box convolution, backward");
}
