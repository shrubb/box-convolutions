#include <torch/extension.h>
#include <TH/THGeneral.h>

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor integral_image_forward(at::Tensor input) {
    CHECK_CONTIGUOUS(input);
    if (input.is_cuda()) {
    	THError("NYI: GPU integral_image_forward");
    } else {
        THError("NYI: CPU integral_image_forward");
    }
}

at::Tensor integral_image_backward(at::Tensor grad_output) {
	CHECK_CONTIGUOUS(grad_output);
    if (grad_output.is_cuda()) {
    	THError("NYI: GPU integral_image_backward");
    } else {
        THError("NYI: CPU integral_image_backward");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image_forward", &integral_image_forward, "Integral image, forward");
    m.def("integral_image_backward", &integral_image_backward, "Integral image, backward");
}
