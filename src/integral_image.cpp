#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <TH/THGeneral.h>

// #include <integral_image_cuda.h>

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor integral_image_forward(at::Tensor input) {
    CHECK_CONTIGUOUS(input);
    AT_CHECK(input.dim() >= 2, "integral image input must have >=2 dimensions")

    const int h = input.size(input.dim() - 2);
    const int w = input.size(input.dim() - 1);
    int nChannels = 1;
    for (int i = 0; i < input.dim() - 2; ++i) {
        nChannels *= input.size(i);
    }

    // The result will have extra row and column for zeros
    // TODO maybe it's faster to eliminate them
    std::vector<int64_t> outputSize(input.sizes().begin(), input.sizes().end());
    ++outputSize[input.dim() - 2];
    ++outputSize[input.dim() - 1];

    auto output = at::empty(outputSize, input.options());

    if (input.is_cuda()) {
        THError("NYI: GPU integral_image_forward");
    } else {
        AT_DISPATCH_ALL_TYPES(input.type(), "integral_image_forward_cpu", ([&] {
            using accscalar_t = at::acc_type<scalar_t, false>;

            scalar_t *inputPtr = input.data<scalar_t>();
            scalar_t *outputPtr = output.data<scalar_t>();
            
            for (int c = 0; c < nChannels; ++c) {
                // Fill the 0-th row
                std::memset(outputPtr, 0, (w+1)*sizeof(scalar_t));
                
                // Fill the rest
                for (int row = 0; row < h; ++row) {
                    outputPtr[(row+1)*(w+1)] = 0.0;

                    accscalar_t sum = 0.0;
                    for (int col = 0; col < w; ++col) {
                        sum += inputPtr[row*w + col];
                        outputPtr[(row+1)*(w+1) + (col+1)] = sum + outputPtr[row*(w+1) + (col+1)];
                    }
                }

                inputPtr += h*w;
                outputPtr += (h+1)*(w+1);
            }
        }));
    }

    return output;
}

at::Tensor integral_image_backward(at::Tensor grad_output) {
    CHECK_CONTIGUOUS(grad_output);
    AT_CHECK(grad_output.dim() >= 2, "grad_output for integral image must have >=2 dimensions")

    const int h = grad_output.size(grad_output.dim() - 2) - 1;
    const int w = grad_output.size(grad_output.dim() - 1) - 1;
    int nChannels = 1;
    for (int i = 0; i < grad_output.dim() - 2; ++i) {
        nChannels *= grad_output.size(i);
    }

    // Remove extra row and column
    std::vector<int64_t> outputSize(grad_output.sizes().begin(), grad_output.sizes().end());
    --outputSize[grad_output.dim() - 2];
    --outputSize[grad_output.dim() - 1];

    auto gradInput = at::ones(outputSize, grad_output.options());

    if (grad_output.is_cuda()) {
        THError("NYI: GPU integral_image_backward");
    } else {
        AT_DISPATCH_ALL_TYPES(grad_output.type(), "integral_image_backward_cpu", ([&] {
            using accscalar_t = at::acc_type<scalar_t, false>;

            scalar_t *gradOutputPtr = grad_output.data<scalar_t>();
            scalar_t *gradInputPtr = gradInput.data<scalar_t>();
            
            for (int c = 0; c < nChannels; ++c) {
                // Fill the last row
                accscalar_t sum = 0;
                for (int col = w-1; col >= 0; --col) {
                    sum += gradOutputPtr[h*(w+1) + (col+1)];
                    gradInputPtr[(h-1)*w + col] = sum;
                }

                // Fill the rest
                for (int row = h-2; row >= 0; --row) {
                    sum = 0;
                    for (int col = w-1; col >= 0; --col) {
                        sum += gradOutputPtr[(row+1)*(w+1) + (col+1)];
                        gradInputPtr[row*w + col] = sum + gradInputPtr[(row+1)*w + col];
                    }
                }

                gradOutputPtr += (h+1)*(w+1);
                gradInputPtr += h*w;
            }
        }));
    }

    return gradInput;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image_forward", &integral_image_forward, "Integral image, forward");
    m.def("integral_image_backward", &integral_image_backward, "Integral image, backward");
}
