#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <TH/THGeneral.h>

// #include <integral_image_cuda.h>

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor integral_image(at::Tensor input) {
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
