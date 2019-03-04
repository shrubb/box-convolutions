#include <torch/extension.h>
#include <ATen/AccumulateType.h>

#include "integral_image.h"

namespace cpu {

void integral_image(at::Tensor & input, at::Tensor & output) {

    const int h = input.size(-2);
    const int w = input.size(-1);
    const int nChannels = input.numel() / (h * w);
    
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

} // namespace cpu
