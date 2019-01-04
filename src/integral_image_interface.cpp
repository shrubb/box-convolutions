#include <torch/extension.h>
#include <TH/THGeneral.h>

#include <integral_image.h>

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor integral_image(at::Tensor input) {
    CHECK_CONTIGUOUS(input);
    AT_CHECK(input.dim() >= 2, "integral image input must have >=2 dimensions")

    // The result will have extra row and column for zeros
    // TODO maybe it's faster to eliminate them
    std::vector<int64_t> outputSize(input.sizes().begin(), input.sizes().end());
    ++outputSize[input.dim() - 2];
    ++outputSize[input.dim() - 1];

    auto output = at::empty(outputSize, input.options());

    if (input.is_cuda()) {
        THError("NYI: GPU integral_image_forward");
    } else {
        cpu::integral_image(input, output);
    }

    return output;
}
