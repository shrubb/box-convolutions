#include <torch/extension.h>
#include <ATen/AccumulateType.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

#include "integral_image.h"

#include <type_traits>

#define BLOCK_SIZE 256

namespace gpu {

template <typename scalar_t, typename accscalar_t>
__global__ void accumulateColsKernel(
    const scalar_t * __restrict__ input, scalar_t * __restrict__ output,
    const int channels, const int h, const int w);

template <typename scalar_t, typename accscalar_t>
__global__ void accumulateColsInplaceTransposedKernel(
    scalar_t * __restrict__ input, const int channels, const int h, const int w);

// contiguous out-of-place transpose
template<typename scalar_t>
void transpose(at::Tensor & input, at::Tensor & output) {

    AT_CHECK(input.dim() == 2);
    AT_CHECK(input.numel() == output.numel());

    if (std::is_same<scalar_t, float>()) {
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cudaStream_t currentStream = at::cuda::getCurrentCUDAStream();
        cublasSetStream(cublasHandle, currentStream);
        const float ONE = 1.0, ZERO = 0.0;

        THCublasCheck(cublasSgeam(
            cublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N, input.size(0), input.size(1),
            &ONE, input.data<float>(), input.size(1),
            &ZERO, output.data<float>(), input.size(0),
            output.data<float>(), input.size(0)));

    } else if (std::is_same<scalar_t, double>()) {
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cudaStream_t currentStream = at::cuda::getCurrentCUDAStream();
        cublasSetStream(cublasHandle, currentStream);
        const double ONE = 1.0, ZERO = 0.0;

        THCublasCheck(cublasDgeam(
            cublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N, input.size(0), input.size(1),
            &ONE, input.data<double>(), input.size(1),
            &ZERO, output.data<double>(), input.size(0),
            output.data<double>(), input.size(0)));
        
    } else {
        // TODO improve
        output.view({input.size(1), input.size(0)}).copy_(input.t());

    }
}

void integral_image(at::Tensor & input, at::Tensor & output) {

    const int h = input.size(-2);
    const int w = input.size(-1);
    const int channels = input.numel() / (h * w);

    auto inputView = input.view({channels, h, w});
    auto outputView = output.view({channels, h+1, w+1});
    auto tmpBuffer = at::empty_like(output);

    cudaStream_t currentStream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "integral_image_forward_gpu", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        // input : (channels) x (h) x (w), contiguous
        // output: (channels) x (h+1) x (w+1), contiguous
        // tmpBuffer   : at least (channels) * (h+1) * (w+1)
        int blockSize1D, gridSize1D;

        // Compute prefix sums of columns, `input` -> `output`
        // (channels) x (h) x (w) ==> (channels) x (h+1) x (w+1)
        // Note: output[:,:,0] remains uninitialized
        const int totalCols = channels * w;
        blockSize1D = BLOCK_SIZE;
        gridSize1D = (totalCols + blockSize1D - 1) / blockSize1D;
        accumulateColsKernel <scalar_t, accscalar_t>
            <<<gridSize1D, blockSize1D, 0, currentStream>>>
            (inputView.data<scalar_t>(), outputView.data<scalar_t>(), channels, h, w);
        THCudaCheck(cudaGetLastError());

        // transpose, `output` -> `tmpBuffer`
        // (channels) x (h+1) x (w+1) ==> (w+1) x (channels) x (h+1)
        auto output2Dim = output.view({channels * (h+1), w+1});
        transpose<scalar_t>(output2Dim, tmpBuffer);

        // Compute prefix sums of columns (former rows), `tmpBuffer` -> `tmpBuffer`
        // (w+1) x (channels) x (h+1) ==> (w+1) x (channels) x (h+1)
        const int totalRows = channels * h; // actually, number of cols in (w+1) x (channels * (h+1)) image
        blockSize1D = BLOCK_SIZE;
        gridSize1D = (totalRows + blockSize1D - 1) / blockSize1D;
        accumulateColsInplaceTransposedKernel <scalar_t, accscalar_t>
            <<<gridSize1D, blockSize1D, 0, currentStream>>>
            (tmpBuffer.data<scalar_t>(), channels, h, w);
        THCudaCheck(cudaGetLastError());

        // transpose, `tmpBuffer` -> `output`
        // (w+1) x (channels) x (h+1) ==> (channels) x (h+1) x (w+1)
        tmpBuffer = tmpBuffer.reshape({w+1, channels * (h+1)});
        transpose<scalar_t>(tmpBuffer, output);
    })); // AT_DISPATCH_FLOATING_TYPES_AND_HALF
}

template <typename scalar_t, typename accscalar_t>
__global__ void accumulateColsKernel(
    const scalar_t * __restrict__ input, scalar_t * __restrict__ output,
    const int channels, const int h, const int w) {
    // input : (channels * h) x (w)
    // output: (channels * (h+1)) x (w+1) -- first column remains untouched

    // global column index (of total `channels * w` columns in this image):
    const int globalColIdx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (globalColIdx < channels * w) {
        const int channelIdx = globalColIdx / w;
        const int colIdx = globalColIdx - channelIdx * w;
        
        // jump to the channel of interest:
        int inputPos = channelIdx * h * w + colIdx;
        // (let local columns be 1-indexed: 0-th output column is always zero)
        int outputPos = channelIdx * (h+1) * (w+1) + colIdx + 1;

        output[outputPos] = 0; // 0-th element of every column is always zero
        accscalar_t sum = 0;
        for (int i = 1; i <= h; ++i) {
            sum += static_cast<accscalar_t>(input[inputPos + (i-1) * w]);
            output[outputPos + i * (w+1)] = static_cast<scalar_t>(sum);
        }
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void accumulateColsInplaceTransposedKernel(
    scalar_t * __restrict__ input, const int channels, const int h, const int w) {
    // in-place.
    // input: (w+1) x (channels * (h+1))

    // global column index (of total `channels * w` columns in this image):
    const int globalColIdx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (globalColIdx < channels * h) {
        const int channelIdx = globalColIdx / h;
        // add `channelIdx + 1` to account for one extra column in each horizontally stacked image
        const int colIdx = globalColIdx + channelIdx + 1;

        // need to zero the (0,0) corner of the output separately >:(
        input[channelIdx * (h+1)] = 0;

        input[colIdx] = 0; // first element of every column is always zero
        accscalar_t sum = 0;
        for (int i = 1; i <= w; ++i) {
            scalar_t *currentElement = &input[i * channels * (h+1) + colIdx];
            sum += static_cast<accscalar_t>(*currentElement);
            *currentElement = static_cast<scalar_t>(sum);
        }
    }
}

} // namespace gpu