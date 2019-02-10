#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

#define BLOCK_SIZE 256

#include "box_convolution.h" // for `enum class Parameter`

namespace gpu {

template <typename scalar_t>
__global__ void splitParametersKernel(
    const scalar_t * __restrict__ xMin, const scalar_t * __restrict__ xMax,
    const scalar_t * __restrict__ yMin, const scalar_t * __restrict__ yMax,
    int32_t * __restrict__ xMinInt, int32_t * __restrict__ xMaxInt,
    int32_t * __restrict__ yMinInt, int32_t * __restrict__ yMaxInt,
    scalar_t * __restrict__ xMinFrac, scalar_t * __restrict__ xMaxFrac,
    scalar_t * __restrict__ yMinFrac, scalar_t * __restrict__ yMaxFrac,
    const int nParameters) {

    const int id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (id < 2 * nParameters) {
        const int paramIndex = id < nParameters ? id : id - nParameters;

        const scalar_t *param;
        scalar_t *fracParam;
        int32_t *intParam;

        param     = id < nParameters ? xMin     : yMin;
        fracParam = id < nParameters ? xMinFrac : yMinFrac;
        intParam  = id < nParameters ? xMinInt  : yMinInt;

        const scalar_t minInt = std::ceil(param[paramIndex]);
        fracParam[paramIndex] = minInt - param[paramIndex];
        intParam[paramIndex] = static_cast<int32_t>(minInt);

        param     = id < nParameters ? xMax     : yMax;
        fracParam = id < nParameters ? xMaxFrac : yMaxFrac;
        intParam  = id < nParameters ? xMaxInt  : yMaxInt;

        const scalar_t maxInt = std::floor(param[paramIndex]);
        fracParam[paramIndex] = param[paramIndex] - maxInt;
        intParam[paramIndex] = static_cast<int32_t>(maxInt) + 1;
    }
}

void splitParameters(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac) {

    const int threadsNeeded = 2 * x_min.numel();
    const int numBlocks = (threadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_min.type(), "gpu::splitParameters", ([&] {
        splitParametersKernel
        <<<numBlocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>> (
            x_min.data<scalar_t>(), x_max.data<scalar_t>(),
            y_min.data<scalar_t>(), y_max.data<scalar_t>(),
            xMinInt.data<int32_t>(), xMaxInt.data<int32_t>(),
            yMinInt.data<int32_t>(), yMaxInt.data<int32_t>(),
            xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
            yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(),
            x_min.numel());
        THCudaCheck(cudaGetLastError());
    }));
}

template <typename scalar_t>
__global__ void splitParametersUpdateGradInputKernel(
    const scalar_t * __restrict__ xMin, const scalar_t * __restrict__ xMax,
    const scalar_t * __restrict__ yMin, const scalar_t * __restrict__ yMax,
    int32_t * __restrict__ xMinInt, int32_t * __restrict__ xMaxInt,
    int32_t * __restrict__ yMinInt, int32_t * __restrict__ yMaxInt,
    scalar_t * __restrict__ xMinFrac, scalar_t * __restrict__ xMaxFrac,
    scalar_t * __restrict__ yMinFrac, scalar_t * __restrict__ yMaxFrac,
    const int nParameters) {

    const int id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (id < 2 * nParameters) {
        const int paramIndex = id < nParameters ? id : id - nParameters;

        const scalar_t *param;
        scalar_t *fracParam;
        int32_t *intParam;

        param     = id < nParameters ? xMin     : yMin;
        fracParam = id < nParameters ? xMinFrac : yMinFrac;
        intParam  = id < nParameters ? xMinInt  : yMinInt;

        const scalar_t minInt = std::ceil(-param[paramIndex]);
        fracParam[paramIndex] = minInt + param[paramIndex];
        intParam[paramIndex] = static_cast<int32_t>(minInt);

        param     = id < nParameters ? xMax     : yMax;
        fracParam = id < nParameters ? xMaxFrac : yMaxFrac;
        intParam  = id < nParameters ? xMaxInt  : yMaxInt;

        const scalar_t maxInt = std::floor(-param[paramIndex]) + 1;
        fracParam[paramIndex] = -param[paramIndex] + 1 - maxInt;
        intParam[paramIndex] = static_cast<int32_t>(maxInt);
    }
}

void splitParametersUpdateGradInput(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac) {

    const int threadsNeeded = 2 * x_min.numel();
    const int numBlocks = (threadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_min.type(), "gpu::splitParametersUpdateGradInput", ([&] {
        splitParametersUpdateGradInputKernel 
        <<<numBlocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>> (
            x_min.data<scalar_t>(), x_max.data<scalar_t>(),
            y_min.data<scalar_t>(), y_max.data<scalar_t>(),
            xMinInt.data<int32_t>(), xMaxInt.data<int32_t>(),
            yMinInt.data<int32_t>(), yMaxInt.data<int32_t>(),
            xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
            yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(),
            x_min.numel());
        THCudaCheck(cudaGetLastError());
    }));
}

template <typename scalar_t>
__global__ void splitParametersAccGradParametersKernel(
    const scalar_t * __restrict__ xMin, const scalar_t * __restrict__ xMax,
    const scalar_t * __restrict__ yMin, const scalar_t * __restrict__ yMax,
    int32_t * __restrict__ xMinInt, int32_t * __restrict__ xMaxInt,
    int32_t * __restrict__ yMinInt, int32_t * __restrict__ yMaxInt,
    scalar_t * __restrict__ xMinFrac, scalar_t * __restrict__ xMaxFrac,
    scalar_t * __restrict__ yMinFrac, scalar_t * __restrict__ yMaxFrac,
    const int nParameters) {

    const int id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (id < 2 * nParameters) {
        const int paramIndex = id < nParameters ? id : id - nParameters;

        const scalar_t *param;
        scalar_t *fracParam;
        int32_t *intParam;

        param     = id < nParameters ? xMin     : yMin;
        fracParam = id < nParameters ? xMinFrac : yMinFrac;
        intParam  = id < nParameters ? xMinInt  : yMinInt;

        const scalar_t minInt = std::ceil(param[paramIndex] - 1);
        fracParam[paramIndex] = minInt - param[paramIndex] + 1;
        intParam[paramIndex] = static_cast<int32_t>(minInt);

        param     = id < nParameters ? xMax     : yMax;
        fracParam = id < nParameters ? xMaxFrac : yMaxFrac;
        intParam  = id < nParameters ? xMaxInt  : yMaxInt;

        const scalar_t maxInt = std::floor(param[paramIndex]);
        fracParam[paramIndex] = param[paramIndex] - maxInt;
        intParam[paramIndex] = static_cast<int32_t>(maxInt);
    }
}

void splitParametersAccGradParameters(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac) {

    const int threadsNeeded = 2 * x_min.numel();
    const int numBlocks = (threadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_min.type(), "gpu::splitParametersAccGradParams", ([&] {
        splitParametersAccGradParametersKernel
        <<<numBlocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>> (
            x_min.data<scalar_t>(), x_max.data<scalar_t>(),
            y_min.data<scalar_t>(), y_max.data<scalar_t>(),
            xMinInt.data<int32_t>(), xMaxInt.data<int32_t>(),
            yMinInt.data<int32_t>(), yMaxInt.data<int32_t>(),
            xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
            yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(),
            x_min.numel());
        THCudaCheck(cudaGetLastError());
    }));
}

}
