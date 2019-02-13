#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

#define BLOCK_SIZE 256

using std::min;
using std::max;

#include "box_convolution.h" // for `enum class Parameter`

namespace gpu {

// TODO make sure xMin and yMin threads don't fall into same warp
template <typename scalar_t>
__global__ void splitParametersKernel(
    const scalar_t * __restrict__ xMin, const scalar_t * __restrict__ xMax,
    const scalar_t * __restrict__ yMin, const scalar_t * __restrict__ yMax,
    int32_t * __restrict__ xMinInt, int32_t * __restrict__ xMaxInt,
    int32_t * __restrict__ yMinInt, int32_t * __restrict__ yMaxInt,
    scalar_t * __restrict__ xMinFrac, scalar_t * __restrict__ xMaxFrac,
    scalar_t * __restrict__ yMinFrac, scalar_t * __restrict__ yMaxFrac,
    const int nParameters) {

    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx < 2 * nParameters) {
        const int paramIndex = idx < nParameters ? idx : idx - nParameters;

        const scalar_t *param;
        scalar_t *fracParam;
        int32_t *intParam;

        param     = idx < nParameters ? xMin     : yMin;
        fracParam = idx < nParameters ? xMinFrac : yMinFrac;
        intParam  = idx < nParameters ? xMinInt  : yMinInt;

        const scalar_t minInt = std::ceil(param[paramIndex]);
        fracParam[paramIndex] = minInt - param[paramIndex];
        intParam[paramIndex] = static_cast<int32_t>(minInt);

        param     = idx < nParameters ? xMax     : yMax;
        fracParam = idx < nParameters ? xMaxFrac : yMaxFrac;
        intParam  = idx < nParameters ? xMaxInt  : yMaxInt;

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

    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx < 2 * nParameters) {
        const int paramIndex = idx < nParameters ? idx : idx - nParameters;

        const scalar_t *param;
        scalar_t *fracParam;
        int32_t *intParam;

        param     = idx < nParameters ? xMax     : yMax; // note: min/max swapped
        fracParam = idx < nParameters ? xMinFrac : yMinFrac;
        intParam  = idx < nParameters ? xMinInt  : yMinInt;

        const scalar_t minInt = std::ceil(-param[paramIndex]);
        fracParam[paramIndex] = minInt + param[paramIndex];
        intParam[paramIndex] = static_cast<int32_t>(minInt);

        param     = idx < nParameters ? xMin     : yMin; // note: min/max swapped
        fracParam = idx < nParameters ? xMaxFrac : yMaxFrac;
        intParam  = idx < nParameters ? xMaxInt  : yMaxInt;

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

    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx < 2 * nParameters) {
        const int paramIndex = idx < nParameters ? idx : idx - nParameters;

        const scalar_t *param;
        scalar_t *fracParam;
        int32_t *intParam;

        param     = idx < nParameters ? xMin     : yMin;
        fracParam = idx < nParameters ? xMinFrac : yMinFrac;
        intParam  = idx < nParameters ? xMinInt  : yMinInt;

        const scalar_t minInt = std::ceil(param[paramIndex] - 1);
        fracParam[paramIndex] = minInt - param[paramIndex] + 1;
        intParam[paramIndex] = static_cast<int32_t>(minInt);

        param     = idx < nParameters ? xMax     : yMax;
        fracParam = idx < nParameters ? xMaxFrac : yMaxFrac;
        intParam  = idx < nParameters ? xMaxInt  : yMaxInt;

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

template <typename scalar_t>
__global__ void clipParametersKernel(
    scalar_t * __restrict__ paramMin, scalar_t * __restrict__ paramMax,
    const double inverseReparam, const double minSize, const double maxSize, const int nElements) {

    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (idx < nElements) {
        double minValue, maxValue;
        const double paramMinCurrent = static_cast<double>(paramMin[idx]);
        const double paramMaxCurrent = static_cast<double>(paramMax[idx]);

        // clamp parameters
        minValue = max(-(maxSize+1) * inverseReparam,
            min((maxSize-1) * inverseReparam, paramMinCurrent));
        maxValue = max(-(maxSize+1) * inverseReparam,
            min((maxSize-1) * inverseReparam, paramMaxCurrent));

        // make sure bottom/right border doesn't come before top/left
        if (minValue + (minSize - 0.9999) * inverseReparam > maxValue) {
            const scalar_t mean = 0.5 * (minValue + maxValue);
            minValue = mean - 0.5 * (minSize - 0.9999) * inverseReparam;
            maxValue = mean + 0.5 * (minSize - 0.9999) * inverseReparam;
        }

        paramMin[idx] = static_cast<scalar_t>(minValue);
        paramMax[idx] = static_cast<scalar_t>(maxValue);
    }
}

void clipParameters(
    at::Tensor & paramMin, at::Tensor & paramMax,
    const double reparametrization, const double minSize, const double maxSize) {

    const int threadsNeeded = paramMin.numel();
    const int numBlocks = (threadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const double inverseReparam = 1.0 / reparametrization;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(paramMin.type(), "gpu::clipParameters", ([&] {
        clipParametersKernel
        <<<numBlocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>> (
            paramMin.data<scalar_t>(), paramMax.data<scalar_t>(),
            inverseReparam, minSize, maxSize, paramMin.numel());
        THCudaCheck(cudaGetLastError());
    }));
}

template <bool needXDeriv, bool needYDeriv, typename scalar_t>
__global__ void computeAreaKernel(
    scalar_t * __restrict__ x_min, scalar_t * __restrict__ x_max,
    scalar_t * __restrict__ y_min, scalar_t * __restrict__ y_max,
    scalar_t * __restrict__ retval, const int nElements) {

    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (idx < nElements) {
        const scalar_t area = 
            (needXDeriv ? x_max[idx]-x_min[idx]+1 : static_cast<scalar_t>(1)) *
            (needYDeriv ? y_max[idx]-y_min[idx]+1 : static_cast<scalar_t>(1));
        retval[idx] = 1 / area;
    }
}

at::Tensor computeArea(
    at::Tensor x_min, at::Tensor x_max, at::Tensor y_min, at::Tensor y_max,
    const bool exact, const bool needXDeriv, const bool needYDeriv) {

    // TODO: how to stop tracking operations??? `.is_variable_(false)` doesn't work
    auto retval = at::empty_like(x_min);

    if (not exact) {
        x_min = x_min.ceil();
        y_min = y_min.ceil();
        x_max = x_max.floor();
        y_max = y_max.floor();
    }

    const int threadsNeeded = x_min.numel();
    const int numBlocks = (threadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_min.type(), "gpu::computeArea", ([&] {
        if (needXDeriv) {
            if (needYDeriv) {
                computeAreaKernel <true, true>
                <<<numBlocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>> (
                    x_min.data<scalar_t>(), x_max.data<scalar_t>(),
                    y_min.data<scalar_t>(), y_max.data<scalar_t>(),
                    retval.data<scalar_t>(), x_min.numel());
            } else {
                computeAreaKernel <true, false>
                <<<numBlocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>> (
                    x_min.data<scalar_t>(), x_max.data<scalar_t>(),
                    y_min.data<scalar_t>(), y_max.data<scalar_t>(),
                    retval.data<scalar_t>(), x_min.numel());
            }
        } else {
            if (needYDeriv) {
                computeAreaKernel <false, true>
                <<<numBlocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>> (
                    x_min.data<scalar_t>(), x_max.data<scalar_t>(),
                    y_min.data<scalar_t>(), y_max.data<scalar_t>(),
                    retval.data<scalar_t>(), x_min.numel());
            } else {
                THError("computeArea called with needXDeriv == needYDeriv == false");
            }
        }
    }));
    THCudaCheck(cudaGetLastError());

    return retval;
}

}
