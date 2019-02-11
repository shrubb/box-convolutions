#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

#define BLOCK_SIZE 256

using std::min;
using std::max;

#include "box_convolution.h" // for `enum class Parameter`

namespace gpu {

// TODO use constant memory when possible
// namespace constant {
//     __constant__ float xMinFrac[1536], xMaxFrac[1536];
//     __constant__ float yMinFrac[1536], yMaxFrac[1536];
//     __constant__ int xMinInt[1536], xMaxInt[1536];
//     __constant__ int yMinInt[1536], yMaxInt[1536];
//     __constant__ float area[1536];
// }

template <typename T, size_t N>
using CudaAcsr = const at::PackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;

// overload for "truncated"/"rounded" mode
template <bool normalize, typename scalar_t>
__global__ void boxConvUpdateOutputKernel(
    CudaAcsr<scalar_t,3> inputInt, CudaAcsr<scalar_t,3> output  ,
    CudaAcsr<int32_t ,1> xMinInt , CudaAcsr<int32_t ,1> xMaxInt ,
    CudaAcsr<int32_t ,1> yMinInt , CudaAcsr<int32_t ,1> yMaxInt ,
    CudaAcsr<scalar_t,1> area) {

    const int32_t y = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t x = blockDim.y * blockIdx.y + threadIdx.y;
    const int32_t inPlaneIdx = blockIdx.z / xMinInt.size(1);
    const int32_t paramIdx = blockIdx.z % (xMinInt.size(0) * xMinInt.size(1));
    const int32_t h = output.size(1);
    const int32_t w = output.size(2);

    if (x < h and y < w) {
        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        const int32_t t = max(0, min(x+xMinInt[paramIdx], h));
        const int32_t b = max(0, min(x+xMaxInt[paramIdx], h));
        const int32_t l = max(0, min(y+yMinInt[paramIdx], w));
        const int32_t r = max(0, min(y+yMaxInt[paramIdx], w));

        scalar_t outValue = 0;

        outValue += inputInt[inPlaneIdx][b][r];
        outValue -= inputInt[inPlaneIdx][t][r];
        outValue -= inputInt[inPlaneIdx][b][l];
        outValue += inputInt[inPlaneIdx][t][l];

        // TODO error: expression must be a modifiable lvalue
        output.data()[(blockIdx.z * h + x) * w + y] =
            outValue * (normalize ? area[paramIdx] : static_cast<scalar_t>(1));
    }
}

// overload for "exact" mode
template <bool normalize, typename scalar_t>
__global__ void boxConvUpdateOutputKernel(
    CudaAcsr<scalar_t,3> inputInt, CudaAcsr<scalar_t,3> output  ,
    CudaAcsr<int32_t ,1> xMinInt , CudaAcsr<int32_t ,1> xMaxInt ,
    CudaAcsr<int32_t ,1> yMinInt , CudaAcsr<int32_t ,1> yMaxInt ,
    CudaAcsr<scalar_t,1> xMinFrac, CudaAcsr<scalar_t,1> xMaxFrac,
    CudaAcsr<scalar_t,1> yMinFrac, CudaAcsr<scalar_t,1> yMaxFrac,
    CudaAcsr<scalar_t,1> area) {

    const int32_t y = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t x = blockDim.y * blockIdx.y + threadIdx.y;
    const int32_t inPlaneIdx = blockIdx.z / xMinInt.size(1);
    const int32_t paramIdx = blockIdx.z % (xMinInt.size(0) * xMinInt.size(1));
    const int32_t h = output.size(1);
    const int32_t w = output.size(2);

    if (x < h and y < w) {
        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.
        const int xMinCurr = xMinInt[paramIdx];
        const int xMaxCurr = xMaxInt[paramIdx];
        const int yMinCurr = yMinInt[paramIdx];
        const int yMaxCurr = yMaxInt[paramIdx];

        const scalar_t xMinCurrFrac = xMinFrac[paramIdx];
        const scalar_t xMaxCurrFrac = xMaxFrac[paramIdx];
        const scalar_t yMinCurrFrac = yMinFrac[paramIdx];
        const scalar_t yMaxCurrFrac = yMaxFrac[paramIdx];

        const int32_t t = max(0, min(x+xMinCurr, h));
        const int32_t b = max(0, min(x+xMaxCurr, h));
        const int32_t l = max(0, min(y+yMinCurr, w));
        const int32_t r = max(0, min(y+yMaxCurr, w));

        const int32_t bAdv = max(0, min(x+xMaxCurr+1, h));
        const int32_t rAdv = max(0, min(y+yMaxCurr+1, w));
        const int32_t tAdv = max(0, min(x+xMinCurr-1, h));
        const int32_t lAdv = max(0, min(y+yMinCurr-1, w));

        scalar_t outValue;

        // -- main area
        outValue = 
              inputInt[inPlaneIdx][b][r]
            - inputInt[inPlaneIdx][t][r]
            - inputInt[inPlaneIdx][b][l]
            + inputInt[inPlaneIdx][t][l];

        // -- xMax border
        outValue +=
            ( inputInt[inPlaneIdx][bAdv][r]
            - inputInt[inPlaneIdx][b   ][r]
            - inputInt[inPlaneIdx][bAdv][l]
            + inputInt[inPlaneIdx][b   ][l]) * xMaxCurrFrac;

        // -- yMax border
        outValue +=
            ( inputInt[inPlaneIdx][b][rAdv]
            - inputInt[inPlaneIdx][b][r   ]
            - inputInt[inPlaneIdx][t][rAdv]
            + inputInt[inPlaneIdx][t][r   ]) * yMaxCurrFrac;

        // -- xMin border
        outValue +=
            ( inputInt[inPlaneIdx][t   ][r]
            - inputInt[inPlaneIdx][tAdv][r]
            - inputInt[inPlaneIdx][t   ][l]
            + inputInt[inPlaneIdx][tAdv][l]) * xMinCurrFrac;

        // -- yMin border
        outValue +=
            ( inputInt[inPlaneIdx][b][l   ]
            - inputInt[inPlaneIdx][b][lAdv]
            - inputInt[inPlaneIdx][t][l   ]
            + inputInt[inPlaneIdx][t][lAdv]) * yMinCurrFrac;

        // -- corner pixels
        // Note: before, I used plain `input` to access corner values
        // with lower memory access overhead. Moved to `input_integrated`
        // to get rid of an extra input to this function.
        if (not ((x+xMaxCurr >= h) | (y+yMaxCurr >= w) |
                 (x+xMaxCurr <  0) | (y+yMaxCurr <  0))) {
            outValue += 
                xMaxCurrFrac * yMaxCurrFrac *
                ( inputInt[inPlaneIdx][b+1][r+1]
                - inputInt[inPlaneIdx][b  ][r+1]
                - inputInt[inPlaneIdx][b+1][r  ]
                + inputInt[inPlaneIdx][b  ][r  ]);
        }

        if (not ((x+xMinCurr >  h) | (y+yMaxCurr >= w) |
                 (x+xMinCurr <= 0) | (y+yMaxCurr <  0))) {
            outValue +=
                xMinCurrFrac * yMaxCurrFrac *
                ( inputInt[inPlaneIdx][t  ][r+1]
                - inputInt[inPlaneIdx][t-1][r+1]
                - inputInt[inPlaneIdx][t  ][r  ]
                + inputInt[inPlaneIdx][t-1][r  ]);
        }

        if (not ((x+xMaxCurr >= h) | (y+yMinCurr >  w) |
                 (x+xMaxCurr <  0) | (y+yMinCurr <= 0))) {
            outValue +=
                xMaxCurrFrac * yMinCurrFrac *
                ( inputInt[inPlaneIdx][b+1][l  ]
                - inputInt[inPlaneIdx][b  ][l  ]
                - inputInt[inPlaneIdx][b+1][l-1]
                + inputInt[inPlaneIdx][b  ][l-1]);
        }

        if (not ((x+xMinCurr >  h) | (y+yMinCurr >  w) |
                 (x+xMinCurr <= 0) | (y+yMinCurr <= 0))) {
            outValue +=
                xMinCurrFrac * yMinCurrFrac *
                ( inputInt[inPlaneIdx][t  ][l  ]
                - inputInt[inPlaneIdx][t-1][l  ]
                - inputInt[inPlaneIdx][t  ][l-1]
                + inputInt[inPlaneIdx][t-1][l-1]);
        }

        // TODO error: expression must be a modifiable lvalue
        output.data()[(blockIdx.z * h + x) * w + y] =
            outValue * (normalize ? area[paramIdx] : static_cast<scalar_t>(1));
    }
}

// TODO put split params and area into constant memory
template <bool normalize, bool exact>
void boxConvUpdateOutput(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & area, at::Tensor & input_integrated, at::Tensor & output) {

    const int h = output.size(-2);
    const int w = output.size(-1);
    const int totalOutputChannels = output.numel() / (h * w);

    const dim3 blockSize(32, 32, 1);
    const dim3 gridSize(
        (w + blockSize.x - 1) / blockSize.x,
        (h + blockSize.y - 1) / blockSize.y,
        (totalOutputChannels  + blockSize.z - 1) / blockSize.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "gpu::boxConvUpdateOutput", ([&] {

        auto xMinIntFlat = xMinInt.view(-1);
        auto xMaxIntFlat = xMaxInt.view(-1);
        auto yMinIntFlat = yMinInt.view(-1);
        auto yMaxIntFlat = yMaxInt.view(-1);
        auto xMinIntAcsr = xMinIntFlat.packed_accessor<int32_t, 1, at::RestrictPtrTraits, int32_t>();
        auto xMaxIntAcsr = xMaxIntFlat.packed_accessor<int32_t, 1, at::RestrictPtrTraits, int32_t>();
        auto yMinIntAcsr = yMinIntFlat.packed_accessor<int32_t, 1, at::RestrictPtrTraits, int32_t>();
        auto yMaxIntAcsr = yMaxIntFlat.packed_accessor<int32_t, 1, at::RestrictPtrTraits, int32_t>();

        auto xMinFracFlat = xMinFrac.view(-1);
        auto xMaxFracFlat = xMaxFrac.view(-1);
        auto yMinFracFlat = yMinFrac.view(-1);
        auto yMaxFracFlat = yMaxFrac.view(-1);
        auto xMinFracAcsr = xMinFracFlat.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, int32_t>();
        auto xMaxFracAcsr = xMaxFracFlat.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, int32_t>();
        auto yMinFracAcsr = yMinFracFlat.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, int32_t>();
        auto yMaxFracAcsr = yMaxFracFlat.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, int32_t>();
        
        auto areaAcsr = xMinFracAcsr; // because there's no default ctor :(
        // only initialize the accessor if `area` is defined (errors otherwise)
        if (normalize) {
            auto areaFlat = area.view(-1);
            areaAcsr = areaFlat.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, int32_t>();
        }

        auto inputIntFlattened = input_integrated.view({-1, h+1, w+1});
        auto inputIntAcsr =
            inputIntFlattened.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
            
        auto outputFlattened = output.view({-1, h, w});
        auto outputAcsr = 
            outputFlattened.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();

        if (exact) {
            boxConvUpdateOutputKernel <normalize>
                <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputIntAcsr, outputAcsr,
                xMinIntAcsr , xMaxIntAcsr , yMinIntAcsr , yMaxIntAcsr,
                xMinFracAcsr, xMaxFracAcsr, yMinFracAcsr, yMaxFracAcsr,
                areaAcsr);
        } else {
            boxConvUpdateOutputKernel <normalize>
                <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputIntAcsr, outputAcsr,
                xMinIntAcsr, xMaxIntAcsr, yMinIntAcsr, yMaxIntAcsr,
                areaAcsr);
        }
        THCudaCheck(cudaGetLastError());
    }));
}

// explicitly instantiate
template void boxConvUpdateOutput<true, true>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);

template void boxConvUpdateOutput<false, true>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);

template void boxConvUpdateOutput<true, false>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);

template void boxConvUpdateOutput<false, false>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);

}
