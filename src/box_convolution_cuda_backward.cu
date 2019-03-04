#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

#include "box_convolution.h" // for `enum class Parameter`

#define BLOCK_SIZE 256
#define NUM_THREADS 1024

using std::min;
using std::max;

namespace gpu {

template <typename T, size_t N>
using CudaAcsr = const at::PackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;

// TODO switch to square blocks
template <bool normalize, bool exact, typename scalar_t>
__global__ void boxConvUpdateGradInputKernel(
    CudaAcsr<scalar_t,3> gradOutputInt, scalar_t * __restrict__ tmpArray,
    const int32_t * __restrict__ xMinInt , const int32_t * __restrict__ xMaxInt ,
    const int32_t * __restrict__ yMinInt , const int32_t * __restrict__ yMaxInt ,
    const scalar_t * __restrict__ xMinFrac, const scalar_t * __restrict__ xMaxFrac,
    const scalar_t * __restrict__ yMinFrac, const scalar_t * __restrict__ yMaxFrac,
    const scalar_t * __restrict__ area, const int nParams) {
    
    int32_t id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id;

    const int32_t h = gradOutputInt.size(1) - 1;
    const int32_t w = gradOutputInt.size(2) - 1;
    const int32_t y = id % w; id /= w;
    const int32_t x = id % h; id /= h;
    const int32_t paramIdx = id % nParams;

    // `id` is now the current plane number
    auto gradOutputIntPlane = gradOutputInt[id];

    if (id < gradOutputInt.size(0)) {

        const int32_t xMinCurr = xMinInt[paramIdx];
        const int32_t xMaxCurr = xMaxInt[paramIdx];
        const int32_t yMinCurr = yMinInt[paramIdx];
        const int32_t yMaxCurr = yMaxInt[paramIdx];

        const int t = max(0, min(x+xMinCurr, h));
        const int b = max(0, min(x+xMaxCurr, h));
        const int l = max(0, min(y+yMinCurr, w));
        const int r = max(0, min(y+yMaxCurr, w));

        scalar_t outValue;

        outValue = 
              gradOutputIntPlane[b][r]
            - gradOutputIntPlane[t][r]
            - gradOutputIntPlane[b][l]
            + gradOutputIntPlane[t][l];

        if (exact) {
            const scalar_t xMinCurrFrac = xMinFrac[paramIdx];
            const scalar_t xMaxCurrFrac = xMaxFrac[paramIdx];
            const scalar_t yMinCurrFrac = yMinFrac[paramIdx];
            const scalar_t yMaxCurrFrac = yMaxFrac[paramIdx];

            const int tAdv = x+xMinCurr-1 <  h ? max(0, min(t-1, h)) : t;
            const int bAdv = x+xMaxCurr   >= 0 ? max(0, min(b+1, h)) : b;
            const int lAdv = y+yMinCurr-1 <  w ? max(0, min(l-1, w)) : l;
            const int rAdv = y+yMaxCurr   >= 0 ? max(0, min(r+1, w)) : r;

            // -- xMax border
            outValue +=
                ( gradOutputIntPlane[bAdv][r]
                - gradOutputIntPlane[b   ][r]
                - gradOutputIntPlane[bAdv][l]
                + gradOutputIntPlane[b   ][l]
                ) * xMaxCurrFrac;

            // -- yMax border
            outValue +=
                ( gradOutputIntPlane[b][rAdv]
                - gradOutputIntPlane[b][r   ]
                - gradOutputIntPlane[t][rAdv]
                + gradOutputIntPlane[t][r   ]
                ) * yMaxCurrFrac;

            // -- xMin border
            outValue +=
                ( gradOutputIntPlane[t   ][r]
                - gradOutputIntPlane[tAdv][r]
                - gradOutputIntPlane[t   ][l]
                + gradOutputIntPlane[tAdv][l]
                ) * xMinCurrFrac;

            // -- yMin border
            outValue +=
                ( gradOutputIntPlane[b][l   ]
                - gradOutputIntPlane[b][lAdv]
                - gradOutputIntPlane[t][l   ]
                + gradOutputIntPlane[t][lAdv]
                ) * yMinCurrFrac;

            // -- corner pixels
            outValue += 
                xMaxCurrFrac*yMaxCurrFrac * (
                   (x+xMaxCurr >= h or
                    y+yMaxCurr >= w or
                    x+xMaxCurr <  0 or
                    y+yMaxCurr <  0 or
                    b == bAdv or
                    r == rAdv) ? static_cast<scalar_t>(0) : 
                    
                    ( gradOutputIntPlane[b+1][r+1]
                    - gradOutputIntPlane[b  ][r+1]
                    - gradOutputIntPlane[b+1][r  ]
                    + gradOutputIntPlane[b  ][r  ]));

            outValue +=
                xMinCurrFrac*yMaxCurrFrac * (
                   (x+xMinCurr >  h or
                    y+yMaxCurr >= w or
                    x+xMinCurr <= 0 or
                    y+yMaxCurr <  0 or
                    t == tAdv or
                    r == rAdv) ? static_cast<scalar_t>(0) : 
                    
                    ( gradOutputIntPlane[tAdv+1][r+1]
                    - gradOutputIntPlane[tAdv+1][r  ]
                    - gradOutputIntPlane[tAdv  ][r+1]
                    + gradOutputIntPlane[tAdv  ][r  ]));

            outValue +=
                xMaxCurrFrac*yMinCurrFrac * (
                   (x+xMaxCurr >= h or
                    y+yMinCurr >  w or
                    x+xMaxCurr <  0 or
                    y+yMinCurr <= 0 or
                    b == bAdv or
                    l == lAdv) ? static_cast<scalar_t>(0) : 
                    
                    ( gradOutputIntPlane[b+1][lAdv+1]
                    - gradOutputIntPlane[b  ][lAdv+1]
                    - gradOutputIntPlane[b+1][lAdv  ]
                    + gradOutputIntPlane[b  ][lAdv  ]));

            outValue +=
                xMinCurrFrac*yMinCurrFrac * (
                   (x+xMinCurr >  h or
                    y+yMinCurr >  w or
                    x+xMinCurr <= 0 or
                    y+yMinCurr <= 0 or
                    t == tAdv or
                    l == lAdv) ? static_cast<scalar_t>(0) : 
                    
                    ( gradOutputIntPlane[tAdv+1][lAdv+1]
                    - gradOutputIntPlane[tAdv+1][lAdv  ]
                    - gradOutputIntPlane[tAdv  ][lAdv+1]
                    + gradOutputIntPlane[tAdv  ][lAdv  ]));
        }

        *tmpArray = outValue * (normalize ? area[paramIdx] : static_cast<scalar_t>(1));
    }
}

template <bool normalize, bool exact>
void boxConvUpdateGradInput(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & area, at::Tensor & grad_output_integrated, at::Tensor & tmpArray) {

    // TODO use square blocks as in `boxConvUpdateOutput`?
    const int threadsNeeded = tmpArray.numel();
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tmpArray.type(), "gpu::boxConvUpdateGradInput", ([&] {
        auto gradOutputIntFlattened = grad_output_integrated.view(
            {-1, grad_output_integrated.size(-2), grad_output_integrated.size(-1)});
        auto gradOutputIntAcsr =
            gradOutputIntFlattened.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();

        boxConvUpdateGradInputKernel <normalize, exact>
            <<<numBlocks, NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>> (
            gradOutputIntAcsr, tmpArray.data<scalar_t>(),
            xMinInt.data<int32_t>(), xMaxInt.data<int32_t>(),
            yMinInt.data<int32_t>(), yMaxInt.data<int32_t>(),
            xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
            yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(),
            normalize ? area.data<scalar_t>() : nullptr, xMinInt.numel());
        THCudaCheck(cudaGetLastError());
    }));
}

// explicitly instantiate
template void boxConvUpdateGradInput<true, true>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);

template void boxConvUpdateGradInput<false, true>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);

template void boxConvUpdateGradInput<true, false>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);

template void boxConvUpdateGradInput<false, false>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &);


// TODO overload for exact/truncated mode
// TODO accept only three pairs of parameter arrays, not four (one is always redundant)
template <Parameter parameter, bool exact, typename scalar_t>
__global__ void boxConvAccGradParametersKernel(
    CudaAcsr<scalar_t,3> inputInt, scalar_t * __restrict__ tmpArray,
    const int32_t * __restrict__ xMinInt , const int32_t * __restrict__ xMaxInt ,
    const int32_t * __restrict__ yMinInt , const int32_t * __restrict__ yMaxInt ,
    const scalar_t * __restrict__ xMinFrac, const scalar_t * __restrict__ xMaxFrac,
    const scalar_t * __restrict__ yMinFrac, const scalar_t * __restrict__ yMaxFrac,
    const int nParams) {
    
    int32_t id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id;

    const int32_t h = inputInt.size(1) - 1;
    const int32_t w = inputInt.size(2) - 1;
    const int32_t y = id % w + 1; id /= w;
    const int32_t x = id % h + 1; id /= h;
    const int32_t paramIdx = id % nParams; id /= nParams;

    // `id` is now the current absolute input plane number
    auto inputIntPlane = inputInt[id];

    if (id < inputInt.size(0)) {

        const int32_t xMinCurr = xMinInt[paramIdx];
        const int32_t xMaxCurr = xMaxInt[paramIdx];
        const int32_t yMinCurr = yMinInt[paramIdx];
        const int32_t yMaxCurr = yMaxInt[paramIdx];

        // TODO only define these if `exact == true`
        const scalar_t xMinCurrFrac = xMinFrac[paramIdx];
        const scalar_t xMaxCurrFrac = xMaxFrac[paramIdx];
        const scalar_t yMinCurrFrac = yMinFrac[paramIdx];
        const scalar_t yMaxCurrFrac = yMaxFrac[paramIdx];

        int valid;
        int cornerX, cornerY;
        
        scalar_t delta = 0;

        if (parameter == Parameter::xMin) {
            if (exact) {
            // TODO maybe use `input` instead of `inputInt`
            valid =
                not (y+yMinCurr < 1) & not (y+yMinCurr > w) & not (x+xMinCurr < 1);
            cornerX = max(0,min(h-1,x+xMinCurr-1));
            cornerY = max(0,min(w-1,y+yMinCurr-1));
            const scalar_t tlCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            valid = 
                not (y+yMaxCurr  < 0) & not (y+yMaxCurr  >= w) & not (x+xMinCurr  < 1);
            cornerX = max(0,min(h-1,x+xMinCurr -1));
            cornerY = max(0,min(w-1,y+yMaxCurr   ));
            const scalar_t trCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            delta += trCorner * yMaxCurrFrac;
            delta += tlCorner * yMinCurrFrac;
            } // if (exact)

            delta += inputIntPlane
                [max(0,min(x+xMinCurr   , h))][max(0,min(y+yMaxCurr   , w))];
            delta -= inputIntPlane
                [max(0,min(x+xMinCurr -1, h))][max(0,min(y+yMaxCurr   , w))];
            delta -= inputIntPlane
                [max(0,min(x+xMinCurr   , h))][max(0,min(y+yMinCurr   , w))];
            delta += inputIntPlane
                [max(0,min(x+xMinCurr -1, h))][max(0,min(y+yMinCurr   , w))];

            delta *= (x+xMinCurr  >= 1) & (x+xMinCurr  <= h);

            *tmpArray = -delta;
        }

        else if (parameter == Parameter::xMax) {
            if (exact) {
            valid =
                not (y+yMinCurr  < 1) & not (y+yMinCurr  > w) & not (x+xMaxCurr  >= h);
            cornerX = max(0,min(h-1,x+xMaxCurr   ));
            cornerY = max(0,min(w-1,y+yMinCurr -1));
            const scalar_t blCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            valid = 
                not (y+yMaxCurr  < 0) & not (y+yMaxCurr  >= w) & not (x+xMaxCurr  >= h);
            cornerX = max(0,min(h-1,x+xMaxCurr   ));
            cornerY = max(0,min(w-1,y+yMaxCurr   ));
            const scalar_t brCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            delta += brCorner * yMaxCurrFrac;
            delta += blCorner * yMinCurrFrac;
            } // if (exact)

            delta += inputIntPlane
                [max(0,min(x+xMaxCurr +1, h))][max(0,min(y+yMaxCurr   , w))];
            delta -= inputIntPlane
                [max(0,min(x+xMaxCurr   , h))][max(0,min(y+yMaxCurr   , w))];
            delta -= inputIntPlane
                [max(0,min(x+xMaxCurr +1, h))][max(0,min(y+yMinCurr   , w))];
            delta += inputIntPlane
                [max(0,min(x+xMaxCurr   , h))][max(0,min(y+yMinCurr   , w))];

            delta *= (x+xMaxCurr  >= 0) & (x+xMaxCurr  < h);

            *tmpArray = delta;
        }

        else if (parameter == Parameter::yMin) {
            if (exact) {
            valid =
                not (y+yMinCurr  < 1) & not (x+xMinCurr  < 1) & not (x+xMinCurr  > h);
            cornerX = max(0,min(h-1,x+xMinCurr -1));
            cornerY = max(0,min(w-1,y+yMinCurr -1));
            const scalar_t tlCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            valid = 
                not (y+yMinCurr  < 1) & not (x+xMaxCurr  < 0) & not (x+xMaxCurr  >= h);
            cornerX = max(0,min(h-1,x+xMaxCurr   ));
            cornerY = max(0,min(w-1,y+yMinCurr -1));
            const scalar_t blCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            delta += tlCorner * xMinCurrFrac;
            delta += blCorner * xMaxCurrFrac;
            } // if (exact)

            delta += inputIntPlane
                [max(0,min(x+xMaxCurr   , h))][max(0,min(y+yMinCurr   , w))];
            delta -= inputIntPlane
                [max(0,min(x+xMaxCurr   , h))][max(0,min(y+yMinCurr -1, w))];
            delta -= inputIntPlane
                [max(0,min(x+xMinCurr   , h))][max(0,min(y+yMinCurr   , w))];
            delta += inputIntPlane
                [max(0,min(x+xMinCurr   , h))][max(0,min(y+yMinCurr -1, w))];

            delta *= (y+yMinCurr  >= 1) & (y+yMinCurr  <= w);

            *tmpArray = -delta;
        }

        else if (parameter == Parameter::yMax) {
            if (exact) {
            valid =
                not (y+yMaxCurr  >= w) & not (x+xMinCurr  < 1) & not (x+xMinCurr  > h);
            cornerX = max(0,min(h-1,x+xMinCurr -1));
            cornerY = max(0,min(w-1,y+yMaxCurr   ));
            const scalar_t trCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            valid = 
                not (y+yMaxCurr  >= w) & not (x+xMaxCurr  < 0) & not (x+xMaxCurr  >= h);
            cornerX = max(0,min(h-1,x+xMaxCurr   ));
            cornerY = max(0,min(w-1,y+yMaxCurr   ));
            const scalar_t brCorner = valid * 
                ( inputIntPlane[cornerX+1][cornerY+1]
                - inputIntPlane[cornerX  ][cornerY+1]
                - inputIntPlane[cornerX+1][cornerY  ]
                + inputIntPlane[cornerX  ][cornerY  ]);
            
            delta += trCorner * xMinCurrFrac;
            delta += brCorner * xMaxCurrFrac;
            } // if (exact)

            delta += inputIntPlane
                [max(0,min(x+xMaxCurr   , h))][max(0,min(y+yMaxCurr +1, w))];
            delta -= inputIntPlane
                [max(0,min(x+xMaxCurr   , h))][max(0,min(y+yMaxCurr   , w))];
            delta -= inputIntPlane
                [max(0,min(x+xMinCurr   , h))][max(0,min(y+yMaxCurr +1, w))];
            delta += inputIntPlane
                [max(0,min(x+xMinCurr   , h))][max(0,min(y+yMaxCurr   , w))];

            delta *= (y+yMaxCurr  >= 0) & (y+yMaxCurr  < w);

            *tmpArray = delta;
        }
    }
}

template <bool exact>
void boxConvAccGradParameters(
    // tmpArray size: {batchSize, nInputPlanes, numFilters, h, w}
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & input_integrated, at::Tensor & tmpArray, Parameter parameter) {

    // TODO switch to square blocks?
    const int threadsNeeded = tmpArray.numel();
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tmpArray.type(), "gpu::boxConvAccGradParameters", ([&] {
        auto inputIntFlattened = input_integrated.view(
            {-1, input_integrated.size(-2), input_integrated.size(-1)});
        auto inputIntAcsr =
            inputIntFlattened.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();

        switch (parameter) {
        case Parameter::xMin:
            boxConvAccGradParametersKernel <Parameter::xMin, exact>
                <<<numBlocks, NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputIntAcsr, tmpArray.data<scalar_t>(),
                xMinInt.data<int32_t>(),  xMaxInt.data<int32_t>(),
                yMinInt.data<int32_t>(),  yMaxInt.data<int32_t>(),
                xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
                yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(), xMinInt.numel()); break;
        case Parameter::xMax:
            boxConvAccGradParametersKernel <Parameter::xMax, exact>
                <<<numBlocks, NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputIntAcsr, tmpArray.data<scalar_t>(),
                xMinInt.data<int32_t>(),  xMaxInt.data<int32_t>(),
                yMinInt.data<int32_t>(),  yMaxInt.data<int32_t>(),
                xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
                yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(), xMinInt.numel()); break;
        case Parameter::yMin:
            boxConvAccGradParametersKernel <Parameter::yMin, exact>
                <<<numBlocks, NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputIntAcsr, tmpArray.data<scalar_t>(),
                xMinInt.data<int32_t>(),  xMaxInt.data<int32_t>(),
                yMinInt.data<int32_t>(),  yMaxInt.data<int32_t>(),
                xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
                yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(), xMinInt.numel()); break;
        case Parameter::yMax:
            boxConvAccGradParametersKernel <Parameter::yMax, exact>
                <<<numBlocks, NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputIntAcsr, tmpArray.data<scalar_t>(),
                xMinInt.data<int32_t>(),  xMaxInt.data<int32_t>(),
                yMinInt.data<int32_t>(),  yMaxInt.data<int32_t>(),
                xMinFrac.data<scalar_t>(), xMaxFrac.data<scalar_t>(),
                yMinFrac.data<scalar_t>(), yMaxFrac.data<scalar_t>(), xMinInt.numel()); break;
        }
        THCudaCheck(cudaGetLastError());
    }));
}

// explicitly instantiate
template void boxConvAccGradParameters<true>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, Parameter);

template void boxConvAccGradParameters<false>(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, Parameter);

}
