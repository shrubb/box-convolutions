#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <TH/THGeneral.h>

// #include <box_convolution_cuda.h>

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor box_convolution_forward(
    at::Tensor input_integrated,
    at::Tensor x_min, at::Tensor x_max,
    at::Tensor y_min, at::Tensor y_max,
    at::Tensor input) {

    CHECK_CONTIGUOUS(input_integrated);
    AT_CHECK(input_integrated.dim() == 4, "box conv input must have 4 dimensions");
    AT_CHECK(
        x_min.dim() == 2 and x_max.dim() == 2 and y_min.dim() == 2 and y_max.dim() == 2, 
        "all box conv parameters must have 2 dimensions");
    AT_CHECK(
        x_min.size(0) == x_max.size(0) and x_min.size(0) == y_min.size(0) and 
        x_min.size(0) == y_max.size(0) and x_min.size(0) == input_integrated.size(1), 
        "all box conv parameters must have as many rows as there are input channels");
    AT_CHECK(
        x_min.size(1) == x_max.size(1) and x_min.size(1) == y_min.size(1) and 
        x_min.size(1) == y_max.size(1),
        "all box conv parameters must have equal number of columns");

    // Split x_min, x_max, y_min, y_max into integer and fractional parts
    auto intOptions = x_min.options().is_variable(false).dtype(caffe2::TypeMeta::Make<int>());
    auto xMinInt = at::empty(x_min.sizes(), intOptions);
    auto xMaxInt = at::empty(x_min.sizes(), intOptions);
    auto yMinInt = at::empty(x_min.sizes(), intOptions);
    auto yMaxInt = at::empty(x_min.sizes(), intOptions);

    auto fracOptions = x_min.options().is_variable(false);
    auto xMinFrac = at::empty(x_min.sizes(), fracOptions);
    auto xMaxFrac = at::empty(x_min.sizes(), fracOptions);
    auto yMinFrac = at::empty(x_min.sizes(), fracOptions);
    auto yMaxFrac = at::empty(x_min.sizes(), fracOptions);

    if (x_min.is_cuda()) {
        THError("NYI: GPU split_params");
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_min.type(), "split_parameters_cpu", ([&] {
            scalar_t minInt, maxInt;

            for (int i = 0; i < x_min.numel(); ++i) {
                minInt = std::ceil(x_min.data<scalar_t>()[i]);
                xMinFrac.data<scalar_t>()[i] = minInt - x_min.data<scalar_t>()[i];
                xMinInt.data<int>()[i] = static_cast<int>(minInt);

                minInt = std::ceil(y_min.data<scalar_t>()[i]);
                yMinFrac.data<scalar_t>()[i] = minInt - y_min.data<scalar_t>()[i];
                yMinInt.data<int>()[i] = static_cast<int>(minInt);

                maxInt = std::floor(x_max.data<scalar_t>()[i]);
                xMaxFrac.data<scalar_t>()[i] = x_max.data<scalar_t>()[i] - maxInt;
                xMaxInt.data<int>()[i] = static_cast<int>(maxInt) + 1;

                maxInt = std::floor(y_max.data<scalar_t>()[i]);
                yMaxFrac.data<scalar_t>()[i] = y_max.data<scalar_t>()[i] - maxInt;
                yMaxInt.data<int>()[i] = static_cast<int>(maxInt) + 1;
            }
        }));
    }

    const int batchSize = input_integrated.size(0);
    const int nInputPlanes = input_integrated.size(1);
    const int numFilters = x_min.size(1);
    const int h = input_integrated.size(2) - 1;
    const int w = input_integrated.size(3) - 1;

    // Output will be 1 pixel smaller and have `num_filters` channels per each input channel
    auto output = at::empty(
        {batchSize, nInputPlanes, numFilters, h, w}, input_integrated.options());
    CHECK_CONTIGUOUS(output);

    // Actually fill `output`
    if (input_integrated.is_cuda()) {
        THError("NYI: GPU box_convolution_forward");
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "box_convolution_forward_cpu", ([&] {
            using std::min;
            using std::max;

            auto xMinIntAcsr = xMinInt.accessor<int, 2>();
            auto xMaxIntAcsr = xMaxInt.accessor<int, 2>();
            auto yMinIntAcsr = yMinInt.accessor<int, 2>();
            auto yMaxIntAcsr = yMaxInt.accessor<int, 2>();

            auto xMinFracAcsr = xMinFrac.accessor<scalar_t, 2>();
            auto xMaxFracAcsr = xMaxFrac.accessor<scalar_t, 2>();
            auto yMinFracAcsr = yMinFrac.accessor<scalar_t, 2>();
            auto yMaxFracAcsr = yMaxFrac.accessor<scalar_t, 2>();

            scalar_t *outputData = output.data<scalar_t>();
            
            for (int batchIdx = 0; batchIdx < input_integrated.size(0); ++batchIdx) {
                for (int inPlaneIdx = 0; inPlaneIdx < input_integrated.size(1); ++inPlaneIdx) {
                    auto inputIntPlane = input_integrated[batchIdx][inPlaneIdx];
                    auto inputIntAcsr = inputIntPlane.accessor<scalar_t, 2>();

                    for (int filterIdx = 0; filterIdx < x_min.size(1); ++filterIdx) {
                        
                        for (int x = 0; x < h; ++x) {
                            for (int y = 0; y < w; ++y) {
                                const int xMinCurr = xMinIntAcsr[inPlaneIdx][filterIdx];
                                const int xMaxCurr = xMaxIntAcsr[inPlaneIdx][filterIdx];
                                const int yMinCurr = yMinIntAcsr[inPlaneIdx][filterIdx];
                                const int yMaxCurr = yMaxIntAcsr[inPlaneIdx][filterIdx];

                                const scalar_t xMinCurrFrac = xMinFracAcsr[inPlaneIdx][filterIdx];
                                const scalar_t xMaxCurrFrac = xMaxFracAcsr[inPlaneIdx][filterIdx];
                                const scalar_t yMinCurrFrac = yMinFracAcsr[inPlaneIdx][filterIdx];
                                const scalar_t yMaxCurrFrac = yMaxFracAcsr[inPlaneIdx][filterIdx];

                                // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
                                // `integral()` behavior. Namely, I(x,0) and I(0,y) are
                                // always 0 (so it's a C-style array sum).

                                // However, when computing sums, we subtract values at points 
                                // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
                                // and yMin, and thus finally they are not affected.

                                const int t = max(0, min(x+xMinCurr, h));
                                const int b = max(0, min(x+xMaxCurr, h));
                                const int l = max(0, min(y+yMinCurr, w));
                                const int r = max(0, min(y+yMaxCurr, w));

                                const int bAdv = max(0, min(x+xMaxCurr+1, h));
                                const int rAdv = max(0, min(y+yMaxCurr+1, w));
                                const int tAdv = max(0, min(x+xMinCurr-1, h));
                                const int lAdv = max(0, min(y+yMinCurr-1, w));

                                scalar_t outValue;

                                // -- main area
                                outValue = 
                                      inputIntAcsr[b][r]
                                    - inputIntAcsr[t][r]
                                    - inputIntAcsr[b][l]
                                    + inputIntAcsr[t][l];

                                // -- xMax border
                                outValue +=
                                    ( inputIntAcsr[bAdv][r]
                                    - inputIntAcsr[b   ][r]
                                    - inputIntAcsr[bAdv][l]
                                    + inputIntAcsr[b   ][l]) * xMaxCurrFrac;

                                // -- yMax border
                                outValue +=
                                    ( inputIntAcsr[b][rAdv]
                                    - inputIntAcsr[b][r   ]
                                    - inputIntAcsr[t][rAdv]
                                    + inputIntAcsr[t][r   ]) * yMaxCurrFrac;

                                // -- xMin border
                                outValue +=
                                    ( inputIntAcsr[t   ][r]
                                    - inputIntAcsr[tAdv][r]
                                    - inputIntAcsr[t   ][l]
                                    + inputIntAcsr[tAdv][l]) * xMinCurrFrac;

                                // -- yMin border
                                outValue +=
                                    ( inputIntAcsr[b][l   ]
                                    - inputIntAcsr[b][lAdv]
                                    - inputIntAcsr[t][l   ]
                                    + inputIntAcsr[t][lAdv]) * yMinCurrFrac;

                                // -- corner pixels
                                // Note: before, I used plain `input` to access corner values
                                // with lower memory overhead. Moved to `input_integrated`
                                // to get rid of an extra input.

                                if (not ((x+xMaxCurr >= h) | (y+yMaxCurr >= w) |
                                         (x+xMaxCurr <  0) | (y+yMaxCurr <  0))) {
                                    outValue += 
                                        xMaxCurrFrac * yMaxCurrFrac *
                                        ( inputIntAcsr[b+1][r+1]
                                        - inputIntAcsr[b  ][r+1]
                                        - inputIntAcsr[b+1][r  ]
                                        + inputIntAcsr[b  ][r  ]);
                                }

                                if (not ((x+xMinCurr >  h) | (y+yMaxCurr >= w) |
                                         (x+xMinCurr <= 0) | (y+yMaxCurr <  0))) {
                                    outValue +=
                                        xMinCurrFrac * yMaxCurrFrac *
                                        ( inputIntAcsr[t  ][r+1]
                                        - inputIntAcsr[t-1][r+1]
                                        - inputIntAcsr[t  ][r  ]
                                        + inputIntAcsr[t-1][r  ]);
                                }

                                if (not ((x+xMaxCurr >= h) | (y+yMinCurr >  w) |
                                         (x+xMaxCurr <  0) | (y+yMinCurr <= 0))) {
                                    outValue +=
                                        xMaxCurrFrac * yMinCurrFrac *
                                        ( inputIntAcsr[b+1][l  ]
                                        - inputIntAcsr[b  ][l  ]
                                        - inputIntAcsr[b+1][l-1]
                                        + inputIntAcsr[b  ][l-1]);
                                }

                                if (not ((x+xMinCurr >  h) | (y+yMinCurr >  w) |
                                         (x+xMinCurr <= 0) | (y+yMinCurr <= 0))) {
                                    outValue +=
                                        xMinCurrFrac * yMinCurrFrac *
                                        ( inputIntAcsr[t  ][l  ]
                                        - inputIntAcsr[t-1][l  ]
                                        - inputIntAcsr[t  ][l-1]
                                        + inputIntAcsr[t-1][l-1]);
                                }

                                *(outputData++) = outValue;
                            }
                        }
                    } // filterIdx
                } // inPlaneIdx
            } // batchIdx
        }));
    }

    return output.reshape({batchSize, nInputPlanes * numFilters, h, w});
}

at::Tensor box_convolution_backward(
    at::Tensor input_integrated,
    at::Tensor x_min, at::Tensor x_max,
    at::Tensor y_min, at::Tensor y_max,
    at::Tensor grad_output) {

    CHECK_CONTIGUOUS(grad_output);
    AT_CHECK(grad_output.dim() >= 2, "grad_output for integral image must have >=2 dimensions")

    if (grad_output.is_cuda()) {
        THError("NYI: GPU box_convolution_backward");
    } else {
        // AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_min.type(), "box_convolution_backward_cpu", ([&] {
        //     ;
        // }));
    }

    // return gradInput;
}
