#include <torch/extension.h>

namespace cpu {

// Splits x_min, x_max, y_min, y_max into integer and fractional parts
void splitParameters(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac) {

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_min.type(), "splitParametersCPU", ([&] {
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

void boxConvUpdateOutput(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & input_integrated, at::Tensor & output) {

    const int h = output.size(-2);
    const int w = output.size(-1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "boxConvUpdateOutputCPU", ([&] {
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

                for (int filterIdx = 0; filterIdx < xMinInt.size(1); ++filterIdx) {
                    
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
                            // to get rid of an extra input to this function.

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

// `grad_output_integrated` size: {batchSize, nInputPlanes, numFilters, h+1, w+1}
// `tmpArray` size: {batchSize, nInputPlanes, numFilters, h, w}
void boxConvUpdateGradInput(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & grad_output_integrated, at::Tensor & tmpArray) {

    const int h = tmpArray.size(-2);
    const int w = tmpArray.size(-1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tmpArray.type(), "boxConvUpdateGradInputCPU", ([&] {
        // A special parameters' split for backward pass
        scalar_t minInt, maxInt;

        for (int i = 0; i < x_min.numel(); ++i) {
            minInt = std::ceil(-x_max.data<scalar_t>()[i]);
            xMinFrac.data<scalar_t>()[i] = minInt + x_max.data<scalar_t>()[i];
            xMinInt.data<int>()[i] = static_cast<int>(minInt);

            minInt = std::ceil(-y_max.data<scalar_t>()[i]);
            yMinFrac.data<scalar_t>()[i] = minInt + y_max.data<scalar_t>()[i];
            yMinInt.data<int>()[i] = static_cast<int>(minInt);

            maxInt = std::floor(-x_min.data<scalar_t>()[i]) + 1;
            xMaxFrac.data<scalar_t>()[i] = -x_min.data<scalar_t>()[i] + 1 - maxInt;
            xMaxInt.data<int>()[i] = static_cast<int>(maxInt);

            maxInt = std::floor(-y_min.data<scalar_t>()[i]) + 1;
            yMaxFrac.data<scalar_t>()[i] = -y_min.data<scalar_t>()[i] + 1 - maxInt;
            yMaxInt.data<int>()[i] = static_cast<int>(maxInt);
        }

        // Actually fill gradInput
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

        scalar_t *tmpArrayData = tmpArray.data<scalar_t>();

        for (int batchIdx = 0; batchIdx < grad_output_integrated.size(0); ++batchIdx) {
            for (int inPlaneIdx = 0; inPlaneIdx < grad_output_integrated.size(1); ++inPlaneIdx) {
                for (int filterIdx = 0; filterIdx < xMinInt.size(1); ++filterIdx) {

                    const int xMinCurr = xMinIntAcsr[inPlaneIdx][filterIdx];
                    const int xMaxCurr = xMaxIntAcsr[inPlaneIdx][filterIdx];
                    const int yMinCurr = yMinIntAcsr[inPlaneIdx][filterIdx];
                    const int yMaxCurr = yMaxIntAcsr[inPlaneIdx][filterIdx];

                    const scalar_t xMinCurrFrac = xMinFracAcsr[inPlaneIdx][filterIdx];
                    const scalar_t xMaxCurrFrac = xMaxFracAcsr[inPlaneIdx][filterIdx];
                    const scalar_t yMinCurrFrac = yMinFracAcsr[inPlaneIdx][filterIdx];
                    const scalar_t yMaxCurrFrac = yMaxFracAcsr[inPlaneIdx][filterIdx];
                    
                    auto gradOutputIntPlane = 
                        grad_output_integrated[batchIdx][inPlaneIdx][filterIdx];
                    auto gradOutputAcsr = gradOutputIntPlane.accessor<scalar_t, 2>();
                    
                    for (int x = 0; x < h; ++x) {
                        for (int y = 0; y < w; ++y) {

                            const int t = max(0, min(x+xMinCurr, h));
                            const int b = max(0, min(x+xMaxCurr, h));
                            const int l = max(0, min(y+yMinCurr, w));
                            const int r = max(0, min(y+yMaxCurr, w));

                            const int tAdv = x+xMinCurr-1 <  h ? max(0, min(t-1, h)) : t;
                            const int bAdv = x+xMaxCurr   >= 0 ? max(0, min(b+1, h)) : b;
                            const int lAdv = y+yMinCurr-1 <  w ? max(0, min(l-1, w)) : l;
                            const int rAdv = y+yMaxCurr   >= 0 ? max(0, min(r+1, w)) : r;

                            scalar_t outValue;

                            outValue = 
                                  gradOutputAcsr[b][r]
                                - gradOutputAcsr[t][r]
                                - gradOutputAcsr[b][l]
                                + gradOutputAcsr[t][l];

                            // -- xMax border
                            outValue +=
                                ( gradOutputAcsr[bAdv][r]
                                - gradOutputAcsr[b   ][r]
                                - gradOutputAcsr[bAdv][l]
                                + gradOutputAcsr[b   ][l]
                                ) * xMaxCurrFrac;

                            // -- yMax border
                            outValue +=
                                ( gradOutputAcsr[b][rAdv]
                                - gradOutputAcsr[b][r   ]
                                - gradOutputAcsr[t][rAdv]
                                + gradOutputAcsr[t][r   ]
                                ) * yMaxCurrFrac;

                            // -- xMin border
                            outValue +=
                                ( gradOutputAcsr[t   ][r]
                                - gradOutputAcsr[tAdv][r]
                                - gradOutputAcsr[t   ][l]
                                + gradOutputAcsr[tAdv][l]
                                ) * xMinCurrFrac;

                            // -- yMin border
                            outValue +=
                                ( gradOutputAcsr[b][l   ]
                                - gradOutputAcsr[b][lAdv]
                                - gradOutputAcsr[t][l   ]
                                + gradOutputAcsr[t][lAdv]
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
                                    
                                    ( gradOutputAcsr[b+1][r+1]
                                    - gradOutputAcsr[b  ][r+1]
                                    - gradOutputAcsr[b+1][r  ]
                                    + gradOutputAcsr[b  ][r  ]));

                            outValue +=
                                xMinCurrFrac*yMaxCurrFrac * (
                                   (x+xMinCurr >  h or
                                    y+yMaxCurr >= w or
                                    x+xMinCurr <= 0 or
                                    y+yMaxCurr <  0 or
                                    t == tAdv or
                                    r == rAdv) ? static_cast<scalar_t>(0) : 
                                    
                                    ( gradOutputAcsr[tAdv+1][r+1]
                                    - gradOutputAcsr[tAdv+1][r  ]
                                    - gradOutputAcsr[tAdv  ][r+1]
                                    + gradOutputAcsr[tAdv  ][r  ]));

                            outValue +=
                                xMaxCurrFrac*yMinCurrFrac * (
                                   (x+xMaxCurr >= h or
                                    y+yMinCurr >  w or
                                    x+xMaxCurr <  0 or
                                    y+yMinCurr <= 0 or
                                    b == bAdv or
                                    l == lAdv) ? static_cast<scalar_t>(0) : 
                                    
                                    ( gradOutputAcsr[b+1][lAdv+1]
                                    - gradOutputAcsr[b  ][lAdv+1]
                                    - gradOutputAcsr[b+1][lAdv  ]
                                    + gradOutputAcsr[b  ][lAdv  ]));

                            outValue +=
                                xMinCurrFrac*yMinCurrFrac * (
                                   (x+xMinCurr >  h or
                                    y+yMinCurr >  w or
                                    x+xMinCurr <= 0 or
                                    y+yMinCurr <= 0 or
                                    t == tAdv or
                                    l == lAdv) ? static_cast<scalar_t>(0) : 
                                    
                                    ( gradOutputAcsr[tAdv+1][lAdv+1]
                                    - gradOutputAcsr[tAdv+1][lAdv  ]
                                    - gradOutputAcsr[tAdv  ][lAdv+1]
                                    + gradOutputAcsr[tAdv  ][lAdv  ]));

                            *(tmpArrayData++) = outValue;
                        }
                    }
                } // filterIdx
            } // inPlaneIdx
        } // batchIdx
    }));
}

} // namespace cpu
