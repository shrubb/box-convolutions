import torch

import box_convolution_cpp_cuda as cpp_cuda

# TODO: rename `x_` and `y_` to `h_` and `w_`
class BoxConvolutionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_integrated, x_min, x_max, y_min, y_max, input):
        ctx.save_for_backward(
            input_integrated, x_min, x_max, y_min, y_max, input)
        return cpp_cuda.box_convolution_forward(
            input_integrated, x_min, x_max, y_min, y_max, input)

    @staticmethod
    def backward(ctx, grad_output):
        input_integrated, x_min, x_max, y_min, y_max, input = ctx.saved_variables
        # return cpp_cuda.box_convolution_backward(
        return (input_integrated, x_min, x_max, y_min, y_max) + (input * 0,)
