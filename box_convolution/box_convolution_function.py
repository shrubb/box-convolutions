import torch

# TODO: rename `x_` and `y_` to `h_` and `w_`
class BoxConvolutionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, x_min, x_max, y_min, y_max):
        ctx.save_for_backward(input, x_min, x_max, y_min, y_max)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, x_min, x_max, y_min, y_max = ctx.saved_variables
        return 2 * input * grad_output, x_min*0, x_max*0, y_min*0, y_max*0
