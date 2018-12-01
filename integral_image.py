import torch

class IntegralImageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad):
        input = ctx.saved_variables[0]
        return 2 * input * grad
