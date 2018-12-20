import torch

import box_convolution_cpp_cuda as cpp_cuda

class IntegralImageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return cpp_cuda.integral_image_forward(input)

    @staticmethod
    def backward(ctx, grad):
        return 2 * input * grad
