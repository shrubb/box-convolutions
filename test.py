import random
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def test_integral_image():
    # TODO add ground truth tests
    from .integral_image import IntegralImageFunction

    for i in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        stride_h, stride_w = 1, 1
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)

        torch.autograd.gradcheck(
            IntegralImageFunction.apply, input_image, eps=0.01, raise_exception=True)

def test_box_convolution_function():
    # TODO add ground truth tests
    from .box_convolution_function import BoxConvolutionFunction

    for i in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        num_filters = random.randint(1, 3)
        stride_h, stride_w = 1, 1
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        
        x_min, x_max, y_min, y_max = (torch.empty(in_planes, num_filters) for _ in range(4))
        for plane_idx in range(in_planes):
            for window_idx in range(num_filters):
                x_min[plane_idx, window_idx] = random.uniform(-h+1, h-2)
                y_min[plane_idx, window_idx] = random.uniform(-w+1, w-2)
                x_max[plane_idx, window_idx] = random.uniform(x_min[plane_idx, window_idx]+1, h-1)
                y_max[plane_idx, window_idx] = random.uniform(y_min[plane_idx, window_idx]+1, w-1)

        torch.autograd.gradcheck(
            BoxConvolutionFunction.apply, (input_image, x_min, x_max, y_min, y_max),
            eps=0.01, raise_exception=True)

def test_box_convolution_module():
    # TODO add ground truth tests
    from . import BoxConv2d

    for i in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        num_filters = random.randint(1, 3)
        stride_h, stride_w = 1, 1
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        
        box_conv = BoxConv2d(in_planes, num_filters, h, w, stride_h, stride_w)

        def forward_wrapper(input_image, x_min, x_max, y_min, y_max):
            box_conv.x_min.copy_(x_min)
            box_conv.x_max.copy_(x_max)
            box_conv.y_min.copy_(y_min)
            box_conv.y_max.copy_(y_max)
            return box_conv.forward(input_image)

        box_conv.forward(input_image)
        torch.autograd.gradcheck(
            forward_wrapper,
            (input_image, box_conv.x_min, box_conv.x_max, box_conv.y_min, box_conv.y_max),
            eps=0.001, raise_exception=True)

if __name__ == '__main__':
    for testing_function in \
        test_integral_image, test_box_convolution_function, test_box_convolution_module:

        print('Running %s()...' % testing_function.__name__)
        # TODO [re]set random state etc.
        testing_function()
        print('OK')
