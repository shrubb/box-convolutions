import time
import random
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def test_integral_image():
    from .integral_image import IntegralImageFunction

    # reference implementation
    def integral_image_python(input):
        assert input.ndimension() >= 2
        h, w = input.shape[-2:]
        output_shape = input.shape[:-2] + (h+1, w+1)
        output = torch.empty(output_shape, dtype=input.dtype)

        # zero the 0th columns and rows
        output.select(-2, 0).fill_(0)
        output.select(-1, 0).fill_(0)

        # accumulate rows
        output_no_zero_col = output.narrow(-1, 1, w)
        sum_rows = torch.zeros_like(input.select(-2, 0), dtype=torch.float64)
        for row_idx in range(h):
            sum_rows += input.select(-2, row_idx).double()
            output_no_zero_col.select(-2, row_idx+1).copy_(sum_rows)

        # accumulate columns
        sum_cols = torch.zeros_like(output.select(-1, 0), dtype=torch.float64)
        for col_idx in range(w):
            sum_cols += output.select(-1, col_idx+1).double()
            output.select(-1, col_idx+1).copy_(sum_cols)

        return output

    # check IntegralImageFunction vs reference implementation
    for test_idx in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        stride_h, stride_w = 1, 1
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        grad_output = (torch.rand(batch_size, in_planes, h+1, w+1) < 0.1).to(input_image.dtype)

        reference_result = integral_image_python(input_image)
        reference_result.backward(grad_output)
        reference_grad = input_image.grad.clone()

        our_result = IntegralImageFunction.apply(input_image)
        our_result.backward(grad_output)
        our_grad = input_image.grad.clone()

        if not our_result.allclose(reference_result):
            raise ValueError(
                'Test %d failed at forward pass.\n\nInput:\n%s\n\n'
                'Our output:\n%s\n\nReference output:\n%s\n\n'
                    % (test_idx, input_image, our_result, reference_result))

        if not our_grad.allclose(reference_grad):
            raise ValueError(
                'Test %d failed at backward pass.\n\nInput:\n%s\n\nOutput:\n%s\n\n'
                'gradOutput:\n%s\n\nOur gradInput:\n%s\n\nReference gradInput:\n%s\n\n'
                    % (test_idx, input_image, our_result, grad_output, our_grad, reference_grad))

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
    seed = int(time.time())
    random.seed(seed)
    print('Random seed is %d' % seed)

    for testing_function in \
        test_integral_image, test_box_convolution_function, test_box_convolution_module:

        print('Running %s()...' % testing_function.__name__)
        # TODO [re]set random state etc.
        testing_function()
        print('OK')
