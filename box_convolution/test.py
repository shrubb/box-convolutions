import time
import random
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def test_integral_image():
    def integral_image_reference(input):
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

    from .integral_image import IntegralImageFunction

    # check IntegralImageFunction vs reference implementation
    for test_idx in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        stride_h, stride_w = 1, 1 # may change in the future
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        grad_output = (torch.rand(batch_size, in_planes, h+1, w+1) < 0.1).to(input_image.dtype)

        reference_result = integral_image_reference(input_image)
        reference_result.backward(grad_output)
        reference_grad = input_image.grad.clone()
        input_image.grad.zero_()

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

def test_box_convolution_module():
    def explicit_box_kernel(x_min, x_max, y_min, y_max, normalize=False):
        import math
        h_farthest = math.ceil(max(x_max, -x_min))
        w_farthest = math.ceil(max(y_max, -y_min))

        retval = torch.ones(1+2*h_farthest, 1+2*w_farthest)

        def segments_intersection(a_l, a_r, b_l, b_r):
            common_l = max(a_l, b_l)
            common_r = min(a_r, b_r)
            return max(0.0, common_r - common_l)
            
        for x, row in enumerate(retval, start=-h_farthest):
            # h-extent of the current row: [x; x+1]
            # h-extent of the box of interest: [x_min; x_max+1]
            # find length of their intersection and multiply the row by it
            row *= segments_intersection(x, x+1, x_min, x_max+1)

        for y, col in enumerate(retval.t(), start=-w_farthest):
            # same for columns
            col *= segments_intersection(y, y+1, y_min, y_max+1)

        if normalize:
            area = (y_max-y_min+1) * (x_max-x_min+1)
            retval *= 1/area

        return retval

    # reference implementation
    def box_convolution_reference(input, x_min, x_max, y_min, y_max, normalize=False):
        assert x_min.shape == x_max.shape
        assert x_min.shape == y_min.shape
        assert x_min.shape == y_max.shape

        assert input.ndimension() == 4

        in_planes, num_filters = x_min.shape
        assert input.shape[1] == in_planes

        kernels = [[explicit_box_kernel(*out_c, normalize) for out_c in zip(*in_c)] \
            for in_c in zip(x_min, x_max, y_min, y_max)]
        assert len(kernels) == in_planes
        assert all(len(x) == num_filters for x in kernels)

        def conv2d_single_channel(image, kernel):
            image = image.view((1,1) + image.shape)
            kernel = kernel.view((1,1) + kernel.shape)
            padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
            return torch.conv2d(image, kernel, padding=padding)[0,0]

        output_shape = list(input.shape)
        output_shape.insert(2, num_filters)
        output = torch.empty(output_shape, dtype=input.dtype)

        for in_sample, out_sample in zip(input, output):
            for in_plane_idx, in_plane_kernels in enumerate(kernels):
                for filter_idx, kernel in enumerate(in_plane_kernels):
                    filtered = conv2d_single_channel(in_sample[in_plane_idx], kernel)
                    out_sample[in_plane_idx, filter_idx].copy_(filtered)

        retval_shape = list(input.shape)
        retval_shape[1] *= num_filters
        return output.reshape(retval_shape)

    from . import BoxConv2d

    # same interface for our target function
    def box_convolution_wrapper(input, x_min, x_max, y_min, y_max, normalize=False):
        assert x_min.shape == x_max.shape
        assert x_min.shape == y_min.shape
        assert x_min.shape == y_max.shape

        assert input.ndimension() == 4

        in_planes, num_filters = x_min.shape
        assert input.shape[1] == in_planes

        module = BoxConv2d(in_planes, num_filters, -1, -1)
        module.x_min.copy_(x_min)
        module.x_max.copy_(x_max)
        module.y_min.copy_(y_min)
        module.y_max.copy_(y_max)

        return module(input)

    for test_idx in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        num_filters = random.randint(1, 3)
        stride_h, stride_w = 1, 1 # may change in the future
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)
        # exact = random.random() < 0.8

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        
        x_min, x_max, y_min, y_max = (torch.empty(in_planes, num_filters) for _ in range(4))
        for plane_idx in range(in_planes):
            for window_idx in range(num_filters):
                x_min[plane_idx, window_idx] = random.uniform(-h+2.5, h-1.5)
                y_min[plane_idx, window_idx] = random.uniform(-w+2.5, w-1.5)
                x_max[plane_idx, window_idx] = random.uniform(x_min[plane_idx, window_idx], h-1.5)
                y_max[plane_idx, window_idx] = random.uniform(y_min[plane_idx, window_idx], w-1.5)

        grad_output = \
            (torch.rand(batch_size, in_planes*num_filters, h, w) < 0.1).to(input_image.dtype)

        # check output and grad w.r.t. input vs reference ones
        reference_result = box_convolution_reference(input_image, x_min, x_max, y_min, y_max)
        reference_result.backward(grad_output)
        reference_grad_input = input_image.grad.clone()
        input_image.grad.zero_()

        our_result = box_convolution_wrapper(input_image, x_min, x_max, y_min, y_max)
        our_result.backward(grad_output)
        our_grad_input = input_image.grad.clone()

        if not our_result.allclose(reference_result):
            raise ValueError(
                'Test %d failed at forward pass.\n\nInput:\n%s\n\n'
                'Our output:\n%s\n\nReference output:\n%s\n\n'
                    % (test_idx, input_image, our_result, reference_result))

        if not our_grad_input.allclose(reference_grad_input):
            raise ValueError(
                'Test %d failed at backward pass.\n\nInput:\n%s\n\nOutput:\n%s\n\n'
                'gradOutput:\n%s\n\nOur gradInput:\n%s\n\nReference gradInput:\n%s\n\n'
                    % (test_idx, input_image, our_result, grad_output, our_grad_input, \
                       reference_grad_input))

        # check our grads w.r.t. parameters against finite differences
        for tensor in x_min, x_max, y_min, y_max:
            tensor.requires_grad_()

        torch.autograd.gradcheck(
            box_convolution_wrapper, (input_image, x_min, x_max, y_min, y_max),
            eps=0.05, raise_exception=True)

if __name__ == '__main__':
    seed = int(time.time())
    random.seed(seed)
    print('Random seed is %d' % seed)

    for testing_function in test_integral_image, test_box_convolution_module:
        print('Running %s()...' % testing_function.__name__)
        # TODO [re]set random state etc.
        testing_function()
        print('OK')
