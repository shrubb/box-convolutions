import time
import random
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def test_integral_image():
    # TODO use torch.cumsum
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

    from box_convolution_cpp_cuda import integral_image

    # check IntegralImageFunction vs reference implementation
    for test_idx in tqdm(range(50)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        stride_h, stride_w = 1, 1 # may change in the future
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        grad_output = (torch.rand(batch_size, in_planes, h+1, w+1) < 0.15).to(input_image.dtype)

        reference_result = integral_image_reference(input_image)
        our_result = integral_image(input_image)

        if not our_result.allclose(reference_result):
            raise ValueError(
                'Test %d failed at forward pass.\n\nInput:\n%s\n\n'
                'Our output:\n%s\n\nReference output:\n%s\n\n'
                    % (test_idx, input_image, our_result, reference_result))

def test_box_convolution_module():
    def explicit_box_kernel(x_min, x_max, y_min, y_max, normalize):
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
    def box_convolution_reference(
        input, x_min, x_max, y_min, y_max,
        reparametrization_h, reparametrization_w, normalize, exact):

        assert x_min.shape == x_max.shape
        assert x_min.shape == y_min.shape
        assert x_min.shape == y_max.shape

        assert input.ndimension() == 4
        assert type(normalize) is bool

        x_min, x_max, y_min, y_max = \
            reparametrize(
                x_min, x_max, y_min, y_max,
                reparametrization_h, reparametrization_w, inverse=True)

        if not exact:
            x_min.ceil_()
            y_min.ceil_()
            x_max.floor_()
            y_max.floor_()

        in_planes, num_filters = x_min.shape
        assert input.shape[1] == in_planes

        # in_c, out_c = input channel, output channel
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
    from .box_convolution_function import reparametrize

    # same interface for our target function
    def box_convolution_wrapper(
        input, x_min, x_max, y_min, y_max,
        max_input_h, max_input_w, reparametrization_factor, normalize, exact):

        assert x_min.shape == x_max.shape
        assert x_min.shape == y_min.shape
        assert x_min.shape == y_max.shape

        assert input.ndimension() == 4
        assert type(normalize) is bool

        in_planes, num_filters = x_min.shape
        assert input.shape[1] == in_planes

        module = BoxConv2d(
            in_planes, num_filters, max_input_h, max_input_w,
            reparametrization_factor).type(input.dtype)

        del module.x_min; module.x_min = x_min
        del module.x_max; module.x_max = x_max
        del module.y_min; module.y_min = y_min
        del module.y_max; module.y_max = y_max
        module.normalize = normalize
        module.exact = exact

        params_before = module.get_actual_parameters()
        
        output = module(input)
        
        params_after = module.get_actual_parameters()
        param_names = 'x_min', 'x_max', 'y_min', 'y_max'
        for p_before, p_after, p_name in zip(params_before, params_after, param_names):
            if not torch.equal(p_before, p_after):
                raise ValueError(
                    'Wrong test case configuration: `_clip_parameters` '
                    'has changed one of the parameters.\n\n' + \
                    'h, w = %d, %d\n\n' % (h, w) + \
                    'Before:\n' + \
                    '\n'.join('%s: %s' % (n,p) for n,p in zip(param_names, params_before)) + \
                    '\n\nAfter:\n' + \
                    '\n'.join('%s: %s' % (n,p) for n,p in zip(param_names, params_after)))

        return output

    for test_idx in tqdm(range(40)):
        batch_size = random.randint(1, 1)
        in_planes = random.randint(1, 1)
        num_filters = random.randint(1, 1)
        stride_h, stride_w = 1, 1 # may change in the future
        exact = random.random() < 0.7

        # if not exact, minimum box size changes from 1 to 2
        h = random.randint(1 + stride_h + (not exact), 10)
        w = random.randint(1 + stride_w + (not exact), 10)
        max_input_h, max_input_w = h+1, w+1
        reparametrization_factor = random.random() * 4.5 + 0.5
        reparametrization_h = max_input_h * reparametrization_factor
        reparametrization_w = max_input_w * reparametrization_factor
        gradcheck_step = 0.004

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        
        # sample boxes more or less randomly (algorithm isn't practical but is OK for gradcheck)
        x_min, x_max, y_min, y_max = (torch.empty(in_planes, num_filters) for _ in range(4))
        for plane_idx in range(in_planes):
            for filter_idx in range(num_filters):
                
                box_is_valid = False
                while not box_is_valid:
                    x_min_curr = random.uniform(-h+1.05, h-(not exact)-1.1)
                    y_min_curr = random.uniform(-w+1.05, w-(not exact)-1.1)
                    
                    # set sizes to at least 1.001 because of `_clip_parameters`'s behavior
                    x_max_curr = random.uniform(
                        x_min_curr + (not exact) + 3*gradcheck_step + 0.001, h-1.05)
                    y_max_curr = random.uniform(
                        y_min_curr + (not exact) + 3*gradcheck_step + 0.001, w-1.05)

                    # As a function of box coordinates (x_min etc.), box convolution isn't smooth
                    # at integer points, so the finite difference gradcheck will fail.
                    # Therefore, let's resample the box until all coordinates are far 
                    # enough from integers.
                    box_is_valid = True
                    for value in x_min_curr, y_min_curr, x_max_curr, y_max_curr:
                        if abs(value - round(value)) <= gradcheck_step * 3: # *3 for extra safety
                            box_is_valid = False

                x_min[plane_idx, filter_idx] = x_min_curr
                y_min[plane_idx, filter_idx] = y_min_curr
                x_max[plane_idx, filter_idx] = x_max_curr
                y_max[plane_idx, filter_idx] = y_max_curr

        # reparametrize
        x_min, x_max, y_min, y_max = \
            reparametrize(x_min, x_max, y_min, y_max, reparametrization_h, reparametrization_w)

        # randomly test either sum filter or average filter
        normalize = random.choice((False, True))

        grad_output = \
            (torch.rand(batch_size, in_planes*num_filters, h, w) < 0.15).to(input_image.dtype)

        # check output and grad w.r.t. input vs reference ones
        reference_result = box_convolution_reference(
            input_image, x_min, x_max, y_min, y_max,
            reparametrization_h, reparametrization_w, normalize, exact)
        reference_result.backward(grad_output)
        reference_grad_input = input_image.grad.clone()
        input_image.grad.zero_()

        our_result = box_convolution_wrapper(
            input_image, x_min, x_max, y_min, y_max,
            max_input_h, max_input_w, reparametrization_factor, normalize, exact)
        our_result.backward(grad_output)
        our_grad_input = input_image.grad.clone()
        
        if not our_result.allclose(reference_result, rtol=3e-5, atol=1e-5):
            raise ValueError(
                'Test %d failed at forward pass.\n\nNormalize: %s\n\nInput:\n%s\n\n'
                'Our output:\n%s\n\nReference output:\n%s\n\nMax diff: %f\n\n'
                    % (test_idx, normalize, input_image, our_result, reference_result, \
                       (our_result - reference_result).abs().max()))

        if not our_grad_input.allclose(reference_grad_input, rtol=3e-5, atol=1e-5):
            raise ValueError(
                'Test %d failed at backward pass.\n\nNormalize: %s\n\n'
                'Input:\n%s\n\nOutput:\n%s\n\ngradOutput:\n%s\n\nOur gradInput:\n%s\n\n'
                'Reference gradInput:\n%s\n\nMax diff: %f\n\n'
                    % (test_idx, normalize, input_image, our_result, \
                       grad_output, our_grad_input, reference_grad_input, \
                       (our_grad_input-reference_grad_input).abs().max()))

        # sorry, I don't want to reliably check gradients w.r.t. parameters in rounded mode
        if not exact:
            continue

        # convert to double and check our grads w.r.t. parameters against finite differences
        with torch.no_grad():
            input_image = input_image.double()
            x_min = x_min.double()
            x_max = x_max.double()
            y_min = y_min.double()
            y_max = y_max.double()

        for tensor in x_min, x_max, y_min, y_max:
            tensor.requires_grad_()
        input_image.requires_grad_(False) # already tested above

        try:
            original_parameters = reparametrize(
                x_min, x_max, y_min, y_max, reparametrization_h, reparametrization_w, inverse=True)

            torch.autograd.gradcheck(
                box_convolution_wrapper,
                    (input_image, x_min, x_max, y_min, y_max, max_input_h, max_input_w, \
                     reparametrization_factor, normalize, exact),
                eps=gradcheck_step / max(reparametrization_h, reparametrization_w),
                raise_exception=True)
        except Exception:
            print('Test %d failed at finite difference grad check w.r.t. parameters.' % test_idx)
            print('Normalize: %s' % normalize)
            print('h, w = %d, %d' % (h, w))
            print('x_min, x_max, y_min, y_max are:')
            for parameter in original_parameters:
                print(parameter)
            raise

if __name__ == '__main__':
    seed = int(time.time())
    seed = 1546545757
    torch.manual_seed(seed)
    random.seed(seed)
    print('Random seed is %d' % seed)

    for testing_function in test_integral_image, test_box_convolution_module:
        print('Running %s()...' % testing_function.__name__)
        # TODO [re]set random state etc.
        testing_function()
        print('OK')
