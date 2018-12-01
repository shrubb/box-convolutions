def test_integral_image():
    # TODO add ground truth tests
    from box_convolution.integral_image import IntegralImageFunction

    for i in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        stride_h, stride_w = 1, 1
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)

        torch.autograd.gradcheck(
            IntegralImageFunction.apply, input_image, eps=0.01, raise_exception=True)

def test_box_convolution_utility():
    def rand(a, b):
        return random.random() * (b - a) + a

    # TODO add ground truth tests
    from box_convolution.box_convolution_utility import BoxConvolutionFunction

    for i in tqdm(range(30)):
        batch_size = random.randint(1, 3)
        in_planes = random.randint(1, 3)
        n_windows = random.randint(1, 3)
        stride_h, stride_w = 1, 1
        h, w = random.randint(1+stride_h, 10), random.randint(1+stride_w, 10)

        input_image = torch.rand(batch_size, in_planes, h, w, requires_grad=True)
        
        x_min, x_max, y_min, y_max = (torch.empty(in_planes, n_windows) for _ in range(4))
        for plane_idx in range(in_planes):
            for window_idx in range(n_windows):
                x_min[plane_idx, window_idx] = rand(-h+1, h-2)
                y_min[plane_idx, window_idx] = rand(-w+1, w-2)
                x_max[plane_idx, window_idx] = rand(x_min[plane_idx, window_idx]+1, h-1)
                y_max[plane_idx, window_idx] = rand(y_min[plane_idx, window_idx]+1, w-1)

        torch.autograd.gradcheck(
            BoxConvolutionFunction.apply, (input_image, x_min, x_max, y_min, y_max),
            eps=0.01, raise_exception=True)

if __name__ == '__main__':
    import argparse
    import random
    import torch

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    for testing_function in test_integral_image, test_box_convolution_utility:
        print('Running %s()...' % testing_function.__name__)
        # TODO [re]set random state etc.
        testing_function()
        print('OK')
