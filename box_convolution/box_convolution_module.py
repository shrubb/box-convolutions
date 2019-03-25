import torch
import random

from .box_convolution_function import BoxConvolutionFunction, reparametrize
import box_convolution_cpp_cuda as cpp_cuda

class BoxConv2d(torch.nn.Module):
    """
        Module that performs depthwise box convolution.
        Convolves each of the incoming channels with `num_filters` different,
        possibly normalized, box kernels.

        Input : `(batch_size) x (in_planes) x (h) x (w)`
        Output: `(batch_size) x (in_planes*num_filters) x (h) x (w)`

        Constructor arguments:

        in_planes: int
            Number of channels in the input image (as in Conv2d).
        num_filters: int
            Number of filters to apply per channel (as in depthwise Conv2d).
        max_input_h, max_input_w: int
            Maximum estimated height/width of future input images. This parameter does
            not strictly bind input images to certain sizes. However, this is used
            when clipping the boxes to detect if some box has become too large or has
            drifted too far away from the image. See `_clip_parameters()` for details.
        reparametrization_factor: float
            In module parameters, boxes are not represented directly by their
            relative pixel coordinates, because then the gradients will usually
            be too small. Rather, here they are scaled into a range that is inside
            [-1; 1] by `1 / (reparametrization_factor * max_input_[h/w])`.
            When setting up training, generate a video of boxes using `draw_boxes()`.
            If they move too slow, increasing this parameter might help. If they
            converge too fast, reduce this value.
        stride_h, stride_w: int
            Stride (as in Conv2d). Not yet implemented.
        normalize: bool
            If `False`, computes sums over boxes (traditional box filters).
            If `True`, computes averages over boxes (normalized box filters).

        Useful fields (change after construction):

        exact: bool
            If `False`, box coordinates are rounded (towards smaller box size) before
            the output is computed. Significantly faster, but might be harmful for
            convergence. Well, still, often works for some reason, so try and see.
            Default: `True`.
    """
    def __init__(self,
        in_planes, num_filters, max_input_h, max_input_w,
        reparametrization_factor=8, stride_h=1, stride_w=1, normalize=True):

        super(BoxConv2d, self).__init__()
        self.in_planes = in_planes
        self.num_filters = num_filters
        self.max_input_h, self.max_input_w = max_input_h, max_input_w
        # default reparametrization; can be changed instead of setting a separate learning rate
        self.reparametrization_h = max_input_h * reparametrization_factor
        self.reparametrization_w = max_input_w * reparametrization_factor
        self.stride_h, self.stride_w = stride_h, stride_w
        assert stride_h == 1 and stride_w == 1, 'Sorry, strides are NYI'
        self.normalize = normalize
        self.exact = True

        self.x_min, self.x_max, self.y_min, self.y_max = \
            (torch.nn.Parameter(torch.empty(in_planes, num_filters)) for _ in range(4))
        self.reset_parameters()

    def reset_parameters(self):
        """
            One of the various possible random box initializations.
        """
        with torch.no_grad():
            # TODO speed up
            # TODO use torch's random generator
            # TODO provide the algorithm used in all original paper's experiments?
            max_h, max_w = self.max_input_h, self.max_input_w
            min_h, min_w = 2, 2
            for in_plane_idx in range(self.in_planes):
                for filter_idx in range(self.num_filters):
                    center_h = random.uniform(
                        -max_h*2/4.8+1+min_h/2, max_h*2/4.8-1-min_h/2)
                    center_w = random.uniform(
                        -max_w*2/4.8+1+min_w/2, max_w*2/4.8-1-min_w/2)
                    height = 2 * random.uniform(
                        min_h/2, min((max_h*2/4.8-1)-center_h, center_h-(-max_h*2/4.8+1)))
                    width  = 2 * random.uniform(
                        min_w/2, min((max_w*2/4.8-1)-center_w, center_w-(-max_w*2/4.8+1)))

                    self.x_min[in_plane_idx, filter_idx] = (center_h - height/2) * 1.5
                    self.x_max[in_plane_idx, filter_idx] = (center_h + height/2) * 1.5
                    self.y_min[in_plane_idx, filter_idx] = (center_w - width /2) * 1.5
                    self.y_max[in_plane_idx, filter_idx] = (center_w + width /2) * 1.5

        reparametrize(
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.reparametrization_h, self.reparametrization_w, inplace=True)

    def draw_boxes(self, channels=None, resolution=(600, 600), weights=None):
        """
            Plot all rectangles corresponding to box filters. Useful for debugging.
            Return the resulting image, an (H x W x 3) tensor.

            channels:   List of input channels to draw boxes for.
                        Default: `[0, 1, ..., self.in_planes-1]` (draw all boxes).
            resolution: Tuple (h, w) -- returned image resolution.
                        Default: (600, 600)
            weights:    `len(channels) x self.num_filters` array of values in [0; 1] that define
                        "importance" of each box (e.g. function of weights from a successive
                        convolution). More important boxes are given a brigter color, unimportant
                        are drawn almost transparent.
                        Default: `numpy.ones((len(channels), self.num_filters))`.
        """
        import cv2
        import numpy as np

        if channels is None:
            channels = range(self.in_planes)

        weights_shape = (len(channels), self.num_filters)
        if weights is None:
            weights = np.ones(weights_shape)
        weights = weights.cpu().float().numpy().reshape(weights_shape)
        weights.clip(0.01, 1.0, out=weights)
        
        retval = np.zeros(resolution + (3,), dtype=np.uint8)

        # draw gray lines at center
        center = [resolution[0] // 2, resolution[1] // 2]
        retval[center[0], :] = 70
        retval[:, center[1]] = 70

        colors = np.array([
            [255,   0,   0],
            [  0, 255,   0],
            [  0,   0, 255],
            [255, 255, 255],
            [255, 255,   0],
            [255,   0, 255],
            [  0, 255, 255],
            [ 47,  20, 255],
            [255,  60, 160],
            [ 60, 170, 255],
            [ 30, 105, 210],
            [222, 196, 176],
            [212, 255, 127],
            [250, 206, 135],
            [ 50, 205,  50],
            [  0, 165, 255],
            [ 60,  20, 220],
            [170, 178,  32]], dtype=np.float32)

        x_min, x_max, y_min, y_max = (p.float() for p in self.get_actual_parameters())
        x_min =  x_min      / self.max_input_h * (resolution[0] / 2) + center[0]
        y_min =  y_min      / self.max_input_w * (resolution[1] / 2) + center[1]
        x_max = (x_max + 1) / self.max_input_h * (resolution[0] / 2) + center[0]
        y_max = (y_max + 1) / self.max_input_w * (resolution[1] / 2) + center[1]

        for channel_idx in channels:
            for filter_idx in range(self.num_filters):
                box_weight = weights[channel_idx, filter_idx]
                # heuristic for single-plane inputs
                color = colors[(filter_idx if len(channels) == 1 else channel_idx) % len(colors)]
                # take weights into account
                color = (color * box_weight).astype(int)

                param_2d_idx = channel_idx, filter_idx
                x_min_curr = x_min[param_2d_idx]
                x_max_curr = x_max[param_2d_idx]
                y_min_curr = y_min[param_2d_idx]
                y_max_curr = y_max[param_2d_idx]

                # if a rect has negative size, fill it
                box_is_invalid = x_min_curr > x_max_curr or y_min_curr > y_max_curr
                thickness = -1 if box_is_invalid else round(resolution[0] / 500 + 0.5)

                cv2.rectangle(
                    retval, (y_min_curr, x_min_curr), (y_max_curr, x_max_curr),
                    color.tolist(), thickness)

        return retval

    def get_actual_parameters(self):
        """
            As parameters are scaled (see `reparametrization_factor`), they don't
            represent actual box coordinates.

            Return the **real** parameters (i.e. actual relative box coordinates)
            as if they weren't rescaled.
        """
        return reparametrize(
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.reparametrization_h, self.reparametrization_w, inplace=False, inverse=True)

    def _clip_parameters(self):
        """
            Internal method, do not invoke as a user.

            Dirty parameter fix for projected gradient descent:
            - If a filter's width or height is negative, reset it to the minimum allowed positive.
            - If the filter is >=twice higher or wider than the input image, shrink it back.
        """
        cpp_cuda.clip_parameters(
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.reparametrization_h, self.reparametrization_w,
            self.max_input_h, self.max_input_w, self.exact)

    def train(self, mode=True):
        self.training = mode
        if mode is False:
            # TODO would be good to also precompute rounded parameters and areas
            self._clip_parameters()

        return self

    def forward(self, input):
        if self.training:
            self._clip_parameters()

        return BoxConvolutionFunction.apply(
            input, self.x_min, self.x_max, self.y_min, self.y_max,
            self.reparametrization_h, self.reparametrization_w, self.normalize, self.exact)
