import torch
import random

from .box_convolution_function import BoxConvolutionFunction, reparametrize
import box_convolution_cpp_cuda as cpp_cuda

class BoxConv2d(torch.nn.Module):
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
                        min_h/2, min((max_h-1)-center_h, center_h-(-max_h+1)))
                    width  = 2 * random.uniform(
                        min_w/2, min((max_w-1)-center_w, center_w-(-max_w+1)))

                    self.x_min[in_plane_idx, filter_idx] = center_h - height/2
                    self.x_max[in_plane_idx, filter_idx] = center_h + height/2
                    self.y_min[in_plane_idx, filter_idx] = center_w - width /2
                    self.y_max[in_plane_idx, filter_idx] = center_w + width /2

        reparametrize(
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.reparametrization_h, self.reparametrization_w, inplace=True)

    def draw_boxes(self, channels=None, resolution=(600, 600)):
        """
            Plots all rectangles corresponding to box filters. Useful for debugging.
            Returns the resulting image, an (H x W x 3) tensor.

            channels:   list of input channels to draw boxes for. Default: draw all boxes.
            resolution: return image resolution. Default: 900 x 900
        """
        import cv2
        import numpy as np

        if channels is None:
            channels = range(self.in_planes)
        
        retval = np.zeros(resolution + (3,), dtype=np.uint8)

        # draw gray lines at center
        center = [resolution[0] // 2, resolution[1] // 2]
        retval[center[0]] = 70
        retval[:, center[1]] = 70

        def random_color():
            color = np.random.rand(3) * 255
            mix = np.float64([220, 220, 220])
            return np.uint8(0.5 * (color + mix))

        # heuristic for single-plane inputs
        num_colors = self.num_filters if len(channels) == 1 else self.num_filters

        if not hasattr(self, '_colors'):
            colors = [
                [255,   0,   0],
                [  0, 255,   0],
                [  0,   0, 255],
                [255, 255, 255],
                [255, 255,   0],
                [255,   0, 255],
                [  0, 255, 255],
                [130, 130, 130],
                [255,  60, 160],
                [ 60, 170, 255]] * 2
            colors += [random_color() for _ in range(num_colors - len(colors))]
            self._colors = np.uint8(colors)

        x_min, x_max, y_min, y_max = (p.float() for p in self.get_actual_parameters())
        x_min =  x_min      / self.max_input_h * (resolution[0] / 2) + center[0]
        y_min =  y_min      / self.max_input_w * (resolution[1] / 2) + center[1]
        x_max = (x_max + 1) / self.max_input_h * (resolution[0] / 2) + center[0]
        y_max = (y_max + 1) / self.max_input_w * (resolution[1] / 2) + center[1]

        for channel_idx in channels:
            for filter_idx in range(self.num_filters):
                color = self._colors[filter_idx if len(channels) == 1 else channel_idx]

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
            As parameters are scaled to roughly be in [-1; 1] range (or even smaller -- depends on
            your task), they don't represent actual box coordinates. This function returns the 
            **real** parameters as it they weren't rescaled.
        """
        return reparametrize(
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.reparametrization_h, self.reparametrization_w, inplace=False, inverse=True)

    def _clip_parameters(self):
        """
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
            self._clip_parameters()

        return self

    def forward(self, input):
        if self.training:
            self._clip_parameters()

        return BoxConvolutionFunction.apply(
            input, self.x_min, self.x_max, self.y_min, self.y_max,
            self.reparametrization_h, self.reparametrization_w, self.normalize)
