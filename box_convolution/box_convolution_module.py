import torch
import random

from .box_convolution_function import BoxConvolutionFunction

class BoxConv2d(torch.nn.Module):
    def __init__(self, in_planes, num_filters, h_max, w_max, stride_h=1, stride_w=1):
        super(BoxConv2d, self).__init__()
        self.in_planes = in_planes
        self.num_filters = num_filters
        self.h_max, self.w_max = h_max, w_max
        self.stride_h, self.stride_w = stride_h, stride_w
        self.exact = True

        self.x_min, self.x_max, self.y_min, self.y_max = \
            (torch.empty(in_planes, num_filters) for _ in range(4))
        self.reset_parameters()

    def reset_parameters(self):
        """
            One of the various possible random box initializations.
        """
        # TODO speed up
        # TODO use torch's random generator
        # TODO provide the algorithm used in all original paper's experiments?
        h_min, w_min = 2, 2
        for in_plane_idx in range(self.in_planes):
            for filter_idx in range(self.num_filters):
                center_h = random.uniform(-self.h_max*2/4.8+1+h_min/2, self.h_max*2/4.8-1-h_min/2)
                center_w = random.uniform(-self.w_max*2/4.8+1+w_min/2, self.w_max*2/4.8-1-w_min/2)
                height = 2 * random.uniform(
                    h_min/2, min((self.h_max-1)-center_h, center_h-(-self.h_max+1)))
                width  = 2 * random.uniform(
                    w_min/2, min((self.w_max-1)-center_w, center_w-(-self.w_max+1)))

                self.x_min[in_plane_idx, filter_idx] = center_h - height/2
                self.x_max[in_plane_idx, filter_idx] = center_h + height/2
                self.y_min[in_plane_idx, filter_idx] = center_w - width /2
                self.y_max[in_plane_idx, filter_idx] = center_w + width /2

    def draw_boxes(self, channels=None, resolution=(900, 900)):
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
            [ 60, 170, 255]]
        colors += [random_color() for _ in range(self.in_planes - len(colors))]
        colors = np.uint8(colors)

        x_min =  self.x_min.float()      / self.h_max * (resolution[0] / 2) + center[0]
        y_min =  self.y_min.float()      / self.w_max * (resolution[1] / 2) + center[1]
        x_max = (self.x_max.float() + 1) / self.h_max * (resolution[0] / 2) + center[0]
        y_max = (self.y_max.float() + 1) / self.w_max * (resolution[1] / 2) + center[1]

        for color, channel_idx in zip(colors, channels):
            for filter_idx in range(self.num_filters):
                cv2.rectangle(retval,
                    (y_min[channel_idx, filter_idx], x_min[channel_idx, filter_idx]),
                    (y_max[channel_idx, filter_idx], x_max[channel_idx, filter_idx]),
                    color.tolist(), 1)

        return retval

    def _clip_parameters(self):
        """
            If a filter's width or height is negative, reset it to the minimum allowed positive.
            If the filter is >=twice higher or wider than the input image, shrink it back.
        """
        pass # TODO

    def forward(self, input):
        return BoxConvolutionFunction.apply(
            input, self.x_min, self.x_max, self.y_min, self.y_max)
