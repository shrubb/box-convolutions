import torch
import random

from .integral_image import IntegralImageFunction
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
            Here's the one used in all original paper's experiments. TODO
        """
        # TODO speed up
        # TODO use torch's random generator
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

    def _clip_parameters(self):
        """
            If a filter's width or height is negative, reset it to the minimum allowed positive.
            If the filter is >=twice higher or wider than the input image, shrink it back.
        """
        pass

    def forward(self, input):
        input_integrated = IntegralImageFunction.apply(input)
        conv_output = BoxConvolutionFunction.apply(
            input_integrated, self.x_min, self.x_max, self.y_min, self.y_max)
        return conv_output
