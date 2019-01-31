import torch

import box_convolution_cpp_cuda as cpp_cuda

def reparametrize(
    x_min, x_max, y_min, y_max, max_input_h, max_input_w, inplace=False, inverse=False):
    """
        If `inverse is False`, scale module's parameters so that their range becomes
        approximately [-1; 1]. Otherwise, do the inverse operation.

        This hack is unfortunately needed for the parameters to work with variants of SGD.
        Without this "reparametrization", box sizes' gradients will be extremely small.

        If `not inplace`, returns 4 new tensors, otherwise modifies the given ones.
    """
    scalar_h = max_input_h if inverse else (1 / max_input_h)
    scalar_w = max_input_w if inverse else (1 / max_input_w)

    with torch.no_grad():
        if inplace:
            x_min *= scalar_h
            x_max *= scalar_h
            y_min *= scalar_w
            y_max *= scalar_w
        else:
            return x_min * scalar_h, x_max * scalar_h, y_min * scalar_w, y_max * scalar_w

# TODO: rename `x_` and `y_` to `h_` and `w_`
class BoxConvolutionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, x_min, x_max, y_min, y_max, max_input_h, max_input_w, normalize):
        # store all non-tensor arguments in `ctx`
        ctx.normalize = normalize
        ctx.max_input_h = max_input_h
        ctx.max_input_w = max_input_w

        x_min, x_max, y_min, y_max = \
            reparametrize(x_min, x_max, y_min, y_max, max_input_h, max_input_w, inverse=True)

        input_integrated = cpp_cuda.integral_image(input)
        
        output = cpp_cuda.box_convolution_forward(
            input_integrated, x_min, x_max, y_min, y_max, normalize)

        ctx.save_for_backward(
            input_integrated, x_min, x_max, y_min, y_max, output if normalize else None)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_integrated, x_min, x_max, y_min, y_max, output = ctx.saved_variables
        if output is None:
            output = torch.empty(0) # to satisfy `box_convolution_backward`'s signature

        retval = cpp_cuda.box_convolution_backward(
            input_integrated, x_min, x_max, y_min, y_max, grad_output, output,
            ctx.max_input_h, ctx.max_input_w, ctx.normalize, *ctx.needs_input_grad[:5])
            
        return tuple(retval) + (None,) * 3 # `None`s for `max_input_h, max_input_w, normalize`
