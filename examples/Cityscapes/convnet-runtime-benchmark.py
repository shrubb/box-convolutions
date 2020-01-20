"""
    The script I use for benchmarking.
    
    Architectures are optimized for inference as if they were deployed in
    production: BatchNorms and Dropouts are absorbed, "void" class is removed.
    
    I managed to fully reproduce Torch7 runtimes from the paper for ENet
    and ERFNet; however, for some reason smaller models (e.g. ENet^-, ERFNet^-)
    are slower in PyTorch 1.0.1 than in Torch7.
"""
import torch

architecture = 'ENet' # choice: ENet / BoxENet / ERFNet / BoxERFNet / ENetMinus
device = 'cuda' # or 'cpu'
dtype = torch.float32

if device == 'cuda':
    assert torch.backends.cudnn.enabled
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

torch.set_num_threads(1)
torch.manual_seed(666)

# Cityscapes 1024x512 configuration
n_classes = 18
input_image = torch.rand((1, 3, 512, 1024), dtype=dtype, device=device)

print('Architecture:', architecture)
print('Device:', input_image.device)
print('Data type:', input_image.dtype)

from models.ERFNet import ERFNet, BoxERFNet
from models.ENet import ENet, BoxENet, BoxOnlyENet, ENetMinus

model = globals()[architecture](n_classes).to(input_image)

# optimize the model for inference
def remove_bn_and_dropout(module):
    for child_name, child in module.named_children():
        child_type = str(type(child))
        if 'BatchNorm' in child_type or 'Dropout' in child_type:
            module.__setattr__(child_name, torch.nn.Sequential())
        else:
            remove_bn_and_dropout(child)

from box_convolution import BoxConv2d
def set_boxconv_to_nonexact(module):
    if isinstance(module, BoxConv2d):
        module.exact = False

model.apply(set_boxconv_to_nonexact)
remove_bn_and_dropout(model)
model.eval()

# warm up
print('Output shape:', model(input_image).shape)

n_runs = 10 if device == 'cpu' else 160
import time

with torch.no_grad():
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        model(input_image)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

time_per_frame = (end - start) / n_runs
print('%.1f ms per frame / %.2f FPS' % (time_per_frame * 1000, 1 / time_per_frame))

