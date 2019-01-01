from setuptools import setup
import torch.utils.cpp_extension

import os

source_root = 'src'
source_files = [
	'integral_image.cpp',
	'box_convolution.cpp'
]

cpp_cuda = torch.utils.cpp_extension.CUDAExtension(
    name='box_convolution_cpp_cuda',
    sources=[os.path.join(source_root, file) for file in source_files]
)

setup(
    name='box_convolution',
    ext_modules=[cpp_cuda],
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension}
)
