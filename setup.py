from setuptools import setup
import torch.utils.cpp_extension

import os

source_root = 'src'
source_files_cpp = [
	'integral_image_interface.cpp',
	'integral_image.cpp',
	'box_convolution_interface.cpp',
	'box_convolution.cpp',
	'bind.cpp'
]
source_files_cuda = [
    'integral_image_cuda.cu',
    'box_convolution_cuda_forward.cu',
    'box_convolution_cuda_misc.cu'
]
source_files = source_files_cpp + source_files_cuda

cpp_cuda = torch.utils.cpp_extension.CUDAExtension(
    name='box_convolution_cpp_cuda',
    sources=[os.path.join(source_root, file) for file in source_files],
    include_dirs=[source_root]
)

setup(
    name='box_convolution',
    ext_modules=[cpp_cuda],
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension}
)
