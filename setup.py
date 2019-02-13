from setuptools import setup
import torch.utils.cpp_extension

# monkey-patch for parallel compilation
# https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
def parallelCCompile(
    self, sources, output_dir=None, macros=None, include_dirs=None, debug=0,
    extra_preargs=None, extra_postargs=None, depends=None):

    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = \
        self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    
    # parallel code
    from multiprocessing import cpu_count
    try:
        n_processes = cpu_count() # number of parallel compilations
    except NotImplementedError:
        print('multiprocessing.cpu_count() failed, building on 1 core')
        n_processes = 1

    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    
    import multiprocessing.pool
    multiprocessing.pool.ThreadPool(n_processes).map(_single_compile, objects)
    
    return objects

import distutils.ccompiler
distutils.ccompiler.CCompiler.compile = parallelCCompile

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
    'box_convolution_cuda_backward.cu',
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
