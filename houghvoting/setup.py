from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='houghvoting',
    ext_modules=[
        CUDAExtension('hv_cuda', [
            'src/hv_cuda.cpp',
            'src/hv_cuda_kernel.cu',
        ])
    ],
    zipsafe=False,
    cmdclass={
        'build_ext': BuildExtension
    })