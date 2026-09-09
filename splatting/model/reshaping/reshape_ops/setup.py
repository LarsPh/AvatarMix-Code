from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ext_src_path = "."
sources = [os.path.join(ext_src_path, f) for f in ["bind.cpp", "nearest_face_kernel.cu", "point_mesh.cu"]]

setup(
    name="reshape_ops",
    version="0.1.0",
    url="https://github.com/prometheus-Lab-HKUST-GZ/SplattingAvatar",
    ext_modules=[
        CUDAExtension(
            name='reshape_ops',
            sources=sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format(ext_src_path)],
                "nvcc": ["-O2", "-I{}".format(ext_src_path)],
            },
            define_macros=[("WITH_CUDA", None)],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
