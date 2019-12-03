from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cust_act_cpp',
      ext_modules=[cpp_extension.CppExtension('cust_act_cpp', ['cust_act.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})