from setuptools import setup
from Cython.Build import cythonize
import numpy
# ext = Extension(name='ctree', sources=['ctree.pxd'])
# ext = Extension(name='cytree', sources=['cytree.pyx'],
#                 # extra_compile_args=['-O3'],
#                 language="c++")
setup(ext_modules=cythonize('cytree.pyx'),include_dirs=[numpy.get_include()], extra_compile_args=['-O3'])
