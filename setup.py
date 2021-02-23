import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

sourcefiles  = ['src/sent2vec.pyx',
                'src/fasttext.cc',
                'src/args.cc',
                'src/dictionary.cc',
                'src/matrix.cc',
                'src/shmem_matrix.cc',
                'src/qmatrix.cc',
                'src/model.cc',
                'src/real.cc',
                'src/utils.cc',
                'src/vector.cc',
                'src/real.cc',
                'src/productquantizer.cc']
compile_opts = ['-std=c++0x', '-Wno-cpp', '-pthread', '-Wno-sign-compare']
libraries = ['rt']
if sys.platform == 'darwin':
    libraries = []
ext=[Extension('*',
            sourcefiles,
            extra_compile_args=compile_opts,
            language='c++',
            include_dirs=[numpy.get_include()],
            libraries=libraries)]

setup(
  name='sent2vec',
  install_requires=['Cython>=0.29.13', 'numpy>=1.17.1'],
  ext_modules=cythonize(ext)
)
