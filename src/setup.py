from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

sourcefiles  = ['sent2vec.pyx', 
                'fasttext.cc', 
                "args.cc", 
                "dictionary.cc", 
                "matrix.cc", 
                "shmem_matrix.cc",
                "qmatrix.cc", 
                "model.cc", 
                'real.cc', 
                'utils.cc', 
                'vector.cc', 
                'real.cc', 
                'productquantizer.cc']
compile_opts = ['-std=c++0x', '-Wno-cpp', '-pthread', '-Wno-sign-compare']
ext=[Extension('*',
            sourcefiles,
            extra_compile_args=compile_opts,
            language='c++',
            include_dirs=[numpy.get_include()],
            libraries=['rt'])]

setup(
  name='sent2vec',
  ext_modules=cythonize(ext)
)


