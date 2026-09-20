"""Build the optional Cython extensions:  python setup.py build_ext --inplace"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("src.filters.lms_cython", ["src/filters/lms_cython.pyx"], include_dirs=[np.get_include()]),
    Extension("src.filters.rls_cython", ["src/filters/rls_cython.pyx"], include_dirs=[np.get_include()]),
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
