# -*- coding: utf-8 -*-
"""
Distutils setup file for compiling to Cython module

@author: Zach Baird
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("pcsaft_electrolyte",
        sources=["pcsaft_electrolyte.pyx"],
        language="c++")]

setup(name='PC-SAFT electrolyte',
      ext_modules=cythonize(ext_modules, language_level="3"))
