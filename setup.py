# -*- coding: utf-8 -*-
"""
Setup file for compiling to Cython module

@author: Zach Baird
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("pcsaft_electrolyte",
        sources=["pcsaft_electrolyte.pyx"],
        language="c++")]

setup(name='pcsaft',
      version='1.0',
      ext_modules=cythonize(ext_modules, language_level="3"))
