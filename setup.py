# -*- coding: utf-8 -*-
"""
Setup file for compiling to Cython module

@author: Zach Baird
"""
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("pcsaft_electrolyte",
        sources=["pcsaft_electrolyte.pyx"],
        language="c++")]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pcsaft',
      version='1.0.0',
      author="Zach Baird",
      description="The PC-SAFT equation of state, including dipole, association and ion terms.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/zmeri/PC-SAFT",
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux",
      ],
      ext_modules=cythonize(ext_modules, language_level="3"))
