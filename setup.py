# -*- coding: utf-8 -*-
"""
Setup file for compiling to Cython module

@author: Zach Baird
"""
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os

ext_modules = [
    Extension("pcsaft",
        sources=["pcsaft.pyx"],
        language="c++")]

with open("docs/README.rst", "r") as fh:
    long_description = fh.read()

requirements_path = './requirements.txt'
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='pcsaft',
      version='1.3.0',
      author="Zach Baird",
      description="The PC-SAFT equation of state, including dipole, association and ion terms.",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      url="https://github.com/zmeri/PC-SAFT",
      packages=find_packages(),
      install_requires=install_requires,
      classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux",
      ],
      ext_modules=cythonize(ext_modules, language_level="3"))
