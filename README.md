# PC-SAFT

These functions implement the PC-SAFT equation of state. In addition to the hard chain and dispersion terms, these functions also include dipole, association and ion terms for use with these types of compounds. When the ion term is included it is also called electrolyte PC-SAFT (ePC-SAFT).

## Dependencies

The Numpy and Scipy packages are required.
When using the faster C++ functions [Cython](http://cython.org/) is needed, along with the [Eigen](https://github.com/eigenteam/eigen-git-mirror) package for linear algebra.

## Cython version

To speed up the original Python code the core functions have been rewritten in C++. These are then connected with the remaining Python code by using Cython. Using the Cython version gives a significant improvement in speed. The Cython code needs to be compiled before use, and for instructions on how to do this see the [Cython documentation](http://docs.cython.org/en/latest/src/quickstart/build.html).

## Author

* **Zach Baird** - [zmeri](https://github.com/zmeri)

## License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

When developing these functions the code from two other groups was used as references:
* Code from Joachim Gross (https://www.th.bci.tu-dortmund.de/cms/de/Forschung/PC-SAFT/Download/index.html)
* The MATLAB/Octave program written by Angel Martin and others (http://hpp.uva.es/open-source-software-eos/) 

