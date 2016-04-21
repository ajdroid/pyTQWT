# pyTQWT
Python implementation of [tunable Q-factor](http://eeweb.poly.edu/iselesni/TQWT/) based image fusion, based on [this](eeweb.poly.edu/iselesni/pubs/TQWT_2011.pdf) paper

## How To Run
1. Have ipython notebook, scipy and numpy 
2. open 1DTQWT.ipynb and run to get demo
3. Load your own data or experiment with different images from the samples provided

## Known issues
Numpy based ifft tends to leave some imaginary parts in. Ignore ComplexWarning thrown from this.
