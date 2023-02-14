''' This module consists of some utilities that resemble IRAF's IMUTIL package.
The python versions have identical names (imcombine, imarith, etc), while the script versions have
different names to avoid namespack crash (pimcombine, pimarith, etc).
'''
from .imcombine import *
from .imcopy import *
from .imarith import *
from .imsmooth import *
