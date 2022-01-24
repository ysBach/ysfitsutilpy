from .astrometry import *
from .airmass import *
from .combutil import *
from .filemgmt import *
from .hduutil import *
from .misc import *
from .preproc import *
from .imutil import *
from .fitting import *

from warnings import warn

try:
    import fitsio
except ImportError:
    warn("fitsio is not installed on your machine. Everything will work fine, "
         "but the FITS I/O may be significantly slow in many occasions.", ImportWarning)

try:
    import numexpr as ne
except ImportError:
    warn("numexpr is not installed on your machine. Everything will work fine, "
         "but numpy arithematics can be few times slower in few occasions.", ImportWarning)