from .astrometry import *
from .airmass import *
from .ccdutil import *
from .combutil import *
from .filemgmt import *
from .hdrutil import *
from .misc import *
from .preproc import *
from .imutil import *

from warnings import warn

try:
    import fitsio
    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False
    warn("fitsio is not installed on your machine. Everything will work fine, "
         "but the FITS I/O may be significantly slow in many occasions.", ImportWarning)