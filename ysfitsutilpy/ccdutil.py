'''
Collection of functions that are quite far from headerutil.
'''
from warnings import warn
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, Cutout2D, StdDevUncertainty
from astropy.wcs import WCS
from astropy import units as u

from .misc import binning

__all__ = ["cutccd", "bin_ccd", "load_ccd", "CCDData_astype",
           "make_errmap"]


# TODO: add LTV-like keys to the header.
def cutccd(ccd, position, size, mode='trim', fill_value=np.nan):
    ''' Converts the Cutout2D object to proper CCDData.
    Parameters
    ----------
    ccd: CCDData
        The ccd to be trimmed.

    position : tuple or `~astropy.coordinates.SkyCoord`
        The position of the cutout array's center with respect to
        the ``data`` array.  The position can be specified either as
        a ``(x, y)`` tuple of pixel coordinates or a
        `~astropy.coordinates.SkyCoord`, in which case ``wcs`` is a
        required input.

    size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array along each axis.  If ``size``
        is a scalar number or a scalar `~astropy.units.Quantity`,
        then a square cutout of ``size`` will be created.  If
        ``size`` has two elements, they should be in ``(ny, nx)``
        order.  Scalar numbers in ``size`` are assumed to be in
        units of pixels.  ``size`` can also be a
        `~astropy.units.Quantity` object or contain
        `~astropy.units.Quantity` objects.  Such
        `~astropy.units.Quantity` objects must be in pixel or
        angular units.  For all cases, ``size`` will be converted to
        an integer number of pixels, rounding the the nearest
        integer.  See the ``mode`` keyword for additional details on
        the final cutout size.

        .. note::
            If ``size`` is in angular units, the cutout size is
            converted to pixels using the pixel scales along each
            axis of the image at the ``CRPIX`` location.  Projection
            and other non-linear distortions are not taken into
            account.

    wcs : `~astropy.wcs.WCS`, optional
        A WCS object associated with the input ``data`` array.  If
        ``wcs`` is not `None`, then the returned cutout object will
        contain a copy of the updated WCS for the cutout data array.

    mode : {'trim', 'partial', 'strict'}, optional
        The mode used for creating the cutout data array.  For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap of the
        cutout array and the input ``data`` array is sufficient.
        For the ``'strict'`` mode, the cutout array has to be fully
        contained within the ``data`` array, otherwise an
        `~astropy.nddata.utils.PartialOverlapError` is raised.   In
        all modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'``
        mode, positions in the cutout array that do not overlap with
        the ``data`` array will be filled with ``fill_value``.  In
        ``'trim'`` mode only the overlapping elements are returned,
        thus the resulting cutout array may be smaller than the
        requested ``shape``.

    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the
        cutout array that do not overlap with the input ``data``.
        ``fill_value`` must have the same ``dtype`` as the input
        ``data`` array.
    '''
    hdr_orig = ccd.header
    w = WCS(hdr_orig)
    cutout = Cutout2D(data=ccd.data, position=position, size=size, wcs=w,
                      mode=mode, fill_value=fill_value, copy=True)
    # Copy True just to avoid any contamination to the original ccd.

    nccd = CCDData(data=cutout.data,
                   header=hdr_orig,
                   wcs=cutout.wcs,
                   unit=ccd.unit)
    ny, nx = nccd.data.shape
    nccd.header["NAXIS1"] = nx
    nccd.header["NAXIS2"] = ny

    nonlin = False
    try:
        for ctype in ccd.wcs.get_axis_types():
            if ctype["scale"] != "linear":
                nonlin = True
                break
    except AttributeError:
        nonlin = False

    if nonlin:
        warn("Since Cutout2D is for small image crop, astropy do not "
             + "currently support distortion in WCS. This may result in "
             + "slightly inaccurate WCS calculation in the future.")

    return nccd


def bin_ccd(ccd, factor_x=1, factor_y=1):
    if factor_x == 1 and factor_y == 1:
        return ccd
    _ccd = ccd.copy()
    _ccd.data = binning(_ccd.data, factor_x=factor_x, factor_y=factor_y)
    hdr = _ccd.header
    hdr.add_history(f"Binned by (xbin, ybin) = ({factor_x}, {factor_y})")
    hdr["XBINFCT"] = (factor_x, "Binning done after the observation in X direction")
    hdr["YBINFCT"] = (factor_y, "Binning done after the observation in Y direction")
    _ccd.header = hdr
    return _ccd


# FIXME: Remove it in the future.
def load_ccd(path, extension=0, usewcs=True, uncertainty_ext="UNCERT",
             unit='adu'):
    '''remove it when astropy updated:
    Note
    ----
    CCDData.read cannot read TPV WCS:
    https://github.com/astropy/astropy/issues/7650
    '''
    hdul = fits.open(path)
    hdu = hdul[extension]
    try:
        unc = StdDevUncertainty(hdul[uncertainty_ext].data)
    except KeyError:
        unc = None

    w = None
    if usewcs:
        w = WCS(hdu.header)

    ccd = CCDData(data=hdu.data, header=hdu.header, wcs=w,
                  uncertainty=unc, unit=unit)
    return ccd


def CCDData_astype(ccd, dtype='float32', uncertainty_dtype=None):
    ''' Assign dtype to the CCDData object (numpy uses float64 by default).
    Parameters
    ----------
    ccd: CCDData
        The ccd to be astyped.
    dtype: dtype-like
        The dtype to be applied to the data
    uncertainty_dtype: dtype-like
        The dtype to be applied to the uncertainty. Be default, use the
        same dtype as data (``uncertainty_dtype = dtype``).

    Example
    -------
    >>> from astropy.nddata import CCDData
    >>> import numpy as np
    >>> ccd = CCDData.read("bias001.fits", ext=0)
    >>> ccd.uncertainty = np.sqrt(ccd.data)
    >>> ccd = yfu.CCDData_astype(ccd, dtype='int16', uncertainty_dtype='float32')
    '''
    nccd = ccd.copy()
    nccd.data = nccd.data.astype(dtype)

    try:
        if uncertainty_dtype is None:
            uncertainty_dtype = dtype
        nccd.uncertainty.array = nccd.uncertainty.array.astype(
            uncertainty_dtype)
    except AttributeError:
        # If there is no uncertainty attribute in the input ``ccd``
        pass

    return nccd


def make_errmap(ccd, gain_epadu=1, rdnoise_electron=0,
                subtracted_dark=None):
    ''' Calculate the simple error map.
    Parameters
    ----------
    ccd: array-like
        The ccd data which will be used to generate error map. It must be bias
        subtracted. If dark is subtracted, give ``subtracted_dark``. This
        array will be added to ``ccd.data`` and used to calculate the Poisson
        noise term. If the amount of this subtracted dark is negligible, you
        may just set ``subtracted_dark = None`` (default).
    gain: float, array-like, or Quantity, optional.
        The effective gain factor in ``electron/ADU`` unit.
    rdnoise: float, array-like, or Quantity, optional.
        The readout noise. Put ``rdnoise=0`` will calculate only the Poissonian
        error. This is useful when generating noise map for dark frames.
    subtracted_dark: array-like
        The subtracted dark map.

    Example
    -------
    >>> from astropy.nddata import CCDData, StdDevUncertainty
    >>> ccd = CCDData.read("obj001.fits", ext=0)
    >>> dark = CCDData.read("master_dark.fits", ext=0)
    >>> params = dict(gain_epadu = ccd.header["GAIN"],
    >>>               rdnoise_electron = ccd.header["RDNOISE"],
    >>>               subtracted_dark = dark.data)
    >>> ccd.uncertainty = StdDevUncertainty(make_errmap(ccd, **params))
    '''
    data = ccd.copy()

    if isinstance(data, CCDData):
        data = data.data

    data[data < 0] = 0  # make all negative pixel to 0

    if isinstance(gain_epadu, u.Quantity):
        gain_epadu = gain_epadu.to(u.electron / u.adu).value
    elif isinstance(gain_epadu, str):
        gain_epadu = float(gain_epadu)

    if isinstance(rdnoise_electron, u.Quantity):
        rdnoise_electron = rdnoise_electron.to(u.electron)
    elif isinstance(rdnoise_electron, str):
        rdnoise_electron = float(rdnoise_electron)

    # Get Poisson noise
    if subtracted_dark is not None:
        dark = subtracted_dark.copy()
        if isinstance(dark, CCDData):
            dark = dark.data
        # If subtracted dark is negative, this may cause negative pixel in ``data``:
        data += dark

    var_Poisson = data / gain_epadu  # (data * gain) / gain**2 to make it ADU
    var_RDnoise = (rdnoise_electron / gain_epadu)**2

    errmap = np.sqrt(var_Poisson + var_RDnoise)

    return errmap
