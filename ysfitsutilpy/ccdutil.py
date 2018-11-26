'''
Collection of functions that are quite far from headerutil.
'''
from warnings import warn
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, Cutout2D, StdDevUncertainty
from astropy.wcs import WCS
from astropy import units as u

__all__ = ["cutout2CCDData", "load_ccd", "CCDData_astype",
           "make_errmap"]


def cutout2CCDData(ccd, position, size, mode='trim', fill_value=np.nan):
    ''' Converts the Cutout2D object to proper CCDData.
    Parameters
    ----------
    ccd: CCDData
        The ccd to be trimmed.
    position, size, mode, fill_value:
        See the ``ccdproc.trim_image`` doc.
    '''
    hdr_orig = ccd.header
    w = WCS(hdr_orig)
    cutout = Cutout2D(data=ccd.data, position=position, size=size, wcs=w,
                      mode=mode, fill_value=fill_value, copy=True)
    # Copy True just to avoid any contamination to the original ccd.

    nccd = CCDData(data=cutout.data, header=hdr_orig, unit=ccd.unit)
    nccd.header.update(cutout.wcs.to_header())
    ny, nx = nccd.data.shape
    nccd.header["NAXIS1"] = nx
    nccd.header["NAXIS2"] = ny

    nonlin = False
    for ctype in ccd.wcs.get_axis_types():
        if ctype["scale"] != "linear":
            nonlin = True
            break
    if nonlin:
        warn("Since Cutout2D is for small image crop, astropy do not "
             + "currently support distortion in WCS. This may result in "
             + "slightly inaccurate WCS calculation in the future.")

    return nccd


# FIXME: Remove it in the future.
def load_ccd(path, extension=0, uncertainty_ext="UNCERT", unit='adu'):
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

    ccd = CCDData(data=hdu.data, header=hdu.header, wcs=WCS(hdu.header),
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
