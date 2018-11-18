'''
Simple mathematical functions that will be used throughout this package. Some
might be useful outside of this package.
'''
import ccdproc
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize

__all__ = ["fitsxy2py", "give_stats", "calc_airmass", "dB2epadu", "epadu2dB"]


def fitsxy2py(fits_section):
    ''' Given FITS section in str, returns the slices in python convention.
    Note
    ----
    >>> np.eye(5)[ccdproc.utils.slices.slice_from_string('[1:2,:]',
        fits_convention=True)]
    # array([[1., 0.],
    #       [0., 1.],
    #       [0., 0.],
    #       [0., 0.],
    #       [0., 0.]])
    '''
    slicer = ccdproc.utils.slices.slice_from_string
    sl = slicer(fits_section, fits_convention=True)
    return sl


def give_stats(data, percentiles=[1, 99], N_extrema=None):
    ''' Calculates simple statistics.
    Parameters
    ----------
    data: array-like
        The data to be analyzed.
    percentiles: list-like, optional
        The percentiles to be calculated.
    N_extrema: int, optinoal
        The number of low and high elements to be returned when the whole data
        are sorted. If ``None``, it will not be calculated. If ``1``, it is
        identical to min/max values.

    Example
    -------
    >>> bias = CCDData.read("bias_bin11.fits")
    >>> dark = CCDData.read("pdark_300s_27C_bin11.fits")
    >>> percentiles = [0.1, 1, 5, 95, 99, 99.9]
    >>> stats.give_stats(bias, percentiles=percentiles, N_extrema=5)
    >>> stats.give_stats(dark, percentiles=percentiles, N_extrema=5)
    '''
    data = np.atleast_1d(data)

    result = {}

    d_num = np.size(data)
    d_min = np.min(data)
    d_pct = np.percentile(data, percentiles)
    d_max = np.max(data)
    d_avg = np.mean(data)
    d_med = np.median(data)
    d_std = np.std(data, ddof=1)

    zs = ImageNormalize(data, interval=ZScaleInterval())
    d_zmin = zs.vmin
    d_zmax = zs.vmax

    result["N"] = d_num
    result["min"] = d_min
    result["max"] = d_max
    result["avg"] = d_avg
    result["med"] = d_med
    result["std"] = d_std
    result["percentiles"] = d_pct
    result["zmin"] = d_zmin
    result["zmax"] = d_zmax

    if N_extrema is not None:
        data_flatten = np.sort(data, axis=None)  # axis=None will do flatten.
        d_los = data_flatten[:N_extrema]
        d_his = data_flatten[-1 * N_extrema:]
        result["ext_lo"] = d_los
        result["ext_hi"] = d_his

    return result


def calc_airmass(zd_deg=None, cos_zd=None, scale=750.):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.
    Note
    ----
    Wiki:
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere
    Identical to the airmass calculation at a given ZD of IRAF's
    asutil.setairmass:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass

    Parameters
    ----------
    zd_deg: float, optional
        The zenithal distance in degrees
    cos_zd: float, optional
        The cosine of zenithal distance. If given, ``zd_deg`` is not used.
    scale: float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere. In IRAF documentation, it is mistakenly written that
        this ``scale`` is the "scale height".
    '''
    if zd_deg is None and cos_zd is None:
        raise ValueError("Either zd_deg or cos_zd should not be None.")

    if cos_zd is None:
        cos_zd = np.cos(np.deg2rad(zd_deg))

    am = np.sqrt((scale * cos_zd)**2 + 2 * scale + 1) - scale * cos_zd

    return am


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB):
    return 5 / 10**(gain_dB / 20)


def epadu2dB(gain_epadu):
    return 20 * np.log10(5 / gain_epadu)
