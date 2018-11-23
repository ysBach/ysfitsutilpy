'''
Simple mathematical functions that will be used throughout this package. Some
might be useful outside of this package.
'''
from warnings import warn
import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import AltAz
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.io import fits
import ccdproc

__all__ = ["fitsxy2py", "give_stats", "calc_airmass", "airmass_obs",
           "dB2epadu", "epadu2dB"]


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


def give_stats(item, extension=0, percentiles=[1, 99], N_extrema=None):
    ''' Calculates simple statistics.
    Parameters
    ----------
    item: array-like or path-like
        The nddata or path to a FITS file to be analyzed.
    extension: int, str, optional
        The extension if ``item`` is the path to the FITS file.
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
    >>> give_stats(bias, percentiles=percentiles, N_extrema=5)
    >>> give_stats(dark, percentiles=percentiles, N_extrema=5)
    Or just simply
    >>> give_stats("bias_bin11.fits", percentiles=percentiles, N_extrema=5)
    '''
    try:
        data = np.atleast_1d(item)
    except TypeError:
        data = fits.open(item)[extension].data

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


def airmass_obs(targetcoord, obscoord, ut, exptime, scale=750., full=False):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.
    Note
    ----
    Wiki:
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere
    Identical to the airmass calculation for a given observational run of
    IRAF's asutil.setairmass:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass
    Partly contributed by Kunwoo Kang (Seoul National University) in Apr 2018.

    '''
    if not isinstance(ut, Time):
        warn("ut is not Time object. Assume format='isot', scale='utc'.")
        ut = Time(ut, format='isot', scale='utc')
    if not isinstance(exptime, u.Quantity):
        warn("exptime is not astropy Quantity. Assume it is in seconds.")
        exptime = exptime * u.s

    t_start = ut
    t_mid = ut + exptime / 2
    t_final = ut + exptime

    altaz = {"alt": [], "az": [], "zd": [], "airmass": []}
    for t in [t_start, t_mid, t_final]:
        C_altaz = AltAz(obstime=t, location=obscoord)
        target = targetcoord.transform_to(C_altaz)
        alt = target.alt.to_string(unit=u.deg, sep=':')
        az = target.az.to_string(unit=u.deg, sep=':')
        zd = target.zen.to(u.deg).value
        am = calc_airmass(zd_deg=zd, scale=scale)
        altaz["alt"].append(alt)
        altaz["az"].append(az)
        altaz["zd"].append(zd)
        altaz["airmass"].append(am)

    am_simpson = (altaz["airmass"][0]
                  + 4 * altaz["airmass"][1]
                  + altaz["airmass"][2]) / 6

    if full:
        return am_simpson, altaz

    return am_simpson


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB):
    return 5 / 10**(gain_dB / 20)


def epadu2dB(gain_epadu):
    return 20 * np.log10(5 / gain_epadu)
