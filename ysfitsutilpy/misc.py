'''
Simple mathematical functions that will be used throughout this package. Some
might be useful outside of this package.
'''
import glob
from os import error
import sys
from pathlib import Path, PosixPath, WindowsPath
from warnings import warn

import ccdproc
import fitsio
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS

__all__ = ["MEDCOMB_KEYS_INT", "SUMCOMB_KEYS_INT", "MEDCOMB_KEYS_FLT32",
           "LACOSMIC_KEYS", "get_size",
           "datahdr_parse", "_getext", "load_ccd", "str_now", "change_to_quantity",
           "binning", "fitsxy2py", "give_stats",
           "chk_keyval"]


MEDCOMB_KEYS_INT = dict(dtype='int16',
                        combine_method='median',
                        reject_method=None,
                        unit=u.adu,
                        combine_uncertainty_function=None)

SUMCOMB_KEYS_INT = dict(dtype='int16',
                        combine_method='sum',
                        reject_method=None,
                        unit=u.adu,
                        combine_uncertainty_function=None)

MEDCOMB_KEYS_FLT32 = dict(dtype='float32',
                          combine_method='median',
                          reject_method=None,
                          unit=u.adu,
                          combine_uncertainty_function=None)

# I skipped two params in IRAF LACOSMIC: gain=2.0, readnoise=6.
LACOSMIC_KEYS = {'sigclip': 4.5,
                 'sigfrac': 0.5,
                 'objlim': 1.0,
                 'satlevel': np.inf,
                 'pssl': 0.0,
                 'niter': 4,
                 'sepmed': False,
                 'cleantype': 'medmask',
                 'fsmode': 'median',
                 'psfmodel': 'gauss',
                 'psffwhm': 2.5,
                 'psfsize': 7,
                 'psfk': None,
                 'psfbeta': 4.765}


def inputs2list(inputs, sort=True, error_if_ccd=False):
    ''' Convert glob pattern or list-like of path-like to list of Path

    Parameters
    ----------
    inputs : str, path-like, CCDData
    '''
    contains_ccddata = False
    if isinstance(inputs, str):
        # If str, "dir/file.fits" --> [Path("dir/file.fits")]
        #         "dir/*.fits"    --> [Path("dir/file.fits"), ...]
        outlist = glob.glob(inputs)
    elif isinstance(inputs, (PosixPath, WindowsPath)):
        # If Path, ``TOP/"file*.fits"`` --> [Path("top/file1.fits"), ...]
        outlist = glob.glob(str(inputs))
    elif isinstance(inputs, CCDData):
        if error_if_ccd:
            raise TypeError("CCDData is given as `inputs`. Turn off error_if_ccd or use path-like.")
        else:
            outlist = [inputs]
    else:
        outlist = []
        for i, item in enumerate(inputs):
            if isinstance(item, CCDData):
                contains_ccddata = True
                if error_if_ccd:
                    raise TypeError(
                        f"CCDData is given in the {i}-th element. Turn off error_if_ccd or use path-like."
                    )
                else:
                    outlist.append(item)
            else:  # assume it is path-like
                outlist.append(Path(item))

    if sort and not contains_ccddata:
        outlist.sort()

    return outlist


def get_size(obj, seen=None):
    """Recursively finds size of objects.
    Directly from
    https://goshippo.com/blog/measure-real-size-any-python-object/
    Returns the size in bytes.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif (hasattr(obj, '__iter__')
          and not isinstance(obj, (str, bytes, bytearray))):
        size += sum([get_size(i, seen) for i in obj])
    return size


def datahdr_parse(ccd_like_object):
    if isinstance(ccd_like_object, (CCDData, fits.PrimaryHDU, fits.ImageHDU)):
        data = ccd_like_object.data.copy()
        hdr = ccd_like_object.header.copy()
    elif isinstance(ccd_like_object, fits.HDUList):
        data = ccd_like_object[0].data.copy()
        hdr = ccd_like_object[0].header.copy()
    else:
        data = ccd_like_object.copy()
        hdr = None
    return data, hdr


def _getext(*args, ext=None, extname=None, extver=None):
    """
    Open the input file, return the `HDUList` and the extension.

    This supports several different styles of extension selection.  See the
    :func:`getdata()` documentation for the different possibilities.

    Direct copy from astropy, but removing "opening HDUList" part
    https://github.com/astropy/astropy/blob/master/astropy/io/fits/convenience.py#L988

    This is essential for fits_ccddata_reader, because it only has ``hdu``, not all three of ext,
    extname, and extver (facepalm).
    """

    err_msg = ('Redundant/conflicting extension arguments(s): {}'.format(
        {'args': args, 'ext': ext, 'extname': extname, 'extver': extver})
    )

    # This code would be much simpler if just one way of specifying an extension were picked.  But now
    # we need to support all possible ways for the time being.
    if len(args) == 1:
        # Must be either an extension number, an extension name, or an (extname, extver) tuple
        if isinstance(args[0], (int, np.integer)) or (isinstance(ext, tuple) and len(ext) == 2):
            if ext is not None or extname is not None or extver is not None:
                raise TypeError(err_msg)
            ext = args[0]
        elif isinstance(args[0], str):
            # The first arg is an extension name; it could still be valid to provide an extver kwarg
            if ext is not None or extname is not None:
                raise TypeError(err_msg)
            extname = args[0]
        else:
            # Take whatever we have as the ext argument; we'll validate it below
            ext = args[0]
    elif len(args) == 2:
        # Must be an extname and extver
        if ext is not None or extname is not None or extver is not None:
            raise TypeError(err_msg)
        extname = args[0]
        extver = args[1]
    elif len(args) > 2:
        raise TypeError('Too many positional arguments.')

    if (ext is not None and
            not (isinstance(ext, (int, np.integer)) or
                 (isinstance(ext, tuple) and len(ext) == 2 and
                  isinstance(ext[0], str) and isinstance(ext[1], (int, np.integer))))):
        raise ValueError(
            'The ext keyword must be either an extension number (zero-indexed) or a (extname, extver) tuple.'
        )
    if extname is not None and not isinstance(extname, str):
        raise ValueError('The extname argument must be a string.')
    if extver is not None and not isinstance(extver, (int, np.integer)):
        raise ValueError('The extver argument must be an integer.')

    if ext is None and extname is None and extver is None:
        ext = 0
    elif ext is not None and (extname is not None or extver is not None):
        raise TypeError(err_msg)
    elif extname:
        if extver:
            ext = (extname, extver)
        else:
            ext = (extname, 1)
    elif extver and extname is None:
        raise TypeError('extver alone cannot specify an extension.')

    return ext


def load_ccd(path, *args, ext=None, extname=None, extver=None, as_ccd=True, use_wcs=True, unit=None,
             hdu_uncertainty="UNCERT", hdu_mask='MASK', hdu_flags=None, key_uncertainty_type='UTYPE',
             memmap=False, **kwd):
    ''' Loads FITS file of CCD image data (not table, etc).
    Paramters
    ---------
    path : path-like
        The path to the FITS file to load.

    ext
        The rest of the arguments are for extension specification.
        They are flexible and are best illustrated by examples.

        No extra arguments implies the primary extension::

            load_ccd('in.fits')

        By extension number::

            load_ccd('in.fits', 0)      # the primary extension
            load_ccd('in.fits', 2)      # the second extension
            load_ccd('in.fits', ext=2)  # the second extension

        By name, i.e., ``EXTNAME`` value (if unique)::

            load_ccd('in.fits', 'sci')
            load_ccd('in.fits', extname='sci')  # equivalent

        Note ``EXTNAME`` values are not case sensitive

        By combination of ``EXTNAME`` and EXTVER`` as separate
        arguments or as a tuple::

            load_ccd('in.fits', 'sci', 2)  # EXTNAME='SCI' & EXTVER=2
            load_ccd('in.fits', extname='sci', extver=2)  # equivalent
            load_ccd('in.fits', ('sci', 2))  # equivalent

        Ambiguous or conflicting specifications will raise an exception::

            load_ccd('in.fits', ext=('sci',1), extname='err', extver=2)

    ext : int
        The extension index (0-indexing).

    extname : str
        The extension name (``XTENSION``).

    extver : int
        The version of the extension; used only if extname is given.

    as_ccd : bool, optional.
        Whether to return `~astropy.nddata.CCDData`. Default is `True`. If it is `False`, **all the
        arguments below are ignored**, except for the keyword arguments that will be passed to
        ``fitsio.read``.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a unit for the image in the
        FITS header (the keyword ``BUNIT`` is used as the unit, if present), this argument is used for
        the unit.
        Default is ``None``.

        .. note::
            The behavior differs from astropy's original fits_ccddata_reader: If no ``BUNIT`` is found
            and ``unit`` is `None`, ADU is assumed.

    hdu_uncertainty : str or None, optional
        FITS extension from which the uncertainty should be initialized. If the extension does not
        exist the uncertainty of the CCDData is ``None``.
        Default is ``'UNCERT'``.

    hdu_mask : str or None, optional
        FITS extension from which the mask should be initialized. If the extension does not exist the
        mask of the CCDData is ``None``.
        Default is ``'MASK'``.

    hdu_flags : str or None, optional
        Currently not implemented.
        Default is ``None``.

    key_uncertainty_type : str, optional
        The header key name where the class name of the uncertainty  is stored in the hdu of the
        uncertainty (if any).
        Default is ``UTYPE``.

    memmap : bool, optional
        Is memory mapping to be used? This value is obtained from the
        configuration item ``astropy.io.fits.Conf.use_memmap``.
        Default is `False` (opposite of astropy).

    kwd :
        Any additional keyword parameters that will be used in fits_ccddata_reader (if ``as_ccd=True``)
        or ``fitsio.read()`` (if ``as_ccd=False``).

    Notes
    -----
    Many of the parameter explanations adopted from astropy
    (https://github.com/astropy/astropy/blob/master/astropy/nddata/ccddata.py#L527 and
    https://github.com/astropy/astropy/blob/master/astropy/io/fits/convenience.py#L120).

    CCDData.read cannot read TPV WCS:
    https://github.com/astropy/astropy/issues/7650
    Also memory map must be set False to avoid memory problem
    https://github.com/astropy/astropy/issues/9096
    Plus, WCS info from astrometry.net solve-field sometimes not understood by CCDData.read....
    2020-05-31 16:39:51 (KST: GMT+09:00) ysBach
    Why the name of the argument is different (``hdu``) in fits_ccddata_reader...;;

    Using fitsio, we get ~ 6-100 times faster loading time for FITS files.
    Thus, when you just need data without header information (combine or stacking images, simple image
    arithmetics without header updates, etc) for MANY images, the gain is enormous by using FITSIO.
    This also boosts the speed of some processes which have to open the same FITS file repeatedly due
    to the memory limit.

    ```
        !fitsinfo test.fits
        Filename: test.fits
        No.    Name      Ver    Type      Cards   Dimensions   Format
          0  PRIMARY       1 PrimaryHDU       6   (1,)   int64
          1  a             1 ImageHDU         7   (1,)   int64
          2  a             1 ImageHDU         7   (1,)   int64
          3  a             2 ImageHDU         8   (1,)   int64

        %timeit fitsio.FITS("test.fits")["a", 2].read()
        %timeit fitsio.FITS("test.fits")[0].read()
        118 µs +/- 564 ns per loop (mean +/- std. dev. of 7 runs, 10000 loops each)
        117 µs +/- 944 ns per loop (mean +/- std. dev. of 7 runs, 10000 loops each)

        %timeit CCDData.read("test.fits")
        %timeit CCDData.read("test.fits", hdu=("a", 2), unit='adu')
        10.7 ms +/- 113 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
        11 ms +/- 114 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    ```
    For a 1k by 1k image, it's ~ 6 times faster
    ```
        np.random.seed(123)
        ccd = CCDData(data=np.random.normal(size=(1000, 1000)).astype('float32'), unit='adu')
        ccd.write("test1k_32bit.fits")
        %timeit fitsio.FITS("test10k_32bit.fits")[0].read()
        1.49 ms +/- 91.1 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
        %timeit CCDData.read("test10k_32bit.fits")
        8.9 ms +/- 97.6 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    ```
    For a 10k by 10k image, it's still ~ 6 times faster
    ```
        ccd = CCDData(data=np.random.normal(size=(10000, 10000)).astype('float32'), unit='adu')
        %timeit fitsio.FITS("test10k_32bit.fits")[0].read()
        1.4 ms +/- 123 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
        %timeit CCDData.read("test10k_32bit.fits")
        9.42 ms +/- 391 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    ```
    '''
    ext = _getext(*args, ext=ext, extname=extname, extver=extver)

    if as_ccd:
        reader_kw = dict(hdu=ext, hdu_uncertainty=hdu_uncertainty, hdu_mask=hdu_mask, hdu_flags=hdu_flags,
                         key_uncertainty_type=key_uncertainty_type, memmap=memmap, **kwd)

        # FIXME: Remove this if block in the future if WCS issue is resolved.
        if use_wcs:  # Because of the TPV WCS issue
            hdr = fits.getheader(path)
            reader_kw["wcs"] = WCS(hdr)
            del hdr

        try:
            ccd = CCDData.read(path, unit=unit, **reader_kw)
        except ValueError:  # e.g., user did not give unit and there's no BUNIT
            ccd = CCDData.read(path, unit='adu', **reader_kw)
        # if prefer_bunit:
        #     try:  # Try with no unit to CCDData
        #         ccd = CCDData.read(path, unit=None, **reader_kw)
        #     except ValueError:  # Try with user-given unit to CCDData
        #         ccd = CCDData.read(path, unit=unit, **reader_kw)
        # else:  # prefer user's input
        #     ccd = CCDData.read(path, unit=unit, **reader_kw)

        return ccd

    else:
        # Use fitsio and only load the data as soon as possible. This is much quicker than astropy's getdata
        try:
            if isinstance(ext, (list, tuple, np.ndarray)):
                return fitsio.FITS(path)[ext[0], ext[1]].read()
            else:
                return fitsio.FITS(path)[ext].read()
        except OSError:
            raise ValueError(f"Extension `{ext}` is not found (file: {path})")


def str_now(precision=3, fmt="{:.>72s}", t_ref=None,
            dt_fmt="(dt = {:.3f} s)", return_time=False):
    ''' Get stringfied time now in UT ISOT format.
    Parameters
    ----------
    precision : int, optional.
        The precision of the isot format time.
    fmt : str, optional.
        The Python 3 format string to format the time.
        Examples:
          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in parentheses
            ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with ``_``.
    t_ref : Time, optional.
        The reference time. If not ``None``, delta time is calculated.
    dt_fmt : str, optional.
        The Python 3 format string to format the delta time.
    return_time : bool, optional.
        Whether to return the time at the start of this function and the
        delta time (``dt``), as well as the time information string. If
        ``t_ref`` is ``None``, ``dt`` is automatically set to ``None``.
    '''
    now = Time(Time.now(), precision=precision)
    timestr = now.isot
    if t_ref is not None:
        dt = (now - Time(t_ref)).sec  # float in seconds unit
        timestr = dt_fmt.format(dt) + " " + timestr
    else:
        dt = None

    if return_time:
        return fmt.format(timestr), now, dt
    else:
        return fmt.format(timestr)


def change_to_quantity(x, desired='', to_value=False):
    ''' Change the non-Quantity object to astropy Quantity.
    Parameters
    ----------
    x : object changable to astropy Quantity
        The input to be changed to a Quantity. If a Quantity is given,
        ``x`` is changed to the ``desired``, i.e., ``x.to(desired)``.
    desired : str or astropy Unit
        The desired unit for ``x``.
    to_value : bool, optional.
        Whether to return as scalar value. If ``True``, just the
        value(s) of the ``desired`` unit will be returned after
        conversion.

    Return
    ------
    ux: Quantity

    Note
    ----
    If Quantity, transform to ``desired``. If ``desired = None``, return
    it as is. If not Quantity, multiply the ``desired``. ``desired =
    None``, return ``x`` with dimensionless unscaled unit.
    '''
    def _copy(xx):
        try:
            xcopy = xx.copy()
        except AttributeError:
            import copy
            xcopy = copy.deepcopy(xx)
        return xcopy

    try:
        ux = x.to(desired)
        if to_value:
            ux = ux.value
    except AttributeError:
        if not to_value:
            if isinstance(desired, str):
                desired = u.Unit(desired)
            try:
                ux = x*desired
            except TypeError:
                ux = _copy(x)
        else:
            ux = _copy(x)
    except TypeError:
        ux = _copy(x)
    except u.UnitConversionError:
        raise ValueError("If you use astropy.Quantity, you should use "
                         + "unit convertible to `desired`. \nYou gave "
                         + f'"{x.unit}", unconvertible with "{desired}".')

    return ux


def binning(arr, factor_x=1, factor_y=1, binfunc=np.mean, trim_end=False):
    ''' Bins the given arr frame.
    Paramters
    ---------
    arr: 2d array
        The array to be binned
    factor_x, factor_y: int
        The binning factors in x, y direction.
    binfunc : funciton object
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.
    trim_end: bool
        Whether to trim the end of x, y axes such that binning is done without
        error.

    Note
    ----
    This is ~ 20-30 to upto 10^5 times faster than astropy.nddata's
    block_reduce:
    >>> from astropy.nddata import block_reduce
    >>> import ysfitsutilpy as yfu
    >>> from astropy.nddata import CCDData
    >>> import numpy as np
    >>> ccd = CCDData(data=np.arange(1000).reshape(20, 50), unit='adu')
    >>> kw = dict(factor_x=5, factor_y=5, binfunc=np.sum, trim_end=True)
    >>> %timeit yfu.binning(ccd.data, **kw)
    >>> # 10.9 +- 0.216 us (7 runs, 100000 loops each)
    >>> %timeit yfu.bin_ccd(ccd, **kw, update_header=False)
    >>> # 32.9 µs +- 878 ns per loop (7 runs, 10000 loops each)
    >>> %timeit -r 1 -n 1 block_reduce(ccd, block_size=5)
    >>> # 518 ms, 2.13 ms, 250 us, 252 us, 257 us, 267 us
    >>> # 5.e+5   ...      ...     ...     ...     27  -- times slower
    >>> # some strange chaching happens?
    Tested on MBP 15" 2018, macOS 10.14.6, 2.6 GHz i7
    '''
    binned = arr.copy()
    if trim_end:
        ny_orig, nx_orig = binned.shape
        iy_max = ny_orig - (ny_orig % factor_y)
        ix_max = nx_orig - (nx_orig % factor_x)
        binned = binned[:iy_max, :ix_max]
    ny, nx = binned.shape
    nby = ny // factor_y
    nbx = nx // factor_x
    binned = binned.reshape(nby, factor_y, nbx, factor_x)
    binned = binfunc(binned, axis=(-1, 1))
    return binned


def fitsxy2py(fits_section):
    ''' Given FITS section in str, returns the slices in python convention.

    Parameters
    ----------
    fits_section : str or list-like of such
        The section specified by FITS convention, i.e., bracket
        embraced, comma separated, XY order, 1-indexing, and including
        the end index.

    Note
    ----
    >>> np.eye(5)[fitsxy2py('[1:2,:]')]
    # array([[1., 0.],
    #       [0., 1.],
    #       [0., 0.],
    #       [0., 0.],
    #       [0., 0.]])
    '''
    fits_sections = np.atleast_1d(fits_section)
    slicer = ccdproc.utils.slices.slice_from_string
    sl = [slicer(sect, fits_convention=True) for sect in fits_sections]
    if len(sl) == 1:
        return sl[0]
    else:
        return sl


def give_stats(item, extension=0, percentiles=[1, 99], N_extrema=None,
               return_header=False, nanfunc=False):
    ''' Calculates simple statistics.
    Parameters
    ----------
    item: array-like, CCDData, HDUList, PrimaryHDU, ImageHDU, or path-like
        The data or path to a FITS file to be analyzed.
    extension: int, str, optional
        The extension if ``item`` is the path to the FITS file or
        ``HDUList``.
    percentiles: list-like, optional
        The percentiles to be calculated.
    N_extrema: int, optinoal
        The number of low and high elements to be returned when the
        whole data are sorted. If ``None``, it will not be calculated.
        If ``1``, it is identical to min/max values.
    return_header : bool, optional.
        Works only if you gave ``item`` as FITS file path or
        ``CCDData``. The statistics information will be added to the
        header and the updated header will be returned.
    nanfunc : bool, optional.
        Whether to use nan-related functions (e.g., ``np.nanmedian``).

    Return
    ------
    result : dict
        The dict which contains all the statistics.
    hdr : Header
        The updated header. Returned only if ``update_header`` is
        ``True`` and ``item`` is FITS file path or has ``header``
        attribute (e.g., ``CCDData`` or ``hdu``)

    Note
    ----
    If you have bottleneck package, the functions from bottleneck will
    be used. Otherwise, numpy is used.

    Example
    -------
    >>> bias = CCDData.read("bias_bin11.fits")
    >>> dark = CCDData.read("pdark_300s_27C_bin11.fits")
    >>> percentiles = [0.1, 1, 5, 95, 99, 99.9]
    >>> give_stats(bias, percentiles=percentiles, N_extrema=5)
    >>> give_stats(dark, percentiles=percentiles, N_extrema=5)
    Or just simply
    >>> give_stats("bias_bin11.fits", percentiles=percentiles, N_extrema=5)
    To update the header
    >>> ccd = CCDDAta.read("bias_bin11.fits", unit='adu')
    >>> _, hdr = (ccd, N_extrema=10, update_header=True)
    >>> ccd.header = hdr
    To read the stringfied list into python list (e.g., percentiles):
    >>> import json
    >>> percentiles = json.loads(ccd.header['percentiles'])
    '''
    try:
        fpath = Path(item)
        item = CCDData.read(fpath)
    except (TypeError, ValueError):
        pass

    data, hdr = datahdr_parse(item)

    try:
        import bottleneck as bn
        if nanfunc:
            minf = bn.nanmin
            maxf = bn.nanmax
            avgf = bn.nanmean
            medf = bn.nanmedian
            stdf = bn.nanstd
            pctf = np.nanpercentile  # no bn function
        else:
            minf = np.min
            maxf = np.max
            avgf = np.mean
            medf = bn.median  # Still median from bn seems faster!
            stdf = np.std
            pctf = np.percentile
    except ImportError:
        if nanfunc:
            minf = np.nanmin
            maxf = np.nanmax
            avgf = np.nanmean
            medf = np.nanmedian
            stdf = np.nanstd
            pctf = np.nancentile
        else:
            minf = np.min
            maxf = np.max
            avgf = np.mean
            medf = np.median
            stdf = np.std
            pctf = np.percentile

    result = dict(num=np.size(data),
                  min=minf(data),
                  max=maxf(data),
                  avg=avgf(data),
                  med=medf(data),
                  std=stdf(data, ddof=1),
                  percentiles=percentiles,
                  pct=pctf(data, percentiles)
                  )
    # d_pct = np.percentile(data, percentiles)
    # for i, pct in enumerate(percentiles):
    #     result[f"percentile_{round(pct, 4)}"] = d_pct[i]

    zs = ImageNormalize(data, interval=ZScaleInterval())
    d_zmin = zs.vmin
    d_zmax = zs.vmax
    result["zmin"] = d_zmin
    result["zmax"] = d_zmax

    if N_extrema is not None:
        if 2*N_extrema > result['num']:
            warn("There will be extrema overlaps because "
                 + f"2*N_extrema ({2*N_extrema}) > N_pix ({result['num']})")
        data_flatten = np.sort(data, axis=None)  # axis=None will do flatten.
        d_los = data_flatten[:N_extrema]
        d_his = data_flatten[-1*N_extrema:]
        result["ext_lo"] = d_los
        result["ext_hi"] = d_his

    if return_header and hdr is not None:
        hdr["STATNPIX"] = (result['num'],
                           "Number of pixels used in statistics below")
        hdr["STATMIN"] = (result['min'],
                          "Minimum value of the pixels")
        hdr["STATMAX"] = (result['max'],
                          "Maximum value of the pixels")
        hdr["STATAVG"] = (result['avg'],
                          "Average value of the pixels")
        hdr["STATMED"] = (result['med'],
                          "Median value of the pixels")
        hdr["STATSTD"] = (result['std'],
                          "Sample standard deviation value of the pixels")
        hdr["STATMED"] = (result['zmin'],
                          "Median value of the pixels")
        hdr["STATZMIN"] = (result['zmin'],
                           "zscale minimum value of the pixels")
        hdr["STATZMAX"] = (result['zmax'],
                           "zscale minimum value of the pixels")
        for i, p in enumerate(percentiles):
            hdr[f"PERCTS{i+1:02d}"] = (
                percentiles[i],
                "The percentile used in STATPC"
            )
            hdr[f"STATPC{i+1:02d}"] = (
                result['pct'][i],
                "Percentile value at PERCTS")

        if N_extrema is not None:
            if N_extrema > 99:
                warn("N_extrema > 99 may not work properly in header.")
            for i in range(N_extrema):
                hdr[f"STATLO{i+1:02d}"] = (
                    result['ext_lo'][i],
                    f"Lower extreme values (N_extrema={N_extrema})"
                )
                hdr[f"STATHI{i+1:02d}"] = (
                    result['ext_hi'][i],
                    f"Upper extreme values (N_extrema={N_extrema})"
                )
        return result, hdr
    return result


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB):
    return 5 / 10**(gain_dB / 20)


def epadu2dB(gain_epadu):
    return 20 * np.log10(5 / gain_epadu)


def chk_keyval(type_key, type_val, group_key):
    ''' Checks the validity of key and values used heavily in combutil.
    Parameters
    ----------
    type_key : None, str, list of str, optional
        The header keyword for the ccd type you want to use for match.

    type_val : None, int, str, float, etc and list of such
        The header keyword values for the ccd type you want to match.


    group_key : None, str, list of str, optional
        The header keyword which will be used to make groups for the CCDs
        that have selected from ``type_key`` and ``type_val``.
        If ``None`` (default), no grouping will occur, but it will return
        the `~pandas.DataFrameGroupBy` object will be returned for the sake
        of consistency.

    Return
    ------
    type_key, type_val, group_key
    '''
    # Make type_key to list
    if type_key is None:
        type_key = []
    elif isinstance(type_key, str):
        type_key = [type_key]
    else:
        try:
            type_key = list(type_key)
            if not all(isinstance(x, str) for x in type_key):
                raise TypeError("Some of type_key are not str.")
        except TypeError:
            raise TypeError("type_key should be str or convertible to list.")

    # Make type_val to list
    if type_val is None:
        type_val = []
    elif isinstance(type_val, str):
        type_val = [type_val]
    else:
        try:
            type_val = list(type_val)
        except TypeError:
            raise TypeError("type_val should be str or convertible to list.")

    # Make group_key to list
    if group_key is None:
        group_key = []
    elif isinstance(group_key, str):
        group_key = [group_key]
    else:
        try:
            group_key = list(group_key)
            if not all(isinstance(x, str) for x in group_key):
                raise TypeError("Some of group_key are not str.")
        except TypeError:
            raise TypeError("group_key should be str or convertible to list.")

    if len(type_key) != len(type_val):
        raise ValueError("type_key and type_val must have the same length!")

    # If there is overlap
    overlap = set(type_key).intersection(set(group_key))
    if len(overlap) > 0:
        warn(f"{overlap} appear in both type_key and group_key. It may not "
             + "be harmful but better to avoid.")

    return type_key, type_val, group_key
