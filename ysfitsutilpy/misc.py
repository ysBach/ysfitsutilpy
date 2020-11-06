'''
Simple mathematical functions that will be used throughout this package. Some
might be useful outside of this package.
'''
import glob
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

__all__ = ["MEDCOMB_KEYS_INT", "SUMCOMB_KEYS_INT", "MEDCOMB_KEYS_FLT32", "LACOSMIC_KEYS",
           "get_size",
           "load_ccd",
           "_parse_data_header", "_parse_extension", "_parse_image",
           "str_now", "change_to_quantity", "binning", "fitsxy2py", "give_stats", "chk_keyval"]


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


def _parse_data_header(ccd_like_object):
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


def _parse_extension(*args, ext=None, extname=None, extver=None):
    """
    Open the input file, return the `HDUList` and the extension.

    This supports several different styles of extension selection.  See the :func:`getdata()`
    documentation for the different possibilities.

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


def _parse_image(im, extension, name, force_ccd=False, prefer_ccd=False):
    '''Parse and return input image as desired format (ndarray or CCDData)
    Parameters
    ----------
    im : CCDdata, ndarray, path-like, or number-like
        The "image" that will be parsed. A string that can be converted to float (``float(im)``)
        will be interpreted as numbers; if not, it will be interpreted as a path to the FITS file.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    force_ccd: bool, optional.
        To force the retun im as `~astropy.nddata.CCDData` object. This is useful when error
        calculation is turned on.

    prefer_ccd: bool, optional.
        Mildly use `~astropy.nddata.CCDData`, i.e., return `~astropy.nddata.CCDData` only if ``im`` was
        `~astropy.nddata.CCDData` or Path-like to a FITS file.

    Returns
    -------
    new_im : ndarray or CCDData
        Depending on the options ``force_ccd`` and ``prefer_ccd``.

    imname : str
        The name of the image.

    imtype : str
        The type of the image.
    '''
    has_no_name = name is None
    extension = _parse_extension(extension)
    if extension is None:
        extstr = ''
    else:
        if isinstance(extension, (tuple, list)):
            extstr = f"[{extension[0]}, {extension[1]}]"
        else:
            extstr = f"[{extension}]"

    if isinstance(im, CCDData):
        # force_ccd: CCDData // prefer_ccd: CCDData // else: ndarray
        imname = f"User-provided CCDData{extstr}" if has_no_name else name
        new_im = im if (force_ccd or prefer_ccd) else im.data
        imtype = "CCDdata"
    elif isinstance(im, np.ndarray):
        # force_ccd: CCDData // prefer_ccd: ndarray // else: ndarray
        imname = "User-provided ndarray" if has_no_name else name
        new_im = CCDData(data=im, unit='adu') if force_ccd else im
        imtype = "ndarray"
    else:
        try:  # IF number (ex: im = 1.3)
            # force_ccd: CCDData // prefer_ccd: number // else: number
            imname = f"User-provided number {im}" if has_no_name else name
            _im = float(im)
            new_im = CCDData(data=_im, unit='adu') if force_ccd else _im
            imtype = "num"
        except ValueError:
            try:  # IF path-like
                # force_ccd: CCDData // prefer_ccd: CCDData // else: ndarray
                fpath = Path(im)
                imname = f"{str(fpath)}{extstr}" if has_no_name else name
                new_im = load_ccd(fpath, extension, ccddata=(force_ccd or prefer_ccd))
                imtype = "path"
            except TypeError:
                raise TypeError("im1/im2 must be CCDData, ndarray, path-like (to FITS), or a number.")

    return new_im, imname, imtype


def load_ccd(path, extension=None, ccddata=True, as_ccd=True, use_wcs=True, unit=None,
             extension_uncertainty="UNCERT", extension_mask='MASK', extension_flags=None,
             load_primary_only_fitsio=True, key_uncertainty_type='UTYPE', memmap=False, **kwd):
    ''' Loads FITS file of CCD image data (not table, etc).
    Paramters
    ---------
    path : path-like
        The path to the FITS file to load.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    ccddata : bool, optional.
        Whether to return `~astropy.nddata.CCDData`. Default is `True`. If it is `False`, **all the
        arguments below are ignored**, except for the keyword arguments that will be passed to
        ``fitsio.read``, and an ndarray will be returned without astropy unit.

    as_ccd : bool, optional.
        Deprecated. (identical to ``ccddata``)

    use_wcs : bool, optional.
        Whether to load WCS by ``fits.getheader``, **not** by `~astropy.nddata.fits_ccdddata_reader`.
        This is necessary as of now because TPV WCS is not properly understood by the latter.
        Default is `True`.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a unit for the image in the
        FITS header (the keyword ``BUNIT`` is used as the unit, if present), this argument is used for
        the unit.
        Default is `None`.

        .. note::
            The behavior differs from astropy's original fits_ccddata_reader: If no ``BUNIT`` is found
            and ``unit`` is `None`, ADU is assumed.

    load_primary_only_fitsio : bool, optional.
        Whether to ignore uncertainty, mask, and flags extensions when using fitsio (i.e., when
        ``use_ccd=False``). This is `True` by default, because that's the most common usage for fitsio.

    extension_uncertainty : str or None, optional
        FITS extension from which the uncertainty should be initialized. If the extension does not
        exist the uncertainty is `None`. Name is changed from ``hdu_uncertainty`` in ccdproc to
        ``extension_uncertainty`` here. See explanation of ``extension``.
        Default is ``'UNCERT'``.

    extension_mask : str or None, optional
        FITS extension from which the mask should be initialized. If the extension does not exist the
        mask is `None`. Name is changed from ``hdu_mask`` in ccdproc to ``extension_mask`` here.  See
        explanation of ``extension``.
        Default is ``'MASK'``.

    hdu_flags : str or None, optional
        Currently not implemented.N ame is changed from ``hdu_flags`` in ccdproc to ``extension_flags``
        here.
        Default is `None`.

    key_uncertainty_type : str, optional
        The header key name where the class name of the uncertainty is stored in the hdu of the
        uncertainty (if any).
        Default is ``UTYPE``.

        ..warning::
            If ``ccddata=False`` and ``load_primary_only_fitsio=False``, the uncertainty type by
            ``key_uncertainty_type`` will be completely ignored.

    memmap : bool, optional
        Is memory mapping to be used? This value is obtained from the configuration item
        ``astropy.io.fits.Conf.use_memmap``.
        Default is `False` (opposite of astropy).

    kwd :
        Any additional keyword parameters that will be used in `~astropy.nddata.fits_ccddata_reader`
        (if ``ccddata=True``) or ``fitsio.read()`` (if ``ccddata=False``).

    Returns
    -------
    CCDData (``ccddata=True``) or ndarray (``ccddata=False``). For the latter case, if
    ``load_primary_only_fitsio=False``, the uncertainty and mask extensions, as well as flags (not
    supported, so just `None`) will be returned as well as the one specified by ``extension``.

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

    Using fitsio, we get ~ 6-100 times faster loading time for FITS files on MBP 15" [2018, macOS
    10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)].
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

    extension = _parse_extension(extension)
    extension_unc = _parse_extension(extension_uncertainty)
    extension_mask = _parse_extension(extension_mask)
    extension_flag = _parse_extension(extension_flags)

    if ccddata and as_ccd:  # if at least one of these is False, it uses fitsio.
        reader_kw = dict(hdu=extension, hdu_uncertainty=extension_unc, hdu_mask=extension_mask,
                         hdu_flags=extension_flag, key_uncertainty_type=key_uncertainty_type, memmap=memmap,
                         **kwd)

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
        def _read_by_fitsio(_hdul, _path, _extension):
            try:
                if isinstance(_extension, (list, tuple, np.ndarray)):
                    # length == 2 is already checked in _parse_extension.
                    arr = hdul[_extension[0], _extension[1]].read()
                else:
                    arr = hdul[_extension].read()
                return arr
            except OSError:
                raise ValueError(f"Extension `{_extension}` is not found (file: {_path})")

        hdul = fitsio.FITS(path)
        if load_primary_only_fitsio:
            data = _read_by_fitsio(hdul, extension)
            hdul.close()
            return data
        else:
            data = _read_by_fitsio(hdul, extension)
            unc = _read_by_fitsio(hdul, extension_unc)
            mask = _read_by_fitsio(hdul, extension_mask)
            flag = None  # FIXME: add this line when CCDData starts to support flags.
            hdul.close()
            return data, unc, mask, flag


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
        The reference time. If not `None`, delta time is calculated.
    dt_fmt : str, optional.
        The Python 3 format string to format the delta time.
    return_time : bool, optional.
        Whether to return the time at the start of this function and the
        delta time (``dt``), as well as the time information string. If
        ``t_ref`` is `None`, ``dt`` is automatically set to `None`.
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
        The input to be changed to a Quantity. If a Quantity is given, ``x`` is changed to the
        ``desired``, i.e., ``x.to(desired)``.

    desired : str or astropy Unit
        The desired unit for ``x``.

    to_value : bool, optional.
        Whether to return as scalar value. If `True`, just the value(s) of the ``desired`` unit will be
        returned after conversion.

    Return
    ------
    ux: Quantity

    Note
    ----
    If Quantity, transform to ``desired``. If ``desired = None``, return it as is. If not Quantity,
    multiply the ``desired``. ``desired = None``, return ``x`` with dimensionless unscaled unit.
    '''
    def _copy(xx):
        try:
            xcopy = xx.copy()
        except AttributeError:
            import copy
            xcopy = copy.deepcopy(xx)
        return xcopy

    if x is None:
        return None

    try:
        ux = x.to(desired)
        if to_value:
            ux = ux.value
    except AttributeError:  # if not Quantity
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
        raise ValueError("If you use astropy.Quantity, you should use unit convertible to `desired`. \n"
                         + f'You gave "{x.unit}", unconvertible with "{desired}".')

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
        The function to be applied for binning, such as ``np.sum``, ``np.mean``, and ``np.median``.

    trim_end: bool
        Whether to trim the end of x, y axes such that binning is done without error.

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
    Tested on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4),
    Radeon Pro 560X (4GB)]
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
        The section specified by FITS convention, i.e., bracket embraced, comma separated, XY order,
        1-indexing, and including the end index.

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


# TODO: add sigma-clipped statistics option (hdr key can be using "SIGC", e.g., SIGCAVG.)
def give_stats(item, extension=None, percentiles=[1, 99], N_extrema=None, return_header=False, nanfunc=False):
    ''' Calculates simple statistics.

    Parameters
    ----------
    item: array-like, CCDData, HDUList, PrimaryHDU, ImageHDU, or path-like
        The data or path to a FITS file to be analyzed.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    percentiles: list-like, optional
        The percentiles to be calculated.

    N_extrema: int, optinoal
        The number of low and high elements to be returned when the whole data are sorted. If `None`,
        it will not be calculated. If ``1``, it is identical to min/max values.

    return_header : bool, optional.
        Works only if you gave ``item`` as FITS file path or ``CCDData``. The statistics information
        will be added to the header and the updated header will be returned.

    nanfunc : bool, optional.
        Whether to use nan-related functions (e.g., ``np.nanmedian``). If any pixel has non-finite
        value (such as ``np.nan`` or ``np.inf``), ``nanfunc`` must be `True` to get proper statistics
        at a cost of computational speed.

    Return
    ------
    result : dict
        The dict which contains all the statistics.

    hdr : Header
        The updated header. Returned only if ``update_header`` is `True` and ``item`` is FITS file path
        or has ``header`` attribute (e.g., ``CCDData`` or ``hdu``)

    Note
    ----
    If you have bottleneck package, the functions from bottleneck will be used. Otherwise, numpy is
    used.

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
    try:  # if Path-like, replace ``item`` to ndarray or CCDData
        fpath = Path(item)
        if return_header:
            item = CCDData.read(fpath, extension)
        else:
            item = fitsio.FITS(fpath)[extension].read()
    except (TypeError, ValueError):
        pass

    data, hdr = _parse_data_header(item)

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
            warn(f"Extrema overlaps (2*N_extrema ({2*N_extrema}) > N_pix ({result['num']}))")
        data_flatten = np.sort(data, axis=None)  # axis=None will do flatten.
        d_los = data_flatten[:N_extrema]
        d_his = data_flatten[-1*N_extrema:]
        result["ext_lo"] = d_los
        result["ext_hi"] = d_his

    if return_header and hdr is not None:
        hdr["STATNPIX"] = (result['num'], "Number of pixels used in statistics below")
        hdr["STATMIN"] = (result['min'], "Minimum value of the pixels")
        hdr["STATMAX"] = (result['max'], "Maximum value of the pixels")
        hdr["STATAVG"] = (result['avg'], "Average value of the pixels")
        hdr["STATMED"] = (result['med'], "Median value of the pixels")
        hdr["STATSTD"] = (result['std'], "Sample standard deviation value of the pixels")
        hdr["STATMED"] = (result['zmin'], "Median value of the pixels")
        hdr["STATZMIN"] = (result['zmin'], "zscale minimum value of the pixels")
        hdr["STATZMAX"] = (result['zmax'], "zscale minimum value of the pixels")
        for i, p in enumerate(percentiles):
            hdr[f"PERCTS{i+1:02d}"] = (p, "The percentile used in STATPCii")
            hdr[f"STATPC{i+1:02d}"] = (result['pct'][i], "Percentile value at PERCTSii")

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
        The header keyword which will be used to make groups for the CCDs that have selected from
        ``type_key`` and ``type_val``. If `None` (default), no grouping will occur, but it will return
        the `~pandas.DataFrameGroupBy` object will be returned for the sake of consistency.

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
        warn(f"{overlap} appear in both type_key and group_key. It may not be harmful but better to avoid.")

    return type_key, type_val, group_key
