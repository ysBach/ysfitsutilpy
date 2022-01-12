import glob
import re
import sys
from copy import deepcopy
from pathlib import Path, PosixPath, WindowsPath
from warnings import warn

import bottleneck as bn
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData, Cutout2D
from astropy.stats import mad_std
from astropy.table import Table
from astropy.time import Time
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS, Wcsprm
from ccdproc import trim_image
# from scipy.interpolate import griddata
from scipy.ndimage import label as ndlabel

from .misc import (_image_shape, bezel2slice, binning, change_to_quantity,
                   fitsxy2py, is_list_like, listify, str_now)

try:
    import fitsio
    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False

try:
    import numexpr as ne
    HAS_NE = True
    NEVAL = ne.evaluate  # "n"umerical "eval"uator
    NPSTR = ""
except ImportError:
    HAS_NE = False
    NEVAL = eval  # "n"umerical "eval"uator
    NPSTR = "np."


__all__ = [
    "ASTROPY_CCD_TYPES",
    # ! file io related:
    "get_size", "write2fits",
    # ! parsers:
    "_parse_data_header", "_parse_image", "_has_header", "_parse_extension",
    # ! loaders:
    "load_ccd", "inputs2list",
    # ! setters:
    "CCDData_astype", "set_ccd_attribute", "set_ccd_gain_rdnoise",
    "propagate_ccdmask",
    # ! ccdproc:
    "trim_ccd", "bezel_ccd", "trim_overlap", "cut_ccd", "bin_ccd",
    "fixpix",
    "make_errormap", "errormap",
    "find_extpix", "find_satpix",
    # ! header update:
    "cmt2hdr", "update_tlm", "update_process", "hedit", "key_remover", "key_mapper", "chk_keyval",
    # ! header accessor:
    "get_from_header", "get_if_none",
    # ! WCS related:
    "wcs_crota", "center_radec", "calc_offset_wcs", "calc_offset_physical",
    "wcsremove", "fov_radius",
    # ! math:
    "give_stats"
]

ASTROPY_CCD_TYPES = (CCDData, fits.PrimaryHDU, fits.ImageHDU)  # fits.CompImageHDU ?


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
        objv = obj.values()
        objk = obj.keys()
        for kv in [objk, objv]:
            for v in kv:
                if not (isinstance(v, np.ndarray) and v.ndim == 0):
                    size += get_size(v, seen)
        # size += sum([get_size(v, seen) for v in obj.values()])
        # size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif (hasattr(obj, '__iter__')
          and not isinstance(obj, (str, bytes, bytearray))):
        size += sum([get_size(i, seen) for i in obj])
    return size


def write2fits(data, header, output, return_ccd=False, **kwargs):
    """ A convenience function to write proper FITS file.
    Parameters
    ----------
    data : ndarray
        The data

    header : `~astropy.io.fits.Header`
        The header

    output : path-like
        The output file path

    return_ccd : bool, optional.
        Whether to return the generated CCDData.

    **kwargs :
        The keyword arguements to write FITS file by
        `~astropy.nddata.fits_data_writer`, such as ``output_verify=True``,
        ``overwrite=True``.
    """
    try:
        unit = header['BUNIT']
    except (KeyError, IndexError):
        unit = 'adu'
    ccd = CCDData(data=data, header=header, unit=unit)

    try:
        ccd.write(output, **kwargs)
    except fits.VerifyError:
        print("Try using output_verify='fix' to avoid this error.")
    if return_ccd:
        return ccd


def _parse_data_header(
        ccdlike,
        extension=None,
        parse_data=True,
        parse_header=True,
        copy=True
):
    '''Parses data and header and return them separately after copy.

    Paramters
    ---------
    ccdlike : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like, None
        The object to be parsed into data and header.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    parse_data, parse_header : bool, optional.
        Because this function uses ``.copy()`` for safety, it may take a bit of
        time if this function is used iteratively. One then can turn off one of
        these to ignore either data or header part.

    Returns
    -------
    data : ndarray, None
        The data part of the input `ccdlike`. If `ccdlike` is ``''`` or `None`,
        `None` is returned.

    hdr : Header, None
        The header if header exists; otherwise, `None` is returned.

    Notes
    -----
    _parse_data_header and _parse_image have different purposes:
    _parse_data_header is to get a quick copy of the data and/or header,
    especially to CHECK if it has header, while _parse_image is to deal mainly
    with the data (and has options to return as CCDData).
    '''
    if ccdlike is None or (isinstance(ccdlike, str) and ccdlike == ''):
        data = None
        hdr = None
    elif isinstance(ccdlike, ASTROPY_CCD_TYPES):
        if parse_data:
            data = ccdlike.data.copy() if copy else ccdlike.data
        else:
            data = None
        if parse_header:
            hdr = ccdlike.header.copy() if copy else ccdlike.header
        else:
            hdr = None
    elif isinstance(ccdlike, fits.HDUList):
        extension = _parse_extension(extension) if (parse_data or parse_header) else 0
        # ^ don't even do _parse_extension if both are False
        if parse_data:
            data = ccdlike[extension].data.copy() if copy else ccdlike[extension].data
        else:
            data = None
        if parse_header:
            hdr = ccdlike[extension].header.copy() if copy else ccdlike[extension].header
        else:
            hdr = None
    elif isinstance(ccdlike, (np.ndarray, list, tuple)):
        if parse_data:
            data = np.array(ccdlike, copy=copy)
        else:
            data = None
        hdr = None  # regardless of parse_header
    elif isinstance(ccdlike, fits.Header):
        data = None  # regardless of parse_data
        if parse_header:
            hdr = ccdlike.copy() if copy else ccdlike
        else:
            hdr = None
    elif HAS_FITSIO and isinstance(ccdlike, fitsio.FITSHDR):
        import copy
        data = None  # regardless of parse_data
        if parse_header:
            hdr = copy.deepcopy(ccdlike) if copy else ccdlike
        else:
            hdr = None
    else:
        try:
            data = float(ccdlike) if (parse_data or parse_header) else None
            hdr = None
        except (ValueError, TypeError):  # Path-like
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")``
            # can be PosixPath without error...
            extension = _parse_extension(extension) if parse_data or parse_header else 0
            # fits.getheader is ~ 10-20 times faster than load_ccd.
            # 2020-11-09 16:06:41 (KST: GMT+09:00) ysBach
            try:
                if parse_header:
                    hdu = fits.open(Path(ccdlike), memmap=False)[extension]
                    # No need to copy because they've been read (loaded) for
                    # the first time here.
                    data = hdu.data if parse_data else None
                    hdr = hdu.header if parse_header else None
                else:
                    if isinstance(extension, tuple):
                        if HAS_FITSIO:
                            data = fitsio.read(Path(ccdlike), ext=extension[0],
                                               extver=extension[1])
                        else:
                            data = fits.getdata(Path(ccdlike), *extension)
                    else:
                        if HAS_FITSIO:
                            data = fitsio.read(Path(ccdlike), ext=extension)
                        else:
                            data = fits.getdata(Path(ccdlike), extension)
                    hdr = None
            except TypeError as E:
                raise E(f"ccdlike type ({type(ccdlike)}) is not acceptable "
                        + "to find header and data.")

    return data, hdr


def _parse_image(
        ccdlike,
        extension=None,
        name=None,
        force_ccddata=False,
        prefer_ccddata=False,
        copy=True,
):
    '''Parse and return input image as desired format (ndarray or CCDData)
    Parameters
    ----------
    ccdlike : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
        The "image" that will be parsed. A string that can be converted to
        float (``float(im)``) will be interpreted as numbers; if not, it will
        be interpreted as a path to the FITS file.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    force_ccddata: bool, optional.
        To force the retun im as `~astropy.nddata.CCDData` object. This is
        useful when error calculation is turned on.
        Default is `False`.

    prefer_ccddata: bool, optional.
        Mildly use `~astropy.nddata.CCDData`, i.e., return
        `~astropy.nddata.CCDData` only if `im` was `~astropy.nddata.CCDData`,
        HDU object, or Path-like to a FITS file, but **not** if it was ndarray
        or numbers.
        Default is `False`.

    Returns
    -------
    new_im : ndarray or CCDData
        Depending on the options `force_ccddata` and `prefer_ccddata`.

    imname : str
        The name of the image.

    imtype : str
        The type of the image.

    Notes
    -----
    _parse_data_header and _parse_image have different purposes:
    _parse_data_header is to get a quick copy of the data and/or header,
    especially to CHECK if it has header, while _parse_image is to deal mainly
    with the data (and has options to return as CCDData).
    '''
    def __extract_extension(ext):
        extension = _parse_extension(ext)
        if extension is None:
            extstr = ''
        else:
            if isinstance(extension, (tuple, list)):
                extstr = f"[{extension[0]}, {extension[1]}]"
            else:
                extstr = f"[{extension}]"
        return extension, extstr

    def __extract_from_hdu(hdu, force_ccddata, prefer_ccddata):
        if force_ccddata or prefer_ccddata:
            unit = ccdlike.header.get("BUNIT", default=u.adu)
            if isinstance(unit, str):
                unit = unit.lower()
            if copy:
                return CCDData(data=hdu.data.copy(), header=hdu.header.copy(), unit=unit)
            else:
                return CCDData(data=hdu.data, header=hdu.header, unit=unit)
            # The two lines above took ~ 5 us and 10-30 us for the simplest
            # header and 1x1 pixel data case (regardless of BUNIT exists), on
            # MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
            # GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
        else:
            return hdu.data.copy() if copy else hdu.data

    ccd_kw = dict(force_ccddata=force_ccddata, prefer_ccddata=prefer_ccddata)
    has_no_name = name is None
    extension, extstr = __extract_extension(extension)
    imname = f"User-provided {ccdlike.__class__.__name__}{extstr}" if has_no_name else name

    if isinstance(ccdlike, CCDData):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        if (force_ccddata or prefer_ccddata):
            new_im = ccdlike.copy() if copy else ccdlike
        else:
            new_im = ccdlike.data.copy() if copy else ccdlike.data
        imtype = "CCDdata"
    elif isinstance(ccdlike, (fits.PrimaryHDU, fits.ImageHDU)):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        new_im = __extract_from_hdu(ccdlike, **ccd_kw)
        imtype = "hdu"
    elif isinstance(ccdlike, fits.HDUList):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        new_im = __extract_from_hdu(ccdlike[extension], **ccd_kw)
        imtype = "HDUList"
    elif isinstance(ccdlike, np.ndarray):
        # force_ccddata: CCDData // prefer_ccddata: ndarray // else: ndarray
        if copy:
            new_im = (CCDData(data=ccdlike.copy(), unit='adu') if force_ccddata
                      else ccdlike.copy())
        else:
            new_im = CCDData(data=ccdlike, unit='adu') if force_ccddata else ccdlike
        imtype = "ndarray"
    else:
        try:  # IF number (ex: im = 1.3)
            # force_ccddata: CCDData // prefer_ccddata: array // else: array
            imname = f"{imname} {ccdlike}" if has_no_name else name
            _im = float(ccdlike)
            new_im = CCDData(data=_im, unit='adu') if force_ccddata else np.asarray(_im)
            imtype = "num"
            # imname can be "int", "float", "str", etc, so imtype might be useful.
        except (ValueError, TypeError):
            try:  # IF path-like
                # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
                fpath = Path(ccdlike)
                imname = f"{str(fpath)}{extstr}" if has_no_name else name
                # set redundant extensions to None so that only the part
                # specified by `extension` be loaded:
                new_im = load_ccd(
                    fpath,
                    extension,
                    ccddata=force_ccddata or force_ccddata,
                    extension_uncertainty=None,
                    extension_mask=None
                )
                imtype = "path"
            except TypeError as E:
                raise E(
                    "input must be CCDData-like, ndarray, path-like (to FITS), or a number."
                )

    return new_im, imname, imtype


def _has_header(ccdlike, extension=None, open_if_file=True):
    '''Checks if the object has header; similar to _parse_data_header.

    Paramters
    ---------
    ccdlike : CCDData, PrimaryHDU, ImageHDU, HDUList, ndarray, number-like, path-like
        The object to be parsed into data and header.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used. Used only if `ccdlike` is HDUList or
        path-like.

    open_if_file : bool, optional.
        Whether to open the file to check if it has a header when `ccdlike` is
        path-like. Any FITS file has a header, so this means it will check the
        existence and validity of the file. If set to `False`, all path-like
        input will return `False` because the path itself has no header.

    Notes
    -----
    It first checks if the input is one of ``(CCDData, fits.PrimaryHDU,
    fits.ImageHDU)``, then if `fits.HDUList`, then if `np.ndarray`, then if
    number-like, and then finally if path-like. Although this has a bit of
    disadvantage considering we may use file-path for most of the time, the
    overhead is only ~ 1 us, tested on MBP 15" [2018, macOS 10.14.6, i7-8850H
    (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)].
    '''
    hashdr = True
    if isinstance(ccdlike, ASTROPY_CCD_TYPES):  # extension not used
        try:
            hashdr = ccdlike.header is not None
        except AttributeError:
            hashdr = False
    elif isinstance(ccdlike, fits.HDUList):
        extension = _parse_extension(extension)
        try:
            hashdr = ccdlike[extension].header is not None
        except AttributeError:
            hashdr = False
    elif is_list_like(ccdlike):
        hashdr = False
    else:
        try:  # if number-like
            _ = float(ccdlike)
            hashdr = False
        except (ValueError, TypeError):  # if path-like
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")``
            # can be PosixPath without error...
            if open_if_file:
                try:
                    # fits.getheader is ~ 10-20 times faster than load_ccd.
                    # 2020-11-09 16:06:41 (KST: GMT+09:00) ysBach
                    _ = fits.getheader(Path(ccdlike), extension)
                except (AttributeError, FileNotFoundError):
                    hashdr = False
            else:
                hashdr = False

    return hashdr


def _parse_extension(*args, ext=None, extname=None, extver=None):
    """
    Open the input file, return the `HDUList` and the extension.

    This supports several different styles of extension selection.  See the
    :func:`getdata()` documentation for the different possibilities.

    Direct copy from astropy, but removing "opening HDUList" part
    https://github.com/astropy/astropy/blob/master/astropy/io/fits/convenience.py#L988

    This is essential for fits_ccddata_reader, because it only has `hdu`, not
    all three of ext, extname, and extver.

    Notes
    -----
    %timeit yfu._parse_extension()
    # 1.52 µs +- 69.3 ns per loop (mean +- std. dev. of 7 runs, 1000000 loops each)
    """

    err_msg = ('Redundant/conflicting extension arguments(s): {}'.format(
        {'args': args, 'ext': ext, 'extname': extname, 'extver': extver})
    )

    # This code would be much simpler if just one way of specifying an
    # extension were picked.  But now we need to support all possible ways for
    # the time being.
    if len(args) == 1:
        # Must be either an extension number, an extension name, or an
        # (extname, extver) tuple
        if (isinstance(args[0], (int, np.integer))
                or (isinstance(ext, tuple) and len(ext) == 2)):
            if ext is not None or extname is not None or extver is not None:
                raise TypeError(err_msg)
            ext = args[0]
        elif isinstance(args[0], str):
            # The first arg is an extension name; it could still be valid to
            # provide an extver kwarg
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
            'The ext keyword must be either an extension number (zero-indexed) '
            + 'or a (extname, extver) tuple.'
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


def load_ccd(
        path,
        extension=None,
        fits_section=None,
        ccddata=True,
        as_ccd=True,
        use_wcs=True,
        unit=None,
        extension_uncertainty="UNCERT",
        extension_mask='MASK',
        extension_flags=None,
        full=False,
        key_uncertainty_type='UTYPE',
        memmap=False,
        **kwd
):
    ''' Loads FITS file of CCD image data (not table, etc).

    Paramters
    ---------
    path : path-like
        The path to the FITS file to load.

    fits_section : str, optional.
        Region of `~astropy.nddata.CCDData` from which the data is extracted.
        Default: `None`.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    ccddata : bool, optional.
        Whether to return `~astropy.nddata.CCDData`. Default is `True`. If it
        is `False`, **all the arguments below are ignored**, except for the
        keyword arguments that will be passed to `fitsio.read`, and an ndarray
        will be returned without astropy unit.

    as_ccd : bool, optional.
        Deprecated. (identical to `ccddata`)

    use_wcs : bool, optional.
        Whether to load WCS by `fits.getheader`, **not** by
        `~astropy.nddata.fits_ccdddata_reader`. This is necessary as of now
        because TPV WCS is not properly understood by the latter.
        Default : `True`.
        Used only if ``ccddata=True``.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used as
        the unit, if present), this argument is used for the unit.
        Default: `None`.
        Used only if ``ccddata=True``.

        .. note::
            The behavior differs from astropy's original fits_ccddata_reader:
            If no ``BUNIT`` is found and `unit` is `None`, ADU is assumed.

    full : bool, optional.
        Whether to return full `(data, unc, mask, flag)` when using
        `fitsio` (i.e., when `ccddata=False`). If `False`(default), only `data`
        will be returned.
        Default: `False`.

    extension_uncertainty : str or None, optional
        FITS extension from which the uncertainty should be initialized. If the
        extension does not exist the uncertainty is `None`. Name is changed
        from `hdu_uncertainty` in ccdproc to `extension_uncertainty` here. See
        explanation of `extension`.
        Default: ``'UNCERT'``.

    extension_mask : str or None, optional
        FITS extension from which the mask should be initialized. If the
        extension does not exist the mask is `None`. Name is changed from
        `hdu_mask` in ccdproc to `extension_mask` here.  See explanation of
        `extension`.
        Default: ``'MASK'``.

    hdu_flags : str or None, optional
        Currently not implemented.N ame is changed from `hdu_flags` in ccdproc
        to `extension_flags` here.
        Default: `None`.

    key_uncertainty_type : str, optional
        The header key name where the class name of the uncertainty is stored
        in the hdu of the uncertainty (if any).
        Default: ``UTYPE``.
        Used only if ``ccddata=True``.

        ..warning::
            If ``ccddata=False`` and ``load_primary_only_fitsio=False``, the
            uncertainty type by `key_uncertainty_type` will be completely
            ignored.

    memmap : bool, optional
        Is memory mapping to be used? This value is obtained from the
        configuration item `astropy.io.fits.Conf.use_memmap`.
        Default: `False` (**opposite of astropy**).
        Used only if ``ccddata=True``.

    kwd :
        Any additional keyword parameters that will be used in
        `~astropy.nddata.fits_ccddata_reader` (if ``ccddata=True``) or
        `fitsio.read()` (if ``ccddata=False``).

    Returns
    -------
    CCDData (``ccddata=True``) or ndarray (``ccddata=False``). For the latter
    case, if ``load_primary_only_fitsio=False``, the uncertainty and mask
    extensions, as well as flags (not supported, so just `None`) will be
    returned as well as the one specified by `extension`.

    If ``ccddata=False``, the returned object can be an ndarray (`full_fitsio`
    is `False`) or a tuple of arrays ``(data, unc, mask, flag)`` (`full_fitsio`
    is `True`).

    Notes
    -----
    Many of the parameter explanations adopted from astropy
    (https://github.com/astropy/astropy/blob/master/astropy/nddata/ccddata.py#L527
    and
    https://github.com/astropy/astropy/blob/master/astropy/io/fits/convenience.py#L120).

    CCDData.read cannot read TPV WCS:
    https://github.com/astropy/astropy/issues/7650
    Also memory map must be set False to avoid memory problem
    https://github.com/astropy/astropy/issues/9096
    Plus, WCS info from astrometry.net solve-field sometimes not understood by
    CCDData.read.... 2020-05-31 16:39:51 (KST: GMT+09:00) ysBach
    Why the name of the argument is different (`hdu`) in
    fits_ccddata_reader...;;

    Using fitsio, we get ~ 6-100 times faster loading time for FITS files on
    MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB
    (2400MHz DDR4), Radeon Pro 560X (4GB)]. Thus, when you just need data
    without header information (combine or stacking images, simple image
    arithmetics without header updates, etc) for MANY images, the gain is
    enormous by using FITSIO. This also boosts the speed of some processes
    which have to open the same FITS file repeatedly due to the memory limit.

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
        ccd = CCDData(data=np.random.normal(
            size=(1000, 1000)).astype('float32'), unit='adu'
        )
        ccd.write("test1k_32bit.fits")
        %timeit fitsio.FITS("test10k_32bit.fits")[0].read()
        1.49 ms +/- 91.1 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
        %timeit CCDData.read("test10k_32bit.fits")
        8.9 ms +/- 97.6 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    ```
    For a 10k by 10k image, it's still ~ 6 times faster
    ```
        ccd = CCDData(data=np.random.normal(
            size=(10000, 10000)).astype('float32'), unit='adu'
        )
        %timeit fitsio.FITS("test10k_32bit.fits")[0].read()
        1.4 ms +/- 123 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
        %timeit CCDData.read("test10k_32bit.fits")
        9.42 ms +/- 391 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    ```
    '''
    def ext_umf(ext):
        """ Return None if ext is None, otherwise, parse it (usu. returns 0)
        """
        return None if ext is None else _parse_extension(ext)

    try:
        path = Path(path)
    except TypeError as E:
        raise E(f"You must provide Path-like, not {type(path)}.")

    extension = _parse_extension(extension)

    if HAS_FITSIO:
        if ccddata and as_ccd:  # if at least one of these is False, it uses fitsio.
            extension_uncertainty = ext_umf(extension_uncertainty)
            extension_mask = ext_umf(extension_mask)
            extension_flag = ext_umf(extension_flags)
            # If not None, this happens: NotImplementedError: loading flags is
            # currently not supported.

            reader_kw = dict(
                hdu=extension,
                hdu_uncertainty=extension_uncertainty,
                hdu_mask=extension_mask,
                hdu_flags=extension_flag,
                key_uncertainty_type=key_uncertainty_type,
                memmap=memmap,
                **kwd
            )

            # FIXME: Remove this if block in the future if WCS issue is resolved.
            if use_wcs:  # Because of the TPV WCS issue
                hdr = fits.getheader(path)
                reader_kw["wcs"] = WCS(hdr)
                del hdr

            try:  # Use BUNIT if unit is None
                ccd = CCDData.read(path, unit=unit, **reader_kw)
            except ValueError:  # e.g., user did not give unit and there's no BUNIT
                ccd = CCDData.read(path, unit=u.adu, **reader_kw)

            return ccd if fits_section is None else trim_ccd(ccd, fits_section=fits_section)

        else:
            # Use fitsio and only load the data as soon as possible.
            # This is much quicker than astropy's getdata
            def _read_by_fitsio(_hdul, _path, _extension, _fits_section=None):
                try:
                    if _fits_section is not None:
                        sl = fitsxy2py(_fits_section)
                        if is_list_like(_extension):
                            # length == 2 is already checked in _parse_extension.
                            arr = _hdul[_extension[0], _extension[1]][sl]
                        else:
                            arr = _hdul[_extension][sl]
                    else:
                        if is_list_like(_extension):
                            # length == 2 is already checked in _parse_extension.
                            arr = _hdul[_extension[0], _extension[1]].read()
                        else:
                            arr = _hdul[_extension].read()
                except OSError:
                    raise ValueError(
                        f"Extension `{_extension}` is not found (file: {_path})"
                    )
                except ValueError as E:
                    raise E()

                return arr

            with fitsio.FITS(path) as hdul:
                data = _read_by_fitsio(hdul, path, extension, _fits_section=fits_section)

                if full:
                    try:  # Read uncertainty if exists
                        e_u = _parse_extension(extension_uncertainty)
                        unc = _read_by_fitsio(hdul, path, e_u, _fits_section=fits_section)
                    except (OSError, ValueError):  # if the extension is not found
                        unc = None
                    try:  # Read uncertainty if exists
                        e_m = _parse_extension(extension_mask)
                        mask = _read_by_fitsio(hdul, path, e_m, _fits_section=fits_section)
                    except (OSError, ValueError):  # if the extension is not found
                        mask = None

                    flag = None  # FIXME: add this line when CCDData starts to support flags.
                    return data, unc, mask, flag

                else:
                    return data

    else:
        ignore_u = True if extension_uncertainty is None else False
        ignore_m = True if extension_mask is None else False

        extension_uncertainty = _parse_extension(extension_uncertainty)
        extension_mask = _parse_extension(extension_mask)
        if extension_flags is not None:
            extension_flags = _parse_extension(extension_flags)
        # If not None, this happens:
        #   NotImplementedError: loading flags is currently not supported.

        reader_kw = dict(
            hdu=extension,
            hdu_uncertainty=extension_uncertainty,
            hdu_mask=extension_mask,
            hdu_flags=extension_flag,
            key_uncertainty_type=key_uncertainty_type,
            memmap=memmap,
            **kwd
        )

        # FIXME: Remove this if block in the future if WCS issue is resolved.
        if use_wcs:  # Because of the TPV WCS issue
            hdr = fits.getheader(path)
            reader_kw["wcs"] = WCS(hdr)
            del hdr

        try:
            ccd = CCDData.read(path, unit=unit, **reader_kw)
        except ValueError:  # e.g., user did not give unit and there's no BUNIT
            ccd = CCDData.read(path, unit='adu', **reader_kw)

        # Force them to be None if extension is not specified
        # (astropy.NDData.CCDData forces them to be loaded, which is not desirable imho)
        ccd.uncertainty = None if ignore_u else ccd.uncertainty
        ccd.mask = None if ignore_m else ccd.mask

        if fits_section is not None:
            ccd = trim_ccd(ccd, fits_section=fits_section)

        if ccddata and as_ccd:  # if at least one of these is False, it uses fitsio.
            return ccd
        else:
            if full:
                try:
                    unc = np.array(ccd.uncertainty.array)
                except AttributeError:
                    unc = None
                mask = None if ccd.mask is None else np.array(ccd.mask)
                flag = None  # FIXME: add this line when CCDData starts to support flags.
                return ccd.data, unc, mask, flag
            else:
                return ccd.data


def inputs2list(
        inputs,
        sort=True,
        accept_ccdlike=True,
        path_to_text=False,
        check_coherency=False
):
    ''' Convert glob pattern or list-like of path-like to list of Path

    Parameters
    ----------
    inputs : str, path-like, CCDData, fits.PrimaryHDU, fits.ImageHDU, DataFrame-convertable.
        If DataFrame-convertable, e.g., dict, `~pandas.DataFrame` or
        `~astropy.table.Table`, it must have column named ``"file"``, such that
        ``outlist = list(inputs["file"])`` is possible. Otherwise, please use,
        e.g., ``inputs = list(that_table["filenamecolumn"])``. If a str starts
        with ``"@"`` (e.g., ``"@darks.list"``), it assumes the file contains a
        list of paths separated by ``"\n"``, as in IRAF.

    sort : bool, optional.
        Whether to sort the output list.
        Default: `True`.

    accept_ccdlike: bool, optional.
        Whether to accept `~astropy.nddata.CCDData`-like objects and simpley
        return ``[inputs]``.
        Default: `True`.

    path_to_text: bool, optional.
        Whether to convert the `pathlib.Path` object to `str`.
        Default: `True`.

    check_coherence: bool, optional.
        Whether to check if all elements of the `inputs` have the identical
        type.
        Default: `False`.
    '''
    contains_ccdlike = False
    if inputs is None:
        return None
    elif isinstance(inputs, str):
        if inputs.startswith("@"):
            with open(inputs[1:]) as ff:
                outlist = ff.read().splitlines()
        else:
            # If str, "dir/file.fits" --> [Path("dir/file.fits")]
            #         "dir/*.fits"    --> [Path("dir/file.fits"), ...]
            outlist = glob.glob(inputs)
    elif isinstance(inputs, (PosixPath, WindowsPath)):
        # If Path, ``TOP/"file*.fits"`` --> [Path("top/file1.fits"), ...]
        outlist = glob.glob(str(inputs))
    elif isinstance(inputs, ASTROPY_CCD_TYPES):
        if accept_ccdlike:
            outlist = [inputs]
        else:
            raise TypeError(f"{type(inputs)} is given as `inputs`. "
                            + "Turn off accept_ccdlike or use path-like.")
    elif isinstance(inputs, (Table, dict, pd.DataFrame)):
        # Do this before is_list_like because DataFrame returns True in
        # is_list_like as it is iterable.
        try:
            outlist = list(inputs["file"])
        except KeyError as E:
            raise E("If inputs is DataFrame convertible, it must have column named 'file'.")
    elif is_list_like(inputs):
        type_ref = type(inputs[0])
        outlist = []
        for i, item in enumerate(inputs):
            if check_coherency and (type(item) != type_ref):
                raise TypeError(
                    f"The 0-th item has {type_ref} while {i}-th has {type(item)}."
                )
            if isinstance(item, ASTROPY_CCD_TYPES):
                contains_ccdlike = True
                if accept_ccdlike:
                    outlist.append(item)
                else:
                    raise TypeError(f"{type(item)} is given in the {i}-th element. "
                                    + "Turn off accept_ccdlike or use path-like.")
            else:  # assume it is path-like
                if path_to_text:
                    outlist.append(str(item))
                else:
                    outlist.append(Path(item))
    else:
        raise TypeError(f"inputs type ({type(inputs)})not accepted.")

    if sort and not contains_ccdlike:
        outlist.sort()

    return outlist


def CCDData_astype(ccd, dtype='float32', uncertainty_dtype=None, copy=True):
    ''' Assign dtype to the CCDData object (numpy uses float64 default).

    Parameters
    ----------
    ccd : CCDData
        The ccd to be astyped.

    dtype : dtype-like
        The dtype to be applied to the data

    uncertainty_dtype : dtype-like
        The dtype to be applied to the uncertainty. Be default, use the same
        dtype as data (``uncertainty_dtype=dtype``).

    Example
    -------
    >>> from astropy.nddata import CCDData
    >>> import numpy as np
    >>> ccd = CCDData.read("image_unitygain001.fits", 0)
    >>> ccd.uncertainty = np.sqrt(ccd.data)
    >>> ccd = yfu.CCDData_astype(ccd, dtype='int16', uncertainty_dtype='float32')
    '''
    if copy:
        nccd = ccd.copy()
    else:
        nccd = ccd
    nccd.data = nccd.data.astype(dtype)

    try:
        if uncertainty_dtype is None:
            uncertainty_dtype = dtype
        nccd.uncertainty.array = nccd.uncertainty.array.astype(uncertainty_dtype)
    except AttributeError:
        # If there is no uncertainty attribute in the input `ccd`
        pass

    update_tlm(nccd.header)
    return nccd


def set_ccd_attribute(
        ccd,
        name,
        value=None,
        key=None,
        default=None,
        unit=None,
        header_comment=None,
        update_header=True,
        verbose=True,
        wrapper=None,
        wrapper_kw={}
):
    ''' Set attributes from given paramters.

    Parameters
    ----------
    ccd : CCDData
        The ccd to add attribute.

    value : Any, optional.
        The value to be set as the attribute. If `None`, the
        ``ccd.header[key]`` will be searched.

    name : str, optional.
        The name of the attribute.

    key : str, optional.
        The key in the ``ccd.header`` to be searched if ``value=None``.

    unit : astropy.Unit, optional.
        The unit that will be applied to the found value.

    header_comment : str, optional.
        The comment string to the header if ``update_header=True``. If `None`
        (default), search for existing comment in the original header by
        ``ccd.comments[key]`` and only overwrite the value by
        ``ccd.header[key]=found_value``. If it's not `None`, the comments will
        also be overwritten if ``update_header=True``.

    wrapper : function object, None, optional.
        The wrapper function that will be applied to the found value. Other
        keyword arguments should be given as a dict to `wrapper_kw`.

    wrapper_kw : dict, optional.
        The keyword argument to `wrapper`.

    Example
    -------
    >>> set_ccd_attribute(ccd, 'gain', value=2, unit='electron/adu')
    >>> set_ccd_attribute(ccd, 'ra', key='RA', unit=u.deg, default=0)

    Notes
    -----
    '''
    _t_start = Time.now()
    str_history = "From {}, {} = {} [unit = {}]"
    #                   value_from, name, value_Q.value, value_Q.unit

    if unit is None:
        try:
            unit = value.unit
        except AttributeError:
            unit = u.dimensionless_unscaled

    value_Q, value_from = get_if_none(
        value=value,
        header=ccd.header,
        key=key,
        unit=unit,
        verbose=verbose,
        default=default
    )
    if wrapper is not None:
        value_Q = wrapper(value_Q, **wrapper_kw)

    if update_header:
        s = [str_history.format(value_from, name, value_Q.value, value_Q.unit)]
        if key is not None:
            if header_comment is None:
                try:
                    header_comment = ccd.header.comments[key]
                except (KeyError, ValueError):
                    header_comment = ''

            try:
                v = ccd.header[key]
                s.append(f"(Original {key} = {v} is overwritten.)")
            except (KeyError, ValueError):
                pass

            ccd.header[key] = (value_Q.value, header_comment)
        # add as history
        cmt2hdr(ccd.header, 'h', s, t_ref=_t_start)

    setattr(ccd, name, value_Q)
    update_tlm(ccd.header)


# TODO: This is quite much overlapping with get_gain_rdnoise...
def set_ccd_gain_rdnoise(
        ccd,
        verbose=True,
        update_header=True,
        gain=None,
        rdnoise=None,
        gain_key="GAIN",
        rdnoise_key="RDNOISE",
        gain_unit=u.electron/u.adu,
        rdnoise_unit=u.electron
):
    """ A convenience set_ccd_attribute for gain and readnoise.

    Parameters
    ----------
    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. If `gain` or `readnoise` is specified,
        they are interpreted with `gain_unit` and `rdnoise_unit`, respectively.
        If they are not specified, this function will seek for the header with
        keywords of `gain_key` and `rdnoise_key`, and interprete the header
        value in the unit of `gain_unit` and `rdnoise_unit`, respectively.

    gain_key, rdnoise_key : str, optional.
        See `gain`, `rdnoise` explanation above.

    gain_unit, rdnoise_unit : str, astropy.Unit, optional.
        See `gain`, `rdnoise` explanation above.

    verbose : bool, optional.
        The verbose option.

    update_header : bool, optional
        Whether to update the given header.
    """
    gain_str = f"[{gain_unit:s}] Gain of the detector"
    rdn_str = f"[{rdnoise_unit:s}] Readout noise of the detector"
    set_ccd_attribute(
        ccd=ccd,
        name='gain',
        value=gain,
        key=gain_key,
        unit=gain_unit,
        default=1.0,
        header_comment=gain_str,
        update_header=update_header,
        verbose=verbose
    )
    set_ccd_attribute(
        ccd=ccd,
        name='rdnoise',
        value=rdnoise,
        key=rdnoise_key,
        unit=rdnoise_unit,
        default=0.0,
        header_comment=rdn_str,
        update_header=update_header,
        verbose=verbose
    )


def propagate_ccdmask(ccd, additional_mask=None):
    ''' Propagate the CCDData's mask and additional mask.

    Parameters
    ----------
    ccd : CCDData, ndarray
        The ccd to extract mask. If ndarray, it will only return a copy of
        `additional_mask`.

    additional_mask : mask-like, None
        The mask to be propagated.

    Notes
    -----
    The original ``ccd.mask`` is not modified. To do so,
    >>> ccd.mask = propagate_ccdmask(ccd, additional_mask=mask2)
    '''
    if additional_mask is None:
        try:
            mask = ccd.mask.copy()
        except AttributeError:  # i.e., if ccd.mask is None
            mask = None
    else:
        try:
            mask = ccd.mask | additional_mask
        except (TypeError, AttributeError):  # i.e., if ccd.mask is None:
            mask = deepcopy(additional_mask)
    return mask


# FIXME: Remove when https://github.com/astropy/ccdproc/issues/718 is solved
def trim_ccd(ccd, fits_section=None, update_header=True, verbose=False):
    _t = Time.now()
    trimmed_ccd = trim_image(ccd, fits_section=fits_section, add_keyword=update_header)

    if update_header:
        trim_str = f"Trimmed using {fits_section}"
        ndim = ccd.data.ndim  # ndim == NAXIS keyword
        shape = ccd.data.shape
        if fits_section is not None:
            trim_slice = fitsxy2py(fits_section)
            ltvs = []
            for i_python, npix in enumerate(shape):
                # example: "[10:110]", we must have LTV = -9, not -10.
                ltvs.append(-1*trim_slice[i_python].indices(npix)[0])
        else:
            ltvs = [0.]*ndim

        # python shape is in z, y, x order, so reverse it
        ltvs = ltvs[::-1]
        hdr = trimmed_ccd.header
        for i_axis, ltv in enumerate(ltvs):
            i = i_axis + 1
            try:  # if LTV exists already
                hdr[f"LTV{i}"] += ltv
                # NB: LTV is negative (gets more negative if more trimmed)
            except KeyError:
                hdr[f"LTV{i}"] = ltv

        for i_axis in range(ndim):
            i = i_axis + 1
            for j_axis in range(ndim):
                j = j_axis + 1
                if i == j:
                    if f"LTM{i}" in hdr:
                        hdr[f"LTM_{i}_{i}"] = hdr[f"LTM{i}"]
                    else:
                        hdr[f"LTM{i}_{i}"] = 1.
                else:
                    if f"LTM{i}_{j}" not in hdr:
                        hdr[f"LTM{i}_{j}"] = 0.

        cmt2hdr(hdr, 'h', trim_str, t_ref=_t, verbose=verbose)

    return trimmed_ccd


# TODO: Make it work for ND-CCD
def bezel_ccd(
        ccd,
        bezel_x=None,
        bezel_y=None,
        replace=np.nan,
        update_header=True,
        verbose=False
):
    """ Replace pixel values at the edges of the image.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The data to be used.

    bezel_x, bezel_y : None, int, float, size-2 of these, optional.
        The bezel width along x and y directions. If `float`, it will be
        rounded to integer. If given as size-2 array-like, it must be
        ``(bezel_lower, bezel_upper)``.

    replace : None, float-like, optinoal.
        If `None`, it is identical to trimming the CCD with given bezels. If
        given as float-like, the bezel pixels will be replaced with this value.
        Defaults to ``np.nan`` to keep the size of the input `ccd`.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData`
        The trimmed (``replace=None``) or bezel-replaced (otherwise) ccd.

    Example
    -------
    >>>

    Notes
    -----

    """
    def _sanitize_bezel(bezel, npix):
        if bezel is None:
            bezel = [0, 0]
        else:
            bezel = np.around(np.atleast_1d(bezel)).astype(int)
            if bezel.size == 1:
                bezel = np.repeat(bezel_x, 2)
            elif bezel.size != 2:
                raise ValueError("bezel must be size of 1 or 2.")

            bezel = bezel.ravel()
            if (bezel[0] >= npix) or (bezel[1] >= npix):
                raise ValueError("bezel width larger than image size")
            if bezel[0] + bezel[1] >= npix:
                raise ValueError("no pixel left after bezel")

        return bezel

    _t = Time.now()

    ccd = _parse_image(ccd, force_ccddata=True)[0]
    if ccd.data.ndim != 2:
        raise ValueError("Only 2-D input data is supported yet.")

    ny, nx = ccd.data.shape

    if bezel_x is None and bezel_y is None:
        return ccd

    bezel_x = _sanitize_bezel(bezel_x, nx)
    bezel_y = _sanitize_bezel(bezel_y, ny)

    if replace is None:
        sl = (f"[{bezel_x[0] + 1}:{nx - bezel_x[1]},{bezel_y[0] + 1}:{ny - bezel_y[1]}]")
        if isinstance(ccd, CCDData):
            nccd = trim_ccd(ccd, fits_section=sl, update_header=update_header)
        else:
            nccd = np.array(ccd)[fitsxy2py(sl)]
        # i.e., use trim_ccd if CCDData, and use python slice if ndarray-like
    else:
        nccd = ccd.copy()
        nccd.data[:bezel_y[0], :] = replace
        nccd.data[ny - bezel_y[1]:, :] = replace
        nccd.data[:, :bezel_x[0]] = replace
        nccd.data[:, nx - bezel_x[0]:] = replace
        if update_header:
            cmt2hdr(
                nccd.header, 'h', t_ref=_t, verbose=verbose,
                s=(f"Replaced pixels with bezel width {bezel_x} along x and "
                   + f"{bezel_y} along y with {replace}.")
            )

    return nccd


# FIXME: not finished.
def trim_overlap(inputs, extension=None, coordinate='image'):
    ''' Trim only the overlapping regions of the two CCDs

    Parameters
    ----------
    coordinate : str, optional.
        Ways to find the overlapping region. If ``'image'`` (default), output
        size will be ``np.min([ccd.shape for ccd in ccds], axis=0)``. If
        ``'physical'``, overlapping region will be found based on the physical
        coordinates.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    Notes
    -----
    WCS is not acceptable because no rotation/scaling is supported.
    '''
    items = inputs2list(inputs, sort=False, accept_ccdlike=True, check_coherency=False)
    if len(items) < 2:
        raise ValueError("inputs must have at least 2 objects.")

    offsets = []
    shapes = []
    reference = _parse_image(items[0], extension=extension, name=None, force_ccddata=True)
    for item in items:
        ccd, _, _ = _parse_image(item, extension=extension, name=None, force_ccddata=True)
        shapes.append(ccd.data.shape)
        offsets.append(
            calc_offset_physical(ccd, reference, order_xyz=False, ignore_ltm=True)
        )

    offsets, new_shape = _image_shape(
        shapes, offsets, method='overlap', intify_offsets=False
    )


# FIXME: docstring looks strange
def cut_ccd(ccd, position, size, mode='trim', fill_value=np.nan):
    ''' Converts the Cutout2D object to proper CCDData.

    Parameters
    ----------
    ccd: CCDData
        The ccd to be trimmed.

    position : tuple or `~astropy.coordinates.SkyCoord`
        The position of the cutout array's center with respect to the ``data``
        array. The position can be specified either as a ``(x, y)`` tuple of
        pixel coordinates or a `~astropy.coordinates.SkyCoord`, in which case
        wcs is a required input.

    size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array along each axis. If `size` is a scalar
        number or a scalar `~astropy.units.Quantity`, then a square cutout of
        `size` will be created. If `size` has two elements, they should be in
        ``(ny, nx)`` order. Scalar numbers in `size` are assumed to be in units
        of pixels. `size` can also be a `~astropy.units.Quantity` object or
        contain `~astropy.units.Quantity` objects. Such
        `~astropy.units.Quantity` objects must be in pixel or angular units.
        For all cases, `size` will be converted to an integer number of pixels,
        rounding the the nearest integer. See the `mode` keyword for additional
        details on the final cutout size.

        .. note::
            If `size` is in angular units, the cutout size is converted to
            pixels using the pixel scales along each axis of the image at the
            ``CRPIX`` location.  Projection and other non-linear distortions
            are not taken into account.

    wcs : `~astropy.wcs.WCS`, optional
        A WCS object associated with the input `data` array.  If `wcs` is not
        `None`, then the returned cutout object will contain a copy of the
        updated WCS for the cutout data array.

    mode : {'trim', 'partial', 'strict'}, optional
        The mode used for creating the cutout data array.  For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap of the cutout
        array and the input `data` array is sufficient. For the ``'strict'``
        mode, the cutout array has to be fully contained within the `data`
        array, otherwise an `~astropy.nddata.utils.PartialOverlapError` is
        raised.   In all modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'`` mode,
        positions in the cutout array that do not overlap with the `data` array
        will be filled with `fill_value`.  In ``'trim'`` mode only the
        overlapping elements are returned, thus the resulting cutout array may
        be smaller than the requested `shape`.

    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the cutout array
        that do not overlap with the input `data`. `fill_value` must have the
        same `dtype` as the input `data` array.
    '''
    hdr_orig = ccd.header
    w = WCS(hdr_orig)
    cutout = Cutout2D(
        data=ccd.data,
        position=position,
        size=size,
        wcs=w,
        mode=mode,
        fill_value=fill_value,
        copy=True
    )
    # Copy True just to avoid any contamination to the original ccd.

    nccd = CCDData(data=cutout.data, header=hdr_orig, wcs=cutout.wcs, unit=ccd.unit)
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
        warn(
            "Since Cutout2D is for small image crop, astropy do not currently support "
            + "distortion in WCS. This may result in slightly inaccurate WCS calculation."
        )

    update_tlm(nccd.header)

    return nccd


def bin_ccd(
        ccd,
        factor_x=1,
        factor_y=1,
        binfunc=np.mean,
        trim_end=False,
        update_header=True,
        copy=True
):
    ''' Bins the given ccd.

    Paramters
    ---------
    ccd : CCDData
        The ccd to be binned

    factor_x, factor_y : int, optional.
        The binning factors in x, y direction.

    binfunc : funciton object, optional.
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.

    trim_end : bool, optional.
        Whether to trim the end of x, y axes such that binning is done without
        error.

    update_header : bool, optional.
        Whether to update header. Defaults to True.

    Notes
    -----
    This is ~ 20-30 to upto 10^5 times faster than astropy.nddata's
    block_reduce:
    >>> from astropy.nddata.blocks import block_reduce
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
    Tested on  MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
    '''
    _t_start = Time.now()

    if not isinstance(ccd, CCDData):
        raise TypeError("ccd must be CCDData object.")

    if factor_x == 1 and factor_y == 1:
        return ccd

    if copy:
        _ccd = ccd.copy()
    else:
        _ccd = ccd

    _ccd.data = binning(_ccd.data,
                        factor_x=factor_x,
                        factor_y=factor_y,
                        binfunc=binfunc,
                        trim_end=trim_end)
    if update_header:
        _ccd.header["BINFUNC"] = (binfunc.__name__,
                                  "The function used for binning.")
        _ccd.header["XBINNING"] = (factor_x,
                                   "Binning done after the observation in X direction")
        _ccd.header["YBINNING"] = (factor_y,
                                   "Binning done after the observation in Y direction")
        # add as history
        cmt2hdr(_ccd.header, 'h', t_ref=_t_start,
                s=f"Binned by (xbin, ybin) = ({factor_x}, {factor_y}) ")
    return _ccd


# TODO: Need something (e.g., cython with pythran) to boost the speed of this function.
def fixpix(
        ccd,
        mask=None,
        maskpath=None,
        extension=None,
        mask_extension=None,
        priority=None,
        update_header=True,
        verbose=True
):
    ''' Interpolate the masked location (N-D generalization of IRAF PROTO.FIXPIX)
    Parameters
    ----------
    ccd : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
        The CCD data to be "fixed".

    mask : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like
        The mask to be used for fixing pixels (pixels to be fixed are where
        `mask` is `True`). If `None`, nothing will happen and `ccd` is
        returned.

    extension, mask_extension: int, str, (str, int), None
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    priority: tuple of int, None, optional.
        The priority of axis as a tuple of non-repeating `int` from ``0`` to
        `ccd.ndim`. It will be used if the mask has the same size along two or
        more of the directions. To specify, use the integers for axis
        directions, descending priority. For example,  ``(2, 1, 0)`` will be
        identical to `priority=None` (default) for 3-D images.
        Default is `None` to follow IRAF's PROTO.FIXPIX: Priority is higher for
        larger axis number (e.g., in 2-D, x-axis (axis=1) has higher priority
        than y-axis (axis=0)).

    Examples
    --------
    Timing test: MBP 15" [2018, macOS 11.4, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)], 2021-11-05 11:14:04 (KST:
    GMT+09:00)
    >>> np.random.RandomState(123)  # RandomState(MT19937) at 0x7FAECA768D40
    >>> data = np.random.normal(size=(1000, 1000))
    >>> mask = np.zeros_like(data).astype(bool)
    >>> mask[10, 10] = True
    >>> %timeit yfu.fixpix(data, mask)
    19.7 ms +- 1.53 ms per loop (mean +- std. dev. of 7 runs, 100 loops each)
    >>> print(data[9:12, 9:12], yfu.fixpix(data, mask)[9:12, 9:12])
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621  1.37965928  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]]
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621 -0.32934328  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]] adu
    '''
    if mask is None:
        return ccd.copy()

    _t_start = Time.now()

    _ccd, _, _ = _parse_image(ccd, extension=extension, force_ccddata=True)
    mask, maskpath, _ = _parse_image(mask, extension=mask_extension, name=maskpath)
    mask = mask.data.astype(bool)
    data = _ccd.data
    naxis = _ccd.shape

    if _ccd.shape != mask.shape:
        raise ValueError("ccd and mask must have the identical shape; "
                         + f"now {_ccd.shape} VS {mask.shape}.")

    ndim = data.ndim

    if priority is None:
        priority = tuple([i for i in range(ndim)][::-1])
    elif len(priority) != ndim:
        raise ValueError("len(priority) and ccd.ndim must be the same; "
                         + f"now {len(priority)} VS {ccd.ndim}.")
    elif not isinstance(priority, tuple):
        priority = tuple(priority)
    elif (np.min(priority) != 0) or (np.max(priority) != ndim - 1):
        raise ValueError(f"`priority` must be a tuple of int (0 <= int <= {ccd.ndim-1=}). "
                         + f"Now it's {priority=}")

    structures = [np.zeros([3]*ndim) for _ in range(ndim)]
    for i in range(ndim):
        sls = [[slice(1, 2, None)]*ndim for _ in range(ndim)][0]
        sls[i] = slice(None, None, None)
        structures[i][tuple(sls)] = 1
    # structures[i] is the structure to obtain the num. of connected pix. along axis=i

    pixels = []
    n_axs = []
    labels = []

    for structure in structures:
        _label, _nlabel = ndlabel(mask, structure=structure)
        _pixels = {}
        _n_axs = {}
        for k in range(1, _nlabel + 1):
            _label_k = (_label == k)
            _pixels[k] = np.where(_label_k)
            _n_axs[k] = np.count_nonzero(_label_k)
        labels.append(_label)
        pixels.append(_pixels)
        n_axs.append(_n_axs)

    idxs = np.where(mask)
    for pos in np.array(idxs).T:
        # The label of this position in each axis
        label_pos = [lab.item(*pos) for lab in labels]
        # number of pixels of the same label for each direction
        n_ax = [_n_ax[lab] for _n_ax, lab in zip(n_axs, label_pos)]

        # The shortest axis along which the interpolation will happen,
        # OR, if 1+ directions having same minimum length, select this axis
        #   according to `priority`
        interp_ax = np.where(n_ax == np.min(n_ax))[0]
        if len(interp_ax) > 1:
            for i_ax in priority:  # check in the identical order to `priority`
                if i_ax in interp_ax:
                    interp_ax = i_ax
                    break
        else:
            interp_ax = interp_ax[0]
        # The coordinates of the pixels having the identical label to this
        # pixel position, along the shortest axis
        coord_samelabel = pixels[interp_ax][label_pos[interp_ax]]
        isinvalid = []
        coord_slice = []
        coord_init = []
        coord_last = []
        for i in range(ndim):
            invalid = False
            if i == interp_ax:
                init = np.min(coord_samelabel[i]) - 1
                last = np.max(coord_samelabel[i]) + 1
                # distance between the initial/last points to be used for the
                # interpolation, along the interpolation axis:
                delta = last - init
                # grid for interpolation:
                grid = np.arange(1, delta - 0.1, 1)
                # Slice to be used for interpolation:
                sl = slice(init + 1, last, None)
                # Should be done here, BEFORE the if clause below.

                # Check if lower/upper are all outside the image
                if init < 0 and last >= naxis[i]:
                    invalid = True
                elif init < 0:  # if only one of lower/upper is outside the image
                    init = last
                elif last >= naxis[i]:
                    last = init
            else:
                init = coord_samelabel[i][0]
                last = coord_samelabel[i][0]
                # coord_samelabel[i] is nothing but an array of same numbers
                sl = slice(init, last + 1, None)

            coord_init.append(init)
            coord_last.append(last)
            coord_slice.append(sl)
            isinvalid.append(invalid)

        val_init = data.item(tuple(coord_init))
        val_last = data.item(tuple(coord_last))
        data[tuple(coord_slice)].flat = (val_last - val_init)/delta*grid + val_init

    if update_header:
        nfix = np.count_nonzero(mask)
        _ccd.header["MASKNPIX"] = (nfix, "No. of pixels masked (fixed) by fixpix.")
        _ccd.header["MASKFILE"] = (maskpath, "Applied mask for fixpix.")
        _ccd.header["MASKORD"] = (str(priority), "Axis priority for fixpix (python order)")
        # MASKFILE: name identical to IRAF
        # add as history
        cmt2hdr(_ccd.header, 'h', t_ref=_t_start, verbose=verbose,
                s="Pixel values fixed by fixpix.")
        update_process(_ccd.header, "P")

    return _ccd


# # FIXME: Remove this after fixpix is completed
# def fixpix_griddata(ccd, mask, extension=None, method='nearest',
#     fill_value=0, update_header=True):
#     ''' Interpolate the masked location (cf. IRAF's PROTO.FIXPIX)
#     Parameters
#     ----------
#     ccd : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
#         The CCD data to be "fixed".

#     mask : ndarray (bool)
#         The mask to be used for fixing pixels (pixels to be fixed are where
#         `mask` is `True`).

#     extension: int, str, (str, int)
#         The extension of FITS to be used. It can be given as integer
#         (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple
#         of str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the
#         *first extension with data* will be used.

#     method: str
#         The interpolation method. Even the ``'linear'`` method takes too long
#         time in many cases, so the default is ``'nearest'``.
#     '''
#     _t_start = Time.now()

#     _ccd, _, _ = _parse_image(ccd, extension=extension, force_ccddata=True)
#     data = _ccd.data

#     x_idx, y_idx = np.meshgrid(np.arange(0, data.shape[1] - 0.1),
#                                np.arange(0, data.shape[0] - 0.1))
#     mask = mask.astype(bool)
#     x_valid = x_idx[~mask]
#     y_valid = y_idx[~mask]
#     z_valid = data[~mask]
#     _ccd.data = griddata((x_valid, y_valid),
#                          z_valid, (x_idx, y_idx), method=method, fill_value=fill_value)

#     if update_header:
#         _ccd.header["MASKMETH"] = (method,
#                                    "The interpolation method for fixpix")
#         _ccd.header["MASKFILL"] = (fill_value,
#                                    "The fill value if interpol. fails in fixpix")
#         _ccd.header["MASKNPIX"] = (np.count_nonzero(mask),
#                                    "Total num of pixesl fixed by fixpix.")
#         # add as history
#         cmt2hdr(_ccd.header, 'h', t_ref=_t_start, s="Pixel values fixed by fixpix")
#     update_tlm(_ccd.header)

#     return _ccd


def find_extpix(
        ccd,
        mask=None,
        npixs=(1, 1),
        bezels=None,
        order_xyz=True,
        sort=True,
        update_header=True,
        verbose=0
):
    """ Finds the N extrema pixel values excluding masked pixels.

    Paramters
    ---------
    ccd : CCDData
        The ccd to find extreme values

    mask : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
        The mask to be used. To reduce file I/O time, better to provide
        ndarray.

    npixs : length-2 tuple of int, optional
        The numbers of extrema to find, in the form of ``[small, large]``, so
        that ``small`` number of smallest and ``large`` number of largest pixel
        values will be found. If `None`, no extrema is found (`None` is
        returned for that extremum).
        Deafult: ``(1, 1)`` (find minimum and maximum)

    bezels : list of list of int, optional.
        If given, must be a list of list of int. Each list of int is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.

    order_xyz : bool, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    sort: bool, optional.
        Whether to sort the extrema in ascending order.

    Returns
    -------
    min
        The list of extrema pixel values.
    """
    if not len(npixs) == 2:
        raise ValueError("npixs must be a length-2 tuple of int.")
    _t = Time.now()
    data = ccd.data.copy().astype("float32")  # Not float64 to reduce memory usage
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]

    if mask is None:
        maskname = "No mask"
        mask = ~np.isfinite(data)
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _parse_image(mask, force_ccddata=True)
            mask = mask.data | ~np.isfinite(data)
        else:
            maskname = "User-provided mask"

    exts = []
    for npix, sign, minmaxval in zip(npixs, [1, -1], [np.inf, -np.inf]):
        if npix is None:
            exts.append(None)
            continue
        data[mask] = minmaxval
        # ^ if getting maximum/minimum pix vals, replace with minimum/maximum
        extvals = np.partition(data.ravel(), sign*npix)
        #         ^^^^^^^^^^^^
        # bn.partitoin has virtually no speed gain.
        extvals = extvals[:npix] if sign > 0 else extvals[-npix:]
        if sort:
            extvals = np.sort(extvals)[::sign]
        exts.append(extvals)

    if update_header:
        for ext, mm in zip(exts, ["min", "max"]):
            if ext is not None:
                for i, extval in enumerate(ext):
                    ccd.header.set(f"{mm.upper()}V{i+1:03d}", extval, f"{mm} pixel value")
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        cmt2hdr(ccd.header, 'h', verbose=verbose, t_ref=_t,
                s=(f"Extrema pixel values found N(smallest, largest) = {npixs} "
                   + f"excluding mask ({maskname}){bezstr}. See MINViii and MAXViii.")
                )
    return exts


def find_satpix(
        ccd,
        mask=None,
        satlevel=65535,
        bezels=None,
        order_xyz=True,
        update_header=True,
        verbose=0
):
    """ Finds saturated pixel values excluding masked pixels.

    Paramters
    ---------
    ccd : CCDData, ndarray
        The ccd to find extreme values. If `ndarray`, `update_header` will
        automatically be set to `False`.

    mask : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
        The mask to be used. To reduce file I/O time, better to provide
        ndarray.

    satlevel: numeric, optional.
        The saturation level. Pixels >= `satlevel` will be retarded as
        saturated pixels, except for those masked by `mask`.

    bezels : list of list of int, optional.
        If given, must be a list of list of int. Each list of int is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.

    order_xyz : bool, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    Returns
    -------
    min
        The list of extrema pixel values.
    """
    _t = Time.now()
    if isinstance(ccd, CCDData):
        data = ccd.data.copy()
    else:
        data = ccd.copy()
        update_header = False
    satmask = np.zeros(data.shape, dtype=bool)
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]
    else:
        sls = [slice(None, None, None) for _ in range(data.ndim)]

    if mask is None:
        maskname = "No mask"
        satmask[sls] = data >= satlevel
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _parse_image(mask, force_ccddata=True)
            mask = mask.data
        else:
            maskname = "User-provided mask"
        satmask[sls] = (data >= satlevel) & (~mask)  # saturated && not masked

    if update_header:
        nsat = np.count_nonzero(satmask[sls])
        ccd.header["NSATPIX"] = (nsat, "No. of saturated pix")
        ccd.header["SATLEVEL"] = (satlevel, "Saturation: pixels >= this value")
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        cmt2hdr(ccd.header, 'h', verbose=verbose, t_ref=_t,
                s=(f"Saturated pixels calculated based on satlevel = {satlevel}, excluding "
                   + f"mask ({maskname}){bezstr}. See NSATPIX and SATLEVEL."))
    return satmask


def make_errormap(
        ccd,
        gain_epadu=1,
        rdnoise_electron=0,
        flat_err=0.0,
        subtracted_dark=None,
        return_variance=False
):
    print("Use `errormap` instead.")
    return errormap(ccd, gain_epadu=gain_epadu, rdnoise_electron=rdnoise_electron,
                    subtracted_dark=subtracted_dark, flat_err=flat_err,
                    return_variance=return_variance)


def errormap(
        ccd_biassub,
        gain_epadu=1,
        rdnoise_electron=0,
        subtracted_dark=0.,
        flat=1.,
        dark_std=0.,
        flat_err=0.,
        dark_std_min='rdnoise',
        return_variance=False
):
    ''' Calculate the detailed pixel-wise error map in ADU unit.

    Parameters
    ----------
    ccd : CCDData, PrimaryHDU, ImageHDU, ndarray.
        The ccd data which will be used to generate error map. It must be
        **bias subtracted**. If dark is subtracted, give `subtracted_dark`.
        This array will be added to ``ccd.data`` and used to calculate the
        Poisson noise term. If the amount of this subtracted dark is
        negligible, you may just set ``subtracted_dark = None`` (default).

    gain_epadu, rdnoise_electron : float, array-like, or Quantity, optional.
        The effective gain factor in ``electron/ADU`` unit and the readout
        noise in ``electron`` unit.

    subtracted_dark : array-like
        The subtracted dark map.
        Default: 0.

    flat : ndarray, optional.
        The flat field value. There is no need that flat values are normalized.
        Default: 1.

    flat_err : float, array-like optional.
        The uncertainty of the flat, which is obtained by the central limit
        theorem (sample standard deviation of the pixel divided by the square
        root of the number of flat frames). An example in IRAF and DAOPHOT: the
        uncertainty from the flat fielding ``flat_err/flat`` is set as a
        constant (see, e.g., eq 10 of StetsonPB 1987, PASP, 99, 191) set as
        Stetson used 0.0075 (0.75% fractional uncertainty), and the same is
        implemented to IRAF DAOPHOT:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daopars
        Default: 0.

    dark_std : float, array-like, optional.
        The sample standard deviation of dark pixels. It **should not be
        divided by the number of dark frames**, because we are interested in
        the uncertainty in the dark (prediction), not the confidence interval
        of the *mean* of the dark.
        Default: 0.

    dark_std_min : 'rdnoise', float, optional.
        The minimum value for `dark_std`. Any `dark_std` value below this will
        be replaced by this value. If ``'rdnoise'`` (default), the
        ``rdnoise_electron/gain_epadu`` will be used.

    return_variance: bool, optional
        Whether to return as variance map. Default is `False`, i.e., return the
        square-rooted standard deviation map. It's better to use variance for
        large image size (computation speed issue).

    Example
    -------
    >>> from astropy.nddata import CCDData, StdDevUncertainty
    >>> ccd = CCDData.read("obj001.fits", 0)
    >>> hdr = ccd.header
    >>> dark = CCDData.read("master_dark.fits", 0)
    >>> params = dict(gain_epadu=hdr["GAIN"], rdnoise_electron=hdr["RDNOISE"],
    >>>               subtracted_dark=dark.data)
    >>> ccd.uncertainty = StdDevUncertainty(errormap(ccd, **params))

    '''
    data, _ = _parse_data_header(ccd_biassub)
    data[data < 0] = 0  # make all negative pixel to 0

    if isinstance(gain_epadu, u.Quantity):
        gain_epadu = gain_epadu.to(u.electron / u.adu).value
    elif isinstance(gain_epadu, str):
        gain_epadu = float(gain_epadu)

    if isinstance(rdnoise_electron, u.Quantity):
        rdnoise_electron = rdnoise_electron.to(u.electron).value
    elif isinstance(rdnoise_electron, str):
        rdnoise_electron = float(rdnoise_electron)

    if dark_std_min == 'rdnoise':
        dark_std_min = rdnoise_electron/gain_epadu
    if isinstance(dark_std, np.ndarray):
        dark_std[dark_std < dark_std_min] = dark_std_min

    # Calculate the full variance map
    # restore dark for Poisson term calculation
    eval_str = ("(data + subtracted_dark)/(gain_epadu*flat**2)"
                + "+ (dark_std/flat)**2"
                + "+ data**2*(flat_err/flat)**2"
                + "+ (rdnoise_electron/(gain_epadu*flat))**2"
                )

    if return_variance:
        return NEVAL(eval_str)
    else:  # Sqrt is the most time-consuming part...
        return NEVAL(f"{NPSTR}sqrt({eval_str})")

    # var_pois = data / (gain_epadu * flat**2)
    # var_rdn = (rdnoise_electron/(gain_epadu*flat))**2
    # var_flat_err = data**2*(flat_err/flat)**2
    # var_dark_err = (dark_err/flat)**2


def cmt2hdr(
        header,
        histcomm,
        s,
        precision=3,
        time_fmt="{:.>72s}",
        t_ref=None,
        dt_fmt="(dt = {:.3f} s)",
        verbose=False,
        set_kw={'after': -1}
):
    ''' Automatically add timestamp as well as HISTORY or COMMENT string

    Parameters
    ----------
    header : Header
        The header.

    histcomm : str in ['h', 'hist', 'history', 'c', 'comm', 'comment']
        Whether to add history or comment.

    s : str or list of str
        The string to add as history or comment.

    precision : int, optional.
        The precision of the isot format time.

    time_fmt : str, None, optional.
        The Python 3 format string to format the time in the header. If `None`,
        the timestamp string will not be added.

        Examples::
          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in ``()``. ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with gain_key="GAIN", ``_``.

    t_ref : Time
        The reference time. If not `None`, delta time is calculated.

    dt_fmt : str, optional.
        The Python 3 format string to format the delta time in the header.

    verbose : bool, optional.
        Whether to print the same information on the output terminal.

    verbose_fmt : str, optional.
        The Python 3 format string to format the time in the terminal.

    set_kw : dict, optional.
        The keyword arguments added to `Header.set()`. Default is
        ``{'after':-1}``, i.e., the history or comment will be appended to the
        very last part of the header.

    Notes
    -----
    The timming benchmark shows that,
    %timeit cmt2hdr(hdu.header, 'comm', 'aadfaer sdf')
    310 µs +/- 2.93 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)

    %timeit cmt2hdr(hdu.header, 'comM', 'aadfaer sdf')
    309 µs +/- 2.48 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)

    %timeit cmt2hdr(hdu.header, 'comMent', 'aadfaer sdf')
    15.4 ms +/- 299 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    '''
    pall = locals()
    if histcomm.lower() in ['h', 'hist']:
        pall['histcomm'] = 'HISTORY'
        return cmt2hdr(**pall)
    elif histcomm.lower() in ['c', 'com', 'comm']:
        pall['histcomm'] = 'COMMENT'
        return cmt2hdr(**pall)
    # The "elif not in raise" gives large bottleneck if, e.g., ``histcomm="ComMent"``...
    # elif histcomm.lower() not in ['history', 'comment']:
    #     raise ValueError("Only HISTORY or COMMENT are supported now.")

    def _add_content(header, content):
        try:
            header.set(histcomm, content, **set_kw)
        except AttributeError:
            # For a CCDData that has just initialized, header is in OrderdDict, not Header
            header[histcomm] = content

    if isinstance(s, str):
        s = [s]

    for _s in s:
        _add_content(header, _s)
        if verbose:
            print(f"{histcomm.upper():<8s} {_s}")

    if time_fmt is not None:
        timestr = str_now(precision=precision, fmt=time_fmt, t_ref=t_ref, dt_fmt=dt_fmt)
        _add_content(header, timestr)
        if verbose:
            print(f"{histcomm.upper():<8s} {timestr}")
    update_tlm(header)


def update_tlm(header):
    ''' Adds the IRAF-like ``FITS-TLM`` right after ``NAXISi``.
    '''
    now = Time(Time.now(), precision=0).isot
    try:
        del header["FITS-TLM"]
    except KeyError:
        pass
    try:
        header.set("FITS-TLM",
                   value=now,
                   comment="UT of last modification of this FITS file",
                   after=f"NAXIS{header['NAXIS']}")
    except AttributeError:  # If header is OrderedDict
        header["FITS-TLM"] = (now, "UT of last modification of this FITS file")


def update_process(
        header,
        process=None,
        key="PROCESS",
        delimiter='-',
        add_comment=True,
        additional_comment=dict()
):
    """ update the process history keyword in the header.

    Parameters
    ----------
    header : header
        The header to update the ``PROCESS`` (tunable by `key` parameter)
        keyword.

    process : str or list-like of str
        The additional process keys to add to the header.

    key : str, optional.
        The key for the process-related header keyword.

    delimiter : str, optional.
        The delimiter for each process. It can be null string (``''``). The
        best is to match it with the pre-existing delimiter of the
        ``header[key]``.

    add_comment : bool, optional.
        Whether to add a comment to the header if there was no `key`
        (``"PROCESS"`` by default) in the header.

    additional_comment : dict, optional.
        The additional comment to add. For instance, ``dict(v="vertical
        pattern", f="fourier pattern")`` will add a new line of comment which
        reads "User added items for `key`: v=vertical pattern, f=fourier
        pattern."
    """
    process = listify(process)

    if key in header:
        process = header[key].split(delimiter) + process
        # do not additionally add comment.
    elif add_comment:
        # add comment.
        cmt2hdr(
            header, 'c', time_fmt=None,
            s=(f"Standard items for {key} includes B=bias, D=dark, F=flat, T=trim, W=WCS, "
               + "C=CRrej, Fr=fringe, P=fixpix, X=crosstalk.")
        )

    header[key] = (delimiter.join(process), "Process (order: 1-2-3-...): see comment.")

    if additional_comment:
        addstr = ", ".join([f"{k}={v}" for k, v in additional_comment.items()])
        cmt2hdr(header, 'c', f"User added items to {key}: {addstr}.", time_fmt=None)
    update_tlm(header)


def hedit(
        item,
        keys,
        values,
        comments=None,
        befores=None,
        afters=None,
        add=False,
        output=None,
        overwrite=False,
        output_verify="fix",
        verbose=True
):
    """ Edit the header key (usu. to update value of a keyword).

    Parameters
    ----------
    item : `astropy` header, path-like, CCDData-like
        The FITS file or header to edit. If `Header`, it is updated
        **inplace**.

    keys : str, list-like of str
        The key to edit.

    values : str, numeric, or list-like of such
        The new value. To pass one single iterable (e.g., `[1, 2, 3]`) for one
        single `key`, use a list of it (e.g., `[[1, 2, 3]]`) to circumvent
        problem.

    comment : str, list-like of str optional.
        The comment to add.

    add : bool, optional.
        Whether to add the key if it is not in the header.

    befores : str, int, list-like of such, optional
        Name of the keyword, or index of the `Card` before which this card
        should be located in the header. The argument `before` takes
        precedence over `after` if both specified.

    after : str, int, list-like of such, optional
        Name of the keyword, or index of the `Card` after which this card
        should be located in the header.

    output: path-like, optional
        The output file.

    Returns
    -------
    ccd : CCDData
        The header-updated CCDData. `None` if `item` was pure Header.
    """
    def _add_key(header, key, val, infostr, cmt=None, before=None, after=None):
        header.set(key, value=val, comment=cmt, before=before, after=after)
        # infostr += " (comment: {})".format(comment) if comment is not None else ""
        if before is not None:
            infostr += f" (moved: {before=})"
        elif after is not None:  # `after` is ignored if `before` is given
            infostr += f" (moved: {after=})"
        cmt2hdr(header, 'h', infostr, verbose=verbose)
        update_tlm(header)

    if isinstance(item, fits.header.Header):
        header = item
        if verbose:
            print("item is astropy Header. (any `output` is igrnoed).")
        output = None
        ccd = None
    elif isinstance(item, ASTROPY_CCD_TYPES):
        ccd, imname, _ = _parse_image(item, force_ccddata=True, copy=False)
        #                                                   ^^^^^^^^^^
        # Use copy=False to update header of the input CCD inplace.z
        header = ccd.header

    keys, values, comments, befores, afters = listify(keys, values, comments,
                                                      befores, afters)

    for key, val, cmt, bef, aft in zip(keys, values, comments, befores, afters):
        if key in header:
            oldv = header[key]
            infostr = (f"[HEDIT] {key}={oldv} ({type(oldv).__name__}) "
                       + f"--> {val} ({type(val).__name__})")
            _add_key(header, key, val, infostr, cmt=cmt, before=bef, after=aft)
        else:
            if add:  # add key only if `add` is True.
                infostr = f"[HEDIT add] {key}= {val} ({type(val).__name__})"
                _add_key(header, key, val, infostr, cmt=cmt, before=bef, after=aft)
            elif verbose:
                print(f"{key = } does not exist in the header. Skipped. (add=True to proceed)")

    if output is not None:
        ccd.write(output, overwrite=overwrite, output_verify=output_verify)
        if verbose:
            print(f"{imname} --> {output}")

    return ccd


def key_remover(header, remove_keys, deepremove=True):
    ''' Removes keywords from the header.

    Parameters
    ----------
    header : Header
        The header to be modified

    remove_keys : list of str
        The header keywords to be removed.

    deepremove : True, optional
        FITS standard does not have any specification of duplication of
        keywords as discussed in the following issue:
        https://github.com/astropy/ccdproc/issues/464
        If it is set to `True`, ALL the keywords having the name specified in
        `remove_keys` will be removed. If not, only the first occurence of each
        key in `remove_keys` will be removed. It is more sensical to set it
        `True` in most of the cases.
    '''
    nhdr = header.copy()
    if deepremove:
        for key in remove_keys:
            while True:
                try:
                    nhdr.remove(key)
                except KeyError:
                    break
    else:
        for key in remove_keys:
            try:
                nhdr.remove(key)
            except KeyError:
                continue

    return nhdr


def key_mapper(header, keymap=None, deprecation=False, remove=False):
    ''' Update the header to meed the standard (keymap).

    Parameters
    ----------
    header : Header
        The header to be modified

    keymap : dict
        The dictionary contains ``{<standard_key>:<original_key>}``
        information. If it is `None` (default), the copied version of the
        header is returned without any change.

    deprecation : bool, optional
        Whether to change the original keywords' comments to contain
        deprecation warning. If `True`, the original keywords' comments will
        become ``DEPRECATED. See <standard_key>.``. It has no effect if
        ``remove=True``.
        Default is `False`.

    remove : bool, optional.
        Whether to remove the original keyword. `deprecation` is ignored if
        ``remove=True``.
        Default is `False`.

    Returns
    -------
    newhdr: Header
        The updated (key-mapped) header.

    Notes
    -----
    If the new keyword already exist in the given header, virtually nothing
    will happen. If ``deprecation=True``, the old one's comment will be
    changed, and if ``remove=True``, the old one will be removed; the new
    keyword will never be changed or overwritten.
    '''
    def _rm_or_dep(hdr, old, new):
        if remove:
            hdr.remove(old)
        elif deprecation:  # do not remove but deprecate
            hdr.comments[old] = f"DEPRECATED. See {new}"

    newhdr = header.copy()
    if keymap is not None:
        for k_new, k_old in keymap.items():
            if k_new == k_old:
                continue

            if k_old is not None:
                if k_new in newhdr:  # if k_new already in the header, JUST deprecate k_old.
                    _rm_or_dep(newhdr, k_old, k_new)
                else:  # if not, copy k_old to k_new and deprecate k_old.
                    try:
                        comment_ori = newhdr.comments[k_old]
                        newhdr[k_new] = (newhdr[k_old], comment_ori)
                        _rm_or_dep(newhdr, k_old, k_new)
                    except (KeyError, IndexError):
                        # don't even warn
                        pass

    return newhdr


def chk_keyval(type_key, type_val, group_key):
    ''' Checks the validity of key and values used heavily in combutil.

    Parameters
    ----------
    type_key : None, str, list of str, optional
        The header keyword for the ccd type you want to use for match.

    type_val : None, int, str, float, etc and list of such
        The header keyword values for the ccd type you want to match.


    group_key : None, str, list of str, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.

    Returns
    -------
    type_key, type_val, group_key
    '''
    # Make type_key to list
    if type_key is None:
        type_key = []
    elif is_list_like(type_key):
        try:
            type_key = list(type_key)
            if not all(isinstance(x, str) for x in type_key):
                raise TypeError("Some of type_key are not str.")
        except TypeError as E:
            raise E("type_key should be str or convertible to list.")
    elif isinstance(type_key, str):
        type_key = [type_key]
    else:
        raise TypeError(f"`type_key` not understood (type = {type(type_key)}): {type_key}")

    # Make type_val to list
    if type_val is None:
        type_val = []
    elif is_list_like(type_val):
        try:
            type_val = list(type_val)
        except TypeError as E:
            raise E("type_val should be str or convertible to list.")
    elif isinstance(type_val, str):
        type_val = [type_val]
    else:
        raise TypeError(f"`type_val` not understood (type = {type(type_val)}): {type_val}")

    # Make group_key to list
    if group_key is None:
        group_key = []
    elif is_list_like(group_key):
        try:
            group_key = list(group_key)
            if not all(isinstance(x, str) for x in group_key):
                raise TypeError("Some of group_key are not str.")
        except TypeError:
            raise TypeError("group_key should be str or convertible to list.")
    elif isinstance(group_key, str):
        group_key = [group_key]
    else:
        raise TypeError(
            f"`group_key` not understood (type = {type(group_key)}): {group_key}"
        )

    if len(type_key) != len(type_val):
        raise ValueError("`type_key` and `type_val` must have the same length!")

    # If there is overlap
    overlap = set(type_key).intersection(set(group_key))
    if len(overlap) > 0:
        warn(f"{overlap} appear in both `type_key` and `group_key`."
             + "It may not be harmful but better to avoid.")

    return type_key, type_val, group_key


def get_from_header(header, key, unit=None, verbose=True, default=0):
    ''' Get a variable from the header object.

    Parameters
    ----------
    header : astropy.Header
        The header to extract the value.

    key : str
        The header keyword to extract.

    unit : astropy unit
        The unit of the value.

    default : str, int, float, ..., or Quantity
        The default if not found from the header.

    Returns
    -------
    q: Quantity or any object
        The extracted quantity from the header. It's a Quantity if the unit is
        given. Otherwise, appropriate type will be assigned.
    '''
    # If using q = header.get(key, default=default),
    # we cannot give any meaningful verboses infostr.
    # Anyway the `header.get` sourcecode contains only 4-line:
    # ``try: return header[key] // except (KeyError, IndexError): return default.
    key = key.upper()
    try:
        q = change_to_quantity(header[key], desired=unit)
        if verbose:
            print(f"header: {key:<8s} = {q}")
    except (KeyError, IndexError):
        q = change_to_quantity(default, desired=unit)
        warn(f"The key {key} not found in header: setting to {default}.")

    return q


def get_if_none(value, header, key, unit=None, verbose=True, default=0, to_value=False):
    ''' Similar to get_from_header, but a convenience wrapper.
    '''
    if value is None:
        value_Q = get_from_header(header, key, unit=unit, verbose=verbose, default=default)
        value_from = f"{key} in header"
    else:
        value_Q = change_to_quantity(value, unit, to_value=False)
        value_from = "the user"

    if to_value:
        return value_Q.value, value_from
    else:
        return value_Q, value_from


def wcs_crota(wcs, degree=True):
    '''
    Notes
    -----
    https://iraf.net/forum/viewtopic.php?showtopic=108893
    CROTA2 = arctan (-CD1_2 / CD2_2) = arctan ( CD2_1 / CD1_1)
    '''
    if isinstance(wcs, WCS):
        wcsprm = wcs.wcs
    elif isinstance(wcs, Wcsprm):
        wcsprm = wcs
    else:
        raise TypeError("wcs type not understood. "
                        + "It must be either astropy.wcs.WCS or astropy.wcs.Wcsprm")

    # numpy arctan2 gets y-coord (numerator) and then x-coord(denominator)
    crota = np.arctan2(wcsprm.cd[0, 0], wcsprm.cd[1, 0])
    if degree:
        crota = np.rad2deg(crota)

    return crota


def center_radec(
        header,
        center_of_image=True,
        ra_key="RA",
        dec_key="DEC",
        equinox=None,
        frame=None,
        equinox_key="EPOCH",
        frame_key="RADECSYS",
        ra_unit=u.hourangle,
        dec_unit=u.deg,
        mode='all',
        verbose=True,
        plain=False
):
    ''' Returns the central ra/dec from header or WCS.

    Notes
    -----
    Even though RA or DEC is in sexagesimal, e.g., "20 53 20", astropy
    correctly reads it in such a form, so no worries.

    Parameters
    ----------
    header : Header
        The header to extract the central RA/DEC from keywords or WCS.

    center_of_image : bool, optional
        If `True`, WCS information will be extracted from the header, rather
        than relying on the `ra_key` and `dec_key` keywords directly. If
        `False`, `ra_key` and `dec_key` from the header will be understood as
        the "center" and the RA, DEC of that location will be returned.

    equinox, frame : str, optional
        The `equinox` and `frame` for SkyCoord. Default (`None`) will use the
        default of SkyCoord. Important only if ``usewcs=False``.

    XX_key : str, optional
        The header key to find XX if ``XX`` is `None`. Important only if
        ``usewcs=False``.

    XX_unit : Quantity, optional
        The unit of ``XX``. Important only if ``usewcs=False``.

    mode : 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core WCS transformation (``'wcs'``). Important
        only if ``usewcs=True``.

    plain : bool
        If `True`, only the values of RA/DEC in degrees will be returned.
    '''
    if center_of_image:
        w = WCS(header)
        nx, ny = float(header["NAXIS1"]), float(header["NAXIS2"])
        centx = nx / 2 - 0.5
        centy = ny / 2 - 0.5
        coo = SkyCoord.from_pixel(centx, centy, wcs=w, origin=0, mode=mode)
    else:
        ra = get_from_header(header, ra_key, verbose=verbose)
        dec = get_from_header(header, dec_key, verbose=verbose)
        if equinox is None:
            equinox = get_from_header(header, equinox_key, verbose=verbose, default=None)
        if frame is None:
            frame = get_from_header(
                header, frame_key, verbose=verbose, default=None
            ).lower()
        coo = SkyCoord(
            ra=ra, dec=dec, unit=(ra_unit, dec_unit), frame=frame, equinox=equinox
        )

    if plain:
        return coo.ra.value, coo.dec.value
    return coo


def calc_offset_wcs(
        target,
        reference,
        loc_target='center',
        loc_reference='center',
        order_xyz=True,
        intify_offset=False
):
    ''' The pixel offset of target's location when using WCS in referene.

    Parameters
    ----------

    target : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like, WCS
        The object to extract header to calculate the position

    reference : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like, WCS
        The object to extract reference WCS (or header to extract WCS) to
        calculate the position *from*.

    loc_target : str (center, origin) or ndarray optional.
        The location to calculate the position (in pixels and in xyz order).
        Default is ``'center'`` (half of ``NAXISi`` keys in `target`). The
        `location`'s world coordinate is calculated from the WCS information in
        `target`. Then it will be transformed to the image coordinate of
        `reference`.

    loc_reference : str (center, origin) or ndarray optional.
        The location of the reference point (in pixels and in xyz order) in
        `reference`'s coordinate
        to calculate the offset.

    order_xyz : bool, optional.
        Whether to return the position in xyz order or not (python order:
        ``[::-1]`` of the former). Default is `True`.
    '''
    def _parse_loc(loc, obj):
        if isinstance(obj, WCS):
            w = obj
        else:
            _, hdr = _parse_data_header(obj, parse_data=False, copy=False)
            w = WCS(hdr)

        if loc == 'center':
            _loc = np.atleast_1d(w._naxis)/2
        elif loc == 'origin':
            _loc = [0.]*w.naxis
        else:
            _loc = np.atleast_1d(loc)

        return w, _loc

    w_targ, _loc_target = _parse_loc(loc_target, target)
    w_ref, _loc_ref = _parse_loc(loc_reference, reference)

    _loc_target_coo = w_targ.all_pix2world(*_loc_target, 0)
    _loc_target_pix_ref = w_ref.all_world2pix(*_loc_target_coo, 0)

    offset = _loc_target_pix_ref - _loc_ref

    if intify_offset:
        offset = np.around(offset).astype(int)

    if order_xyz:
        return offset
    else:
        return offset[::-1]


def calc_offset_physical(
        target,
        reference=None,
        order_xyz=True,
        ignore_ltm=True,
        intify_offset=False
):
    ''' The pixel offset by physical-coordinate information in referene.

    Parameters
    ----------

    target : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like
        The object to extract header to calculate the position

    reference : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like
        The reference to extract header to calculate the position *from*. If
        `None`, it is basically identical to extract the LTV values from
        `target`.
        Default is `None`.

    order_xyz : bool, optional.
        Whether to return the position in xyz order or not (python order:
        ``[::-1]`` of the former).
        Default is `True`.

    ignore_ltm : bool, optional.
        Whether to assuem the LTM matrix is identity. If it is not and
        ``ignore_ltm=False``, a `NotImplementedError` will be raised, i.e.,
        non-identity LTM matrices are not supported.

    Notes
    -----
    Similar to `calc_offset_wcs`, but with locations fixed to origin (as
    non-identity LTM matrix is not supported). Also, input of WCS is not
    accepted because astropy's wcs module does not parse LTV/LTM from header.
    '''
    def _check_ltm(hdr):
        ndim = hdr["NAXIS"]
        for i in range(ndim):
            for j in range(ndim):
                try:
                    assert float(hdr["LTM{i}_{j}"]) == 1.0*(i == j)
                except (KeyError, IndexError):
                    continue
                except (AssertionError):
                    raise NotImplementedError("Non-identity LTM matrix is not supported.")

            try:  # Sometimes LTM matrix is saved as ``LTMi``, not ``LTMi_j``.
                assert float(target["LTM{i}"]) == 1.0
            except (KeyError, IndexError):
                continue
            except (AssertionError):
                raise NotImplementedError("Non-identity LTM matrix is not supported.")

    do_ref = reference is not None
    _, target = _parse_data_header(target, parse_data=False)
    if do_ref:
        _, reference = _parse_data_header(reference, parse_data=False)

    if not ignore_ltm:
        _check_ltm(target)
        if do_ref:
            _check_ltm(reference)

    ndim = target["NAXIS"]
    ltvs_obj = []
    for i in range(ndim):
        try:
            ltvs_obj.append(target[f"LTV{i + 1}"])
        except (KeyError, IndexError):
            ltvs_obj.append(0)

    if do_ref:
        ltvs_ref = []
        for i in range(ndim):
            try:
                ltvs_ref.append(reference[f"LTV{i + 1}"])
            except (KeyError, IndexError):
                ltvs_ref.append(0)
        offset = np.array(ltvs_obj) - np.array(ltvs_ref)
    else:
        offset = np.array(ltvs_obj)

    if intify_offset:
        offset = np.around(offset).astype(int)

    if order_xyz:
        return offset  # This is already xyz order!
    else:
        return offset[::-1]


def fov_radius(header, unit=u.deg):
    ''' Calculates the rough radius (cone) of the (square) FOV using WCS.

    Parameters
    ----------
    header: Header
        The header to extract WCS information.

    Returns
    -------
    radius: `~astropy.Quantity`
        The radius in degrees
    '''
    w = WCS(header)
    nx, ny = float(header["NAXIS1"]), float(header["NAXIS2"])
    # Rough calculation, so use mode='wcs'
    c1 = SkyCoord.from_pixel(0, 0, wcs=w, origin=0, mode='wcs')
    c2 = SkyCoord.from_pixel(nx, 0, wcs=w, origin=0, mode='wcs')
    c3 = SkyCoord.from_pixel(0, ny, wcs=w, origin=0, mode='wcs')
    c4 = SkyCoord.from_pixel(nx, ny, wcs=w, origin=0, mode='wcs')

    # TODO: Can't we just do ``return max(r1, r2).to(unit)``???
    #   Why did I do this? I can't remember...
    #   2020-11-09 14:29:29 (KST: GMT+09:00) ysBach
    r1 = c1.separation(c3).value / 2
    r2 = c2.separation(c4).value / 2
    r = max(r1, r2) * u.deg
    return r.to(unit)


# TODO: do not load data extension if not explicitly ordered
def wcsremove(
        path=None,
        additional_keys=None,
        extension=None,
        output=None,
        output_verify='fix',
        overwrite=False,
        verbose=True,
        close=True
):
    ''' Remove most WCS related keywords from the header.

    Paramters
    ---------
    additional_keys : list of regex str, optional
        Additional keys given by the user to be 'reset'. It must be in regex
        expression. Of course regex accepts just string, like 'NAXIS1'.

    output: str or Path
        The output file path.
    '''
    # Define header keywords to be deleted in regex:
    re2remove = ['CD[0-9]_[0-9]',  # Coordinate Description matrix
                 'CTYPE[0-9]',      # e.g., 'RA---TAN' and 'DEC--TAN'
                 'C[0-9]YPE[0-9]',  # FOCAS
                 'CUNIT[0-9]',      # e.g., 'deg'
                 'C[0-9]NIT[0-9]',  # FOCAS
                 'CRPIX[0-9]',      # The reference pixels in image coordinate
                 'C[0-9]PIX[0-9]',  # FOCAS
                 # The world cooordinate values at CRPIX[1, 2]
                 'CRVAL[0-9]',
                 'C[0-9]VAL[0-9]',  # FOCAS
                 'CDELT[0-9]',      # with CROTA, older version of CD matrix.
                 'C[0-9]ELT[0-9]',  # FOCAS
                 # The angle between image Y and world Y axes
                 'CROTA[0-9]',
                 'CRDELT[0-9]',
                 'CFINT[0-9]',
                 'RADE[C]?SYS*'     # RA/DEC system (frame)
                 'WCS-ORIG',        # FOCAS
                 'LTM[0-9]_[0-9]',
                 'LTV[0-9]*',
                 'PIXXMIT',
                 'PIXOFFST',
                 'WAT[0-9]_[0-9]',  # For TNX and ZPX, e.g., "WAT1_001"
                 'C0[0-9]_[0-9]',   # polynomial CD by imwcs
                 'PC[0-9]_[0-9]',
                 'P[A-Z]?[0-9]?[0-9][0-9][0-9][0-9][0-9][0-9]',  # FOCAS
                 'PV[0-9]_[0-9]',
                 '[A,B][P]?_[0-9]_[0-9]',  # astrometry.net
                 '[A,B][P]?_ORDER',   # astrometry.net
                 '[A,B][P]?_DMAX',    # astrometry.net
                 'WCS[A-Z]',          # see below
                 'AST_[A-Z]',         # astrometry.net
                 'ASTIRMS[0-9]',      # astrometry.net
                 'ASTRRMS[0-9]',      # astrometry.net
                 'FGROUPNO',          # SCAMP field group label
                 'ASTINST',           # SCAMP astrometric instrument label
                 'FLXSCALE',          # SCAMP relative flux scale
                 'MAGZEROP',          # SCAMP zero-point
                 'PHOTIRMS',          # mag dispersion RMS (internal, high S/N)
                 'PHOTINST',          # SCAMP photometric instrument label
                 'PHOTLINK',          # True if linked to a photometric field
                 'SECPIX[0-9]'
                 ]
    # WCS[A-Z] captures, WCS[DIM, RFCAT, IMCAT, MATCH, NREF, TOL, SEP],
    # but not [IM]WCS, for example. These are likely to have been inserted
    # by WCS updating tools like astrometry.net or WCSlib/WCSTools. I
    # intentionally ignored IMWCS just for future reference.

    if additional_keys is not None:
        re2remove = re2remove + listify(additional_keys)

    # If following str is in comment, suggest it if verbose
    candidate_re = ['wcs', 'axis', 'axes', 'coord', 'distortion', 'reference']
    candidate_key = []

    hdul = fits.open(path)
    hdr = hdul[_parse_extension(extension)].header

    if verbose:
        print("Removed keywords: ", end='')

    for k in list(hdr.keys()):
        com = hdr.comments[k]
        deleted = False
        for re_i in re2remove:
            if re.match(re_i, k) is not None and not deleted:
                hdr.remove(k)
                deleted = True
                if verbose:
                    print(f"{k}", end=' ')
                continue
        if not deleted:
            for re_cand in candidate_re:
                if re.match(re_cand, com):
                    candidate_key.append(k)
    if verbose:
        print('\n')
        if len(candidate_key) != 0:
            print(f"\nFollowing keys may be related to WCS too:\n\t{candidate_key}")

    hdul[extension].header = hdr

    if output is not None:
        hdul.writeto(output, output_verify=output_verify, overwrite=overwrite)

    if close:
        hdul.close()
        return
    else:
        return hdul


# def center_coord(header, skycoord=False):
#     ''' Gives the sky coordinate of the center of the image field of view.
#     Parameters
#     ----------
#     header: astropy.header.Header
#         The header to be used to extract WCS information (and image size)
#     skycoord: bool
#         Whether to return in the astropy.coordinates.SkyCoord object. If
#         `False`, a numpy array is returned.
#     '''
#     wcs = WCS(header)
#     cx = float(header['naxis1']) / 2 - 0.5
#     cy = float(header['naxis2']) / 2 - 0.5
#     center_coo = wcs.wcs_pix2world(cx, cy, 0)

#     if skycoord:
#         return SkyCoord(*center_coo, unit='deg')

#     return np.array(center_coo)


def convert_bit(fname, original_bit=12, target_bit=16):
    ''' Converts a FIT(S) file's bit.

    Notes
    -----
    In ASI1600MM, for example, the output data is 12-bit but since FITS
    standard do not accept 12-bit (but the closest integer is 16-bit), so, for
    example, the pixel values can have 0 and 15, but not any integer between
    these two. So it is better to convert to 16-bit.
    '''
    hdul = fits.open(fname)
    dscale = 2**(target_bit - original_bit)
    hdul[0].data = (hdul[0].data / dscale).astype('int')
    hdul[0].header['MAXDATA'] = (2**original_bit - 1,
                                 "maximum valid physical value in raw data")
    # hdul[0].header['BITPIX'] = target_bit
    # FITS ``BITPIX`` cannot have, e.g., 12, so the above is redundant line.
    hdul[0].header['BUNIT'] = 'ADU'
    hdul.close()
    return hdul


# TODO: add sigma-clipped statistics option (hdr key can be using "SIGC", e.g., SIGCAVG.)
def give_stats(
        item,
        extension=None,
        statsecs=None,
        percentiles=[1, 99],
        N_extrema=None,
        return_header=False,
        nanfunc=False
):
    ''' Calculates simple statistics.

    Parameters
    ----------
    item: array-like, CCDData, HDUList, PrimaryHDU, ImageHDU, or path-like
        The data or path to a FITS file to be analyzed.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    statsecs : str, list of str, optional.
        The FITS-convention section to calculate the statistics.

    percentiles: list-like, optional
        The percentiles to be calculated.

    N_extrema: int, optinoal
        The number of low and high elements to be returned when the whole data
        are sorted. If `None`, it will not be calculated. If ``1``, it is
        identical to min/max values.

    return_header : bool, optional.
        Works only if you gave `item` as FITS file path or
        `~astropy.nddata.CCDData`. The statistics information will be added to
        the header and the updated header will be returned.

    nanfunc : bool, optional.
        Whether to use nan-related functions (e.g., ``np.nanmedian``). If any
        pixel has non-finite value (such as ``np.nan`` or ``np.inf``),
        `nanfunc` must be `True` to get proper statistics at a cost of
        computational speed.

    Returns
    -------
    result : dict
        The dict which contains all the statistics.

    hdr : Header
        The updated header. Returned only if `update_header` is `True` and
        `item` is FITS file path or has `header` attribute (e.g.,
        `~astropy.nddata.CCDData` or `hdu`)

    Notes
    -----
    If you have bottleneck package, the functions from bottleneck will be used.
    Otherwise, numpy is used.

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
    >>> ccd = CCDData.read("bias_bin11.fits", unit='adu')
    >>> _, hdr = (ccd, N_extrema=10, update_header=True)
    >>> ccd.header = hdr
    # To read the stringfied list into python list (e.g., percentiles):
    # >>> import json
    # >>> percentiles = json.loads(ccd.header['percentiles'])
    '''
    data, hdr = _parse_data_header(item, extension=extension)
    if statsecs is not None:
        statsecs = [statsecs] if isinstance(statsecs, str) else list(statsecs)
        data = np.array([data[fitsxy2py(sec)] for sec in statsecs])

    data = data.ravel()

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

    result = dict(
        num=np.size(data),
        min=minf(data),
        max=maxf(data),
        avg=avgf(data),
        med=medf(data),
        std=stdf(data, ddof=1),
        madstd=mad_std(data, ignore_nan=not nanfunc),
        percentiles=percentiles,
        pct=pctf(data, percentiles),
        slices=statsecs
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
            warn(
                f"Extrema overlaps (2*N_extrema ({2*N_extrema}) > N_pix ({result['num']}))"
            )
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

        if statsecs is not None:
            for i, sec in enumerate(statsecs):
                hdr[f"STATSEC{i+1:01d}"] = (sec, "Sections used for statistics")

        if N_extrema is not None:
            if N_extrema > 99:
                warn("N_extrema > 99 may not work properly in header.")
            for i in range(N_extrema):
                hdr[f"STATLO{i+1:02d}"] = (result['ext_lo'][i],
                                           f"Lower extreme values (N_extrema={N_extrema})")
                hdr[f"STATHI{i+1:02d}"] = (result['ext_hi'][i],
                                           f"Upper extreme values (N_extrema={N_extrema})")
        return result, hdr
    return result
