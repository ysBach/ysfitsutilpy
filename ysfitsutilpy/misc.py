'''
Simple mathematical functions that will be used throughout this package. Some
might be useful outside of this package.
'''
import glob
import sys
from collections import Iterable
from pathlib import Path, PosixPath, WindowsPath
from warnings import warn

import bottleneck as bn
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
           "get_size", "is_list_like", "circular_mask", "_image_shape", "_offsets2slice",
           "load_ccd", "write2fits",
           "_parse_data_header", "_parse_image", "_has_header", "_parse_extension",
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

CCDLIKE_TYPES = (CCDData, fits.PrimaryHDU, fits.ImageHDU)

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


def inputs2list(inputs, sort=True, accept_ccdlike=True, path_to_text=False, check_coherency=False):
    ''' Convert glob pattern or list-like of path-like to list of Path

    Parameters
    ----------
    inputs : str, path-like, CCDData, fits.PrimaryHDU, fits.ImageHDU

    '''
    contains_ccdlike = False
    # TODO: if str and startswith("@"), read that file to get fpaths as glob pattern.
    if isinstance(inputs, str):
        # If str, "dir/file.fits" --> [Path("dir/file.fits")]
        #         "dir/*.fits"    --> [Path("dir/file.fits"), ...]
        outlist = glob.glob(inputs)
    elif isinstance(inputs, (PosixPath, WindowsPath)):
        # If Path, ``TOP/"file*.fits"`` --> [Path("top/file1.fits"), ...]
        outlist = glob.glob(str(inputs))
    elif isinstance(inputs, CCDLIKE_TYPES):
        if accept_ccdlike:
            outlist = [inputs]
        else:
            kind = type(inputs)
            raise TypeError(f"{kind} is given as `inputs`. Turn off accept_ccdlike or use path-like.")
    elif is_list_like(inputs):
        type_ref = type(inputs[0])
        outlist = []
        for i, item in enumerate(inputs):
            if check_coherency and (type(item) != type_ref):
                raise TypeError(f"The 0-th item has {type_ref} while {i}-th has {type(item)}.")
            if isinstance(item, CCDLIKE_TYPES):
                contains_ccdlike = True
                if accept_ccdlike:
                    outlist.append(item)
                else:
                    kind = type(item)
                    raise TypeError(
                        f"{kind} is given in the {i}-th element. Turn off accept_ccdlike or use path-like."
                    )
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
    # Important mark as seen *before* entering recursion to gracefully handle self-referential objects
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


def is_list_like(obj):
    ''' Direct copy from pandas, with slight modification
    https://github.com/pandas-dev/pandas/blob/bdb00f2d5a12f813e93bc55cdcd56dcb1aae776e/pandas/_libs/lib.pyx#L1026
    '''
    return(
        isinstance(obj, Iterable)
        # we do not count strings/unicode/bytes as list-like
        and not isinstance(obj, (str, bytes))
        # exclude zero-dimensional numpy arrays, effectively scalars
        and not (isinstance(obj, np.ndarray) and obj.ndim == 0)
        # exclude sets if allow_sets is False
        # and not (allow_sets is False and isinstance(obj, abc.Set))
    )


def circular_mask(shape, center=None, radius=None, center_xyz=True):
    ''' Creates an N-D circular (circular, sphereical, ...) mask.
    Parameters
    ----------
    shape : tuple
        The pythonic shape (not xyz order).

    center : tuple, None, optional.
        The center of the circular mask. If `None`, the central position is used.

    radius : float, None, optional.
        The radius of the mask. If `None`, the distance to the closest edge of the image is used.

    center_xyz : bool, optional.
        Whether the center is in xyz order.

    Direct copy from
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''
    if center is None:  # use the middle of the image
        center = [int(npix/2) for npix in shape[::-1]]

    if center_xyz:
        center = center[::-1]

    shape = np.array(shape)
    center = np.array(center)

    if radius is None:  # use the smallest distance between the center and image walls
        radius = np.min([center, shape - center])

    slices = tuple([slice(None, npix, None) for npix in shape])

    ZYX = np.ogrid[slices]
    dist_sq = [(ZYX[i] - center[i])**2 for i in range(len(shape))]
    dist_from_center = np.sqrt(np.sum(dist_sq))

    mask = dist_from_center <= radius
    return mask


def _parse_data_header(ccdlike, extension=None, parse_data=True, parse_header=True, copy=True):
    '''Parses data and header and return them separately after copy.

    Paramters
    ---------
    ccdlike : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like
        The object to be parsed into data and header.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    parse_data, parse_header : bool, optional.
        Because this function uses ``.copy()`` for safety, it may take a bit of time if this function
        is used iteratively. One then can turn off one of these to ignore either data or header part.

    Returns
    -------
    data : ndarray
        The data part of the input ``ccdlike``.

    hdr : Header, None
        The header if header exists; otherwise, `None` is returned.

    Notes
    -----
    _parse_data_header and _parse_image have different purposes: _parse_data_header is to get a quick
    copy of the data and/or header, especially to CHECK if it has header, while _parse_image is to deal
    mainly with the data (and has options to return as CCDData).
    '''
    if isinstance(ccdlike, (CCDData, fits.PrimaryHDU, fits.ImageHDU)):
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
    elif isinstance(ccdlike, np.ndarray):
        if parse_data:
            data = ccdlike.copy() if copy else ccdlike
        else:
            data = None
        hdr = None  # regardless of parse_header
    elif isinstance(ccdlike, fits.Header):
        data = None  # regardless of parse_data
        if parse_header:
            hdr = ccdlike.copy() if copy else ccdlike
        else:
            hdr = None
    elif isinstance(ccdlike, fitsio.FITSHDR):
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
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")`` can be PosixPath without error...
            extension = _parse_extension(extension) if parse_data or parse_header else 0
            # fits.getheader is ~ 10-20 times faster than load_ccd.
            # 2020-11-09 16:06:41 (KST: GMT+09:00) ysBach
            try:
                if parse_header:
                    hdu = fits.open(Path(ccdlike), memmap=False)[extension]
                    # No need to copy because they've been read (loaded) for the first time here.
                    data = hdu.data if parse_data else None
                    hdr = hdu.header if parse_header else None
                else:
                    if isinstance(extension, tuple):
                        data = fitsio.read(Path(ccdlike), ext=extension[0], extver=extension[1])
                    else:
                        data = fitsio.read(Path(ccdlike), ext=extension)
                    hdr = None
            except TypeError:
                raise TypeError(f"ccdlike type ({type(ccdlike)}) is not acceptable to find header and data.")

    return data, hdr


def _parse_image(ccdlike, extension=None, name=None, force_ccddata=False, prefer_ccddata=False,
                 use_ccddata_if_path=True):
    '''Parse and return input image as desired format (ndarray or CCDData)
    Parameters
    ----------
    ccdlike : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
        The "image" that will be parsed. A string that can be converted to float (``float(im)``)
        will be interpreted as numbers; if not, it will be interpreted as a path to the FITS file.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    force_ccddata: bool, optional.
        To force the retun im as `~astropy.nddata.CCDData` object. This is useful when error
        calculation is turned on.

    prefer_ccddata: bool, optional.
        Mildly use `~astropy.nddata.CCDData`, i.e., return `~astropy.nddata.CCDData` only if ``im`` was
        `~astropy.nddata.CCDData`, HDU object, or Path-like to a FITS file, but **not** if it was
        ndarray or numbers. If `False` (default), it prefers ndarray.

    use_ccddata_if_path : bool, optional.
        Whether to load the full CCD information (using `~astropy.nddata.CCDData`). If `False`,
        `fitsio.read` will be used without parsing header, significantly increasing the IO speed.
        Default is `True`.

    Returns
    -------
    new_im : ndarray or CCDData
        Depending on the options ``force_ccddata`` and ``prefer_ccddata``.

    imname : str
        The name of the image.

    imtype : str
        The type of the image.

    Notes
    -----
    _parse_data_header and _parse_image have different purposes: _parse_data_header is to get a quick
    copy of the data and/or header, especially to CHECK if it has header, while _parse_image is to deal
    mainly with the data (and has options to return as CCDData).
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
            return CCDData(data=hdu.data, header=hdu.header, unit=unit)
            # The two lines above took ~ 5 us and 10-30 us for the simplest header and 1x1 pixel data
            # case (regardless of BUNIT exists), on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz;
            # 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
        else:
            return hdu.data

    ccd_kw = dict(force_ccddata=force_ccddata, prefer_ccddata=prefer_ccddata)
    has_no_name = name is None
    extension, extstr = __extract_extension(extension)
    imname = f"User-provided {ccdlike.__class__.__name__}{extstr}" if has_no_name else name

    if isinstance(ccdlike, CCDData):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        new_im = ccdlike.copy() if (force_ccddata or prefer_ccddata) else ccdlike.data.copy()
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
        new_im = CCDData(data=ccdlike.copy(), unit='adu') if force_ccddata else ccdlike
        imtype = "ndarray"
    else:
        try:  # IF number (ex: im = 1.3)
            # force_ccddata: CCDData // prefer_ccddata: number // else: number
            imname = f"{imname} {ccdlike}" if has_no_name else name
            _im = float(ccdlike)
            new_im = CCDData(data=_im, unit='adu') if force_ccddata else np.asarray(_im)
            imtype = "num"  # imname can be "int", "float", "str", etc, so imtype might be useful.
        except (ValueError, TypeError):
            try:  # IF path-like
                # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
                fpath = Path(ccdlike)
                imname = f"{str(fpath)}{extstr}" if has_no_name else name
                # set redundant extensions to None so that only the part specified by ``extension`` be loaded:
                new_im = load_ccd(fpath, extension, ccddata=use_ccddata_if_path,
                                  extension_uncertainty=None, extension_mask=None)
                imtype = "path"
            except TypeError:
                raise TypeError("input must be CCDData-like, ndarray, path-like (to FITS), or a number.")

    return new_im, imname, imtype


def _has_header(ccdlike, extension=None, open_if_file=True):
    '''Checks if the object has header; similar to _parse_data_header.

    Paramters
    ---------
    ccdlike : CCDData, PrimaryHDU, ImageHDU, HDUList, ndarray, number-like, path-like
        The object to be parsed into data and header.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used. Used only if ``ccdlike`` is
        HDUList or path-like.

    open_if_file : bool, optional.
        Whether to open the file to check if it has a header when ``ccdlike`` is path-like. Any
        FITS file has a header, so this means it will check the existence and validity of the file. If
        set to `False`, all path-like input will return `False` because the path itself has no header.

    Notes
    -----
    It first checks if the input is one of ``(CCDData, fits.PrimaryHDU, fits.ImageHDU)``, then if
    ``fits.HDUList``, then if ``np.ndarray``, then if number-like, and then finally if path-like.
    Although this has a bit of disadvantage considering we may use file-path for most of the time, the
    overhead is only ~ 1 us, tested on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)].
    '''
    hashdr = True
    if isinstance(ccdlike, (CCDData, fits.PrimaryHDU, fits.ImageHDU)):  # extension not used
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
    elif isinstance(ccdlike, np.ndarray):
        hashdr = False
    else:
        try:  # if number-like
            _ = float(ccdlike)
            hashdr = False
        except (ValueError, TypeError):  # if path-like
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")`` can be PosixPath without error...
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

    This supports several different styles of extension selection.  See the :func:`getdata()`
    documentation for the different possibilities.

    Direct copy from astropy, but removing "opening HDUList" part
    https://github.com/astropy/astropy/blob/master/astropy/io/fits/convenience.py#L988

    This is essential for fits_ccddata_reader, because it only has ``hdu``, not all three of ext,
    extname, and extver.

    Note
    ----
    %timeit yfu._parse_extension()
    # 1.52 µs +- 69.3 ns per loop (mean +- std. dev. of 7 runs, 1000000 loops each)
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


def _regularize_offsets(offsets, offset_order_xyz=True, intify_offsets=False):
    """ Makes offsets all non-negativevg and relative to each other.
    """
    _offsets = np.atleast_2d(offsets)
    if offset_order_xyz:
        _offsets = np.flip(_offsets, -1)

    # _offsets = np.max(_offsets, axis=0) - _offsets
    _offsets = _offsets - np.min(_offsets, axis=0)
    # This is the convention to follow IRAF (i.e., all of offsets > 0.)
    if intify_offsets:
        _offsets = np.around(_offsets).astype(int)

    return _offsets


def _image_shape(shapes, offsets, method='outer', offset_order_xyz=True, intify_offsets=False,
                 pythonize_offsets=True):
    '''shapes and offsets must be in the order of python/numpy (i.e., z, y, x order).

    Paramters
    ---------
    shapes : ndarray
        The shapes of the arrays to be processed. It must have the shape of ``nimage`` by ``ndim``. The
        order of shape must be pythonic (i.e., ``shapes[i] = image[i].shape``, not in the xyz order).

    offsets : ndarray
        The offsets must be ``(cen_i - cen_ref) + const`` format, i.e., an offset is the position of
        the target frame (position can be, e.g., origin or center) in the coordinate of the reference
        frame, with a possible non-zero constant offset applied. It must have the shape of ``nimage``
        by ``ndim``.

    method : str, optional
        The method to calculate the ``shape_out``::

          * ``'outer'``: To combine images, so every pixel in ``shape_out`` has at least 1 image pixel.
          * ``'inner'``: To process only where all the images have certain pixel (fully-overlap).

    offset_order_xyz : bool, optional
        Whether ``offsets`` are in xyz order. If so, those will be flipped to pythonic order.
        Default: `True`

    Returns
    -------
    _offsets : ndarray
        The *relative* offsets calculated such that at least one image per each dimension must have
        offset of 0.

    shape_out : tuple
        The shape of the array depending on the ``method``.
    '''
    _offsets = _regularize_offsets(offsets, offset_order_xyz=offset_order_xyz, intify_offsets=intify_offsets)

    if method == 'outer':
        shape_out = np.around(np.max(np.array(shapes) + _offsets, axis=0)).astype(int)
        # print(_offsets, shapes, shape_out)
    # elif method == 'stack':
    #     shape_out_comb = np.around(np.max(np.array(shapes) + _offsets, axis=0)).astype(int)
    #     shape_out = (len(shapes), *shape_out_comb)
    elif method == 'inner':
        lower_bound = np.max(_offsets, axis=0)
        upper_bound = np.min(_offsets + shapes, axis=0)
        npix = upper_bound - lower_bound
        shape_out = np.around(npix).astype(int)
        if np.any(npix < 0):
            raise ValueError(f"There doesn't exist fully-overlapping pixel! Naïve output shape={shape_out}.")
        # print(lower_bound, upper_bound, shape_out)
    else:
        raise ValueError("method unacceptable (use one of 'inner', 'outer').")

    if offset_order_xyz and not pythonize_offsets:  # reverse _offsets to original xyz order
        _offsets = np.flip(_offsets, -1)

    return _offsets, tuple(shape_out)


def _offsets2slice(shapes, offsets, method='outer', shape_order_xyz=False, offset_order_xyz=True,
                   outer_for_stack=True, fits_convention=False):
    """ Calculates the slices for each image when to extract overlapping parts.

    Parameters
    ----------
    shapes, offsets : ndarray
        The shape and offset of each image. If multiple images are used, it must have shape of
        ``nimage`` by ``ndim``.

    method : str, optional
        The method to calculate the ``shape_out``::

          * ``'outer'``: To combine images, so every pixel in ``shape_out`` has at least 1 image pixel.
          * ``'inner'``: To process only where all the images have certain pixel (fully-overlap).

    shape_order_xyz, offset_order_xyz : bool, optional.
        Whether the order of the shapes or offsets are in xyz or pythonic. Shapes are usually in
        pythonic as it is obtained by ``image_data.shape``, but offsets are often in xyz order (e.g.,
        if header ``LTVi`` keywords are loaded in their alpha-numeric order; or you have used
        `~calc_offset_wcs` or `~calc_offset_physical` with default ``order_xyz=True``).
        Default is `False` and `True`, respectively.

    outer_for_stack : bool, optional.
        If `True`(default), the output slice is the slice in tne ``N+1``-D array, which will be
        constructed before combining them along ``axis=0``. That is, ``comb =
        np.nan*np.ones(_image_shape(shapes, offsets, method='outer'))`` and ``comb[slices[i]] =
        images[i]``. Then a median combine, for example, is done by ``np.nanmedian(comb, axis=0)``.
        If ``stack_outer=False``, ``slices[i]`` will be ``slices_with_stack_outer_True[i][1:]``.

    fits_convention : bool, optional.
        Whether to return the slices in FITS convention (xyz order, 1-indexing, end index included). If
        `True` (default), returned list contains str; otherwise, slice objects will be contained.

    Returns
    -------
    slices : list of str or list of slice
        The meaning of it differs depending on ``method``::

          * ``'outer'``: the slice of the **output** array where the i-th image should fit in.
          * ``'inner'``: the slice of the **input** array (image) where the overlapping region resides.

    Example
    -------
    >>>

    Note
    ----

    """
    _shapes = np.atleast_2d(shapes)
    if shape_order_xyz:
        _shapes = np.flip(_shapes, -1)

    _offsets = _regularize_offsets(offsets, offset_order_xyz=offset_order_xyz, intify_offsets=True)

    if _shapes.ndim != 2 or _offsets.ndim != 2:
        raise ValueError("Shapes and offsets must be at most 2-D.")

    if _shapes.shape != _offsets.shape:
        raise ValueError("shapes and offsets must have the identical shape.")

    if method == 'outer':
        starts = _offsets
        stops = _offsets + _shapes
        if outer_for_stack:
            _initial_tmp = lambda i: [f"{i + 1}:{i + 1}"] if fits_convention else [slice(i, i + 1, None)]
        else:
            _initial_tmp = lambda i: []  # initialized empty list regardless of argument
    elif method == 'inner':
        offmax = np.max(_offsets, axis=0)
        if np.any(np.min(_shapes + _offsets, axis=0) <= offmax):
            raise ValueError("At least 1 frame has no overlapping pixel with all others. "
                             + "Check if there's any overlapping pixel for images for the given offsets.")

        # 1-D array +/- 2-D array: the former 1-D array is broadcast s.t. it is "tile"d along axis=-1.
        starts = offmax - _offsets
        stops = np.min(_offsets + _shapes, axis=0) - _offsets
        _initial_tmp = lambda i: []  # initialized empty list regardless of argument
    else:
        raise ValueError("method unacceptable (use one of 'inner', 'outer').")

    slices = []
    for image_i, (start, stop) in enumerate(zip(starts, stops)):
        # NOTE: starts/stops are all in pythonic index
        tmp = _initial_tmp(image_i)
        # print(tmp)
        for start_i, stop_i in zip(start, stop):  # i = coordinate, (z y x) order
            if fits_convention:
                tmp.append(f"{start_i + 1:d}:{stop_i:d}")
            else:
                tmp.append(slice(start_i, stop_i, None))
            # print(tmp)

        if fits_convention:
            slices.append('[' + ','.join(tmp[::-1]) + ']')  # order is opposite!
        else:
            slices.append(tmp)

    return slices


def load_ccd(path, extension=None, ccddata=True, as_ccd=True, use_wcs=True, unit=None,
             extension_uncertainty="UNCERT", extension_mask='MASK', extension_flags=None,
             load_primary_only_fitsio=True, return_full_fitsio=False,
             key_uncertainty_type='UTYPE', memmap=False, **kwd):
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
        Used only if ``ccddata=True``.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a unit for the image in the
        FITS header (the keyword ``BUNIT`` is used as the unit, if present), this argument is used for
        the unit.
        Default is `None`.
        Used only if ``ccddata=True``.

        .. note::
            The behavior differs from astropy's original fits_ccddata_reader: If no ``BUNIT`` is found
            and ``unit`` is `None`, ADU is assumed.

    load_primary_only_fitsio : bool, optional.
        Whether to ignore uncertainty, mask, and flags extensions when using fitsio (i.e., when
        ``use_ccd=False``). This is `True` by default, because that's the most common usage for fitsio.

    return_full_fitsio : bool, optional.
        Whether to return full (``data, unc, mask, flag``) even when ``load_primary_only_fitsio=True``
        and ``extension_uncertainty`` is `None` and ``extension_mask`` is `None`, which can be
        convenient for API design.
        Default is `False`.

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
        Used only if ``ccddata=True``.

        ..warning::
            If ``ccddata=False`` and ``load_primary_only_fitsio=False``, the uncertainty type by
            ``key_uncertainty_type`` will be completely ignored.

    memmap : bool, optional
        Is memory mapping to be used? This value is obtained from the configuration item
        ``astropy.io.fits.Conf.use_memmap``.
        Default is `False` (opposite of astropy).
        Used only if ``ccddata=True``.

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
    try:
        path = Path(path)
    except TypeError:
        raise TypeError(f"You must provide Path-like, not {type(path)}.")

    if ccddata and as_ccd:  # if at least one of these is False, it uses fitsio.
        extension = _parse_extension(extension)
        extension_uncertainty = _parse_extension(extension_uncertainty)
        extension_mask = _parse_extension(extension_mask)
        extension_flag = None if extension_flags is None else _parse_extension(extension_flags)
        # If not None, this happens: NotImplementedError: loading flags is currently not supported.

        reader_kw = dict(hdu=extension, hdu_uncertainty=extension_uncertainty, hdu_mask=extension_mask,
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
        extension = _parse_extension(extension)
        # Use fitsio and only load the data as soon as possible. This is much quicker than astropy's getdata

        def _read_by_fitsio(_hdul, _path, _extension):
            try:
                if is_list_like(_extension):
                    # length == 2 is already checked in _parse_extension.
                    arr = _hdul[_extension[0], _extension[1]].read()
                else:
                    arr = _hdul[_extension].read()
            except OSError:
                raise ValueError(f"Extension `{_extension}` is not found (file: {_path})")

            return arr

        hdul = fitsio.FITS(path)
        if load_primary_only_fitsio and extension_uncertainty is None and extension_mask is None:
            data = _read_by_fitsio(hdul, path, extension)
            hdul.close()
            if return_full_fitsio:
                return data, None, None, None
            else:
                return data

        else:
            data = _read_by_fitsio(hdul, path, extension)
            try:  # Read uncertainty if exists
                e_u = _parse_extension(extension_uncertainty)
                unc = _read_by_fitsio(hdul, path, e_u)
            except (OSError, ValueError):  # if the extension is not found
                unc = None
            try:  # Read uncertainty if exists
                e_m = _parse_extension(extension_mask)
                mask = _read_by_fitsio(hdul, path, e_m)
            except (OSError, ValueError):  # if the extension is not found
                mask = None

            flag = None  # FIXME: add this line when CCDData starts to support flags.
            hdul.close()
            return data, unc, mask, flag


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
        The keyword arguements to write FITS file by `~astropy.nddata.fits_data_writer`, such as
        ``output_verify=True``, ``overwrite=True``.
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


def str_now(precision=3, fmt="{:.>72s}", t_ref=None, dt_fmt="(dt = {:.3f} s)", return_time=False):
    ''' Get stringfied time now in UT ISOT format.

    Parameters
    ----------
    precision : int, optional.
        The precision of the isot format time.

    fmt : str, optional.
        The Python 3 format string to format the time. Examples::

          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in parentheses ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with ``_``.

    t_ref : Time, optional.
        The reference time. If not `None`, delta time is calculated.

    dt_fmt : str, optional.
        The Python 3 format string to format the delta time.

    return_time : bool, optional.
        Whether to return the time at the start of this function and the delta time (``dt``), as well
        as the time information string. If ``t_ref`` is `None`, ``dt`` is automatically set to `None`.
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


def binning(arr, factor_x=None, factor_y=None, factors=None, order_xyz=True, binfunc=np.mean, trim_end=False):
    ''' Bins the given arr frame.

    Paramters
    ---------
    arr: 2d array
        The array to be binned

    factor_x, factor_y: int or None, optional.
        The binning factors in x, y direction. This is left as legacy and for clarity, because mostly
        this function is used for 2-D CCD data. If any of these is given, ``order_xyz`` is overridden
        as `True`.

    factors : list-like of int, optional.
        The factors in pythonic axis order (``order_xyz=False``) or in the xyz order
        (``order_xyz=True``). If any of the tuple is `None`, that will be replaced by the size of the
        array along that axis, i.e., collapse along that axis.

    binfunc : funciton object
        The function to be applied for binning, such as ``np.sum``, ``np.mean``, and ``np.median``.

    trim_end: bool
        Whether to trim the end of x, y axes such that binning is done without error.

    Note
    ----
    This kind of binning is ~ 20-30 to upto 10^5 times faster than astropy.nddata's block_reduce:

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
    Tested on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4),
    Radeon Pro 560X (4GB)]
    '''
    # def binning(arr, factor_x=1, factor_y=1, binfunc=np.mean, trim_end=False):
    #     binned = arr.copy()
    #     if trim_end:
    #         ny_orig, nx_orig = binned.shape
    #         iy_max = ny_orig - (ny_orig % factor_y)
    #         ix_max = nx_orig - (nx_orig % factor_x)
    #         binned = binned[:iy_max, :ix_max]
    #     ny, nx = binned.shape
    #     nby = ny // factor_y
    #     nbx = nx // factor_x
    #     binned = binned.reshape(nby, factor_y, nbx, factor_x)
    #     binned = binfunc(binned, axis=(-1, 1))
    #     return binned

    binned = arr.copy()

    if factor_x is not None or factor_y is not None:
        factors = (factor_x, factor_y)
        order_xyz = True

    if factors is None:
        factors = np.ones(arr.ndim)
    else:
        factors = np.array(factors).ravel()
        for i, f in enumerate(factors):
            if f is None:
                factors[i] = arr.shape[i]

    if order_xyz:
        factors = factors[::-1]  # convert back to python order

    if trim_end:
        n_orig = binned.shape
        i_max = n_orig - (n_orig % factors)
        slices = [slice(None, im, None) for im in i_max]
        binned = binned[slices]

    npix = binned.shape
    nbin = npix // factors
    nbin[nbin == 0] = 1
    newshape = []
    for nbin_i, factor_i in zip(nbin, factors):
        newshape.append(nbin_i)
        newshape.append(factor_i)

    binned = binned.reshape(newshape)
    funcaxis = np.arange(1, binned.ndim + 1, 2).astype(int)
    binned = binfunc(binned, axis=tuple(funcaxis))
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
    sl = [ccdproc.utils.slices.slice_from_string(sect, fits_convention=True) for sect in fits_sections]
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
    >>> ccd = CCDData.read("bias_bin11.fits", unit='adu')
    >>> _, hdr = (ccd, N_extrema=10, update_header=True)
    >>> ccd.header = hdr
    To read the stringfied list into python list (e.g., percentiles):
    >>> import json
    >>> percentiles = json.loads(ccd.header['percentiles'])
    '''
    if is_list_like(item):
        data = np.array(item)
        hdr = None
    else:
        try:  # if Path-like, replace ``item`` to ndarray or CCDData
            fpath = Path(item)
            if return_header:
                item = CCDData.read(fpath, extension)
            else:
                item = fitsio.FITS(fpath)[extension].read()
        except (TypeError, ValueError):
            pass

        data, hdr = _parse_data_header(item)

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
