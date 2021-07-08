'''
(Basic) Functions that are completely INDEPENDENT of all other modules of this package.
'''
import ccdproc
import numpy as np
from astropy import units as u
from astropy.time import Time


__all__ = ["MEDCOMB_KEYS_INT", "SUMCOMB_KEYS_INT", "MEDCOMB_KEYS_FLT32", "LACOSMIC_KEYS",
           "circular_mask", "_image_shape", "_offsets2slice",
           "str_now", "change_to_quantity", "binning", "fitsxy2py",
           "quantile_lh", "quantile_sigma"]


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

    Idea copied from
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

    zyx = np.ogrid[slices]
    dist_sq = [((zyx[i] - center[i])**2) for i in range(len(shape))]
    dist_from_center = np.sqrt(np.sum(np.array(dist_sq, dtype=object)))

    mask = dist_from_center <= radius
    return mask


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


def _image_shape(
        shapes, offsets, method='outer', offset_order_xyz=True, intify_offsets=False, pythonize_offsets=True):
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
        The method to calculate the `shape_out`::

          * ``'outer'``: To combine images, so every pixel in `shape_out` has at least 1 image pixel.
          * ``'inner'``: To process only where all the images have certain pixel (fully-overlap).

    offset_order_xyz : bool, optional
        Whether `offsets` are in xyz order. If so, those will be flipped to pythonic order.
        Default: `True`

    Returns
    -------
    _offsets : ndarray
        The *relative* offsets calculated such that at least one image per each dimension must have
        offset of 0.

    shape_out : tuple
        The shape of the array depending on the `method`.
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


def _offsets2slice(
        shapes, offsets, method='outer', shape_order_xyz=False, offset_order_xyz=True, outer_for_stack=True,
        fits_convention=False):
    """ Calculates the slices for each image when to extract overlapping parts.

    Parameters
    ----------
    shapes, offsets : ndarray
        The shape and offset of each image. If multiple images are used, it must have shape of
        ``nimage`` by ``ndim``.

    method : str, optional
        The method to calculate the `shape_out`::

          * ``'outer'``: To combine images, so every pixel in `shape_out` has at least 1 image pixel.
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
        The meaning of it differs depending on `method`::

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
        Whether to return the time at the start of this function and the delta time (`dt`), as well
        as the time information string. If `t_ref` is `None`, `dt` is automatically set to `None`.
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
    ''' Change the non-Quantity object to astropy Quantity or vice versa.

    Parameters
    ----------
    x : object changable to astropy Quantity
        The input to be changed to a Quantity. If a Quantity is given, `x` is changed to the
        `desired`, i.e., ``x.to(desired)``.

    desired : str or astropy Unit
        The desired unit for `x`.

    to_value : bool, optional.
        Whether to return as scalar value. If `True`, just the value(s) of the `desired` unit will be
        returned after conversion.

    Return
    ------
    ux: Quantity

    Note
    ----
    If Quantity, transform to `desired`. If `desired` is `None`, return it as is. If not `Quantity`,
    multiply the `desired`. `desired` is `None`, return `x` with dimensionless unscaled unit.
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
        this function is used for 2-D CCD data. If any of these is given, `order_xyz` is overridden
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


# TODO: add `coord` to select whether image/physical. If physical, header is required.
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


def quantile_lh(a, lq, hq, axis=None, nanfunc=False, interpolation='linear', linterp=None, hinterp=None):
    """Find quantiles for lower and higher values
    Parameters
    ----------
    a : ndarray

    lq, hq : array_like of float
        Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.

    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is to compute the quantile(s)
        along a flattened version of the array.

    nanfunc : bool, optional.
        Whether to use `~np.nanquantile` instead of `~np.qualtile`. Default: `False`.

    interpolation, linterp, hinterp : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional.
        This optional parameter specifies the interpolation method to use when the desired quantile
        lies between two data points ``i < j``:
        * 'linear': ``i + (j - i) * fraction``, where ``fraction`` is the fractional part of the index
          surrounded by ``i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
        To tune the interpolation method for lower and higher quantiles individually, set `linterp` and
        `hinterp` separately. An idea is to use ``linterp='higher', hinterp='lower'`` to estimate the
        robust standard deviation estimate.
    """
    linterp = interpolation if linterp is None else linterp
    hinterp = interpolation if hinterp is None else hinterp

    qfunc = np.nanquantile if nanfunc else np.quantile

    try:
        lq = float(lq)
        hq = float(hq)
    except TypeError:
        raise TypeError("lq and hq must be floats, not array-like.")

    if linterp == hinterp:
        out = qfunc(a, (lq, hq), axis=axis, interpolation=linterp)
    else:
        out_l = qfunc(a, lq, axis=axis, interpolation=linterp)
        out_h = qfunc(a, hq, axis=axis, interpolation=hinterp)
        out = [out_l, out_h]

    return out


def quantile_sigma(a, axis=None, nanfunc=False, interpolation='linear', linterp=None, hinterp=None):
    """ Extract "sigma" (std. dev.) from quantile to avoid bad values.
    Parameters
    ----------
    a : ndarray

    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is to compute the
        quantile(s) along a flattened version of the array.

    nanfunc : bool, optional.
        Whether to use `~np.nanquantile` instead of `~np.quantile`. Default: `False`.

    interpolation, linterp, hinterp : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional.
        This optional parameter specifies the interpolation method to use when the desired quantile
        lies between two data points ``i < j``:
        * 'linear': ``i + (j - i) * fraction``, where ``fraction`` is the fractional part of the index
          surrounded by ``i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
        To tune the interpolation method for lower and higher quantiles individually, set `linterp` and
        `hinterp` separately. An idea is to use ``linterp='higher', hinterp='lower'`` to estimate the
        robust standard deviation estimate.
    """
    low, upp = quantile_lh(a, 0.1587, 0.8413, axis=axis, nanfunc=nanfunc,
                           interpolation=interpolation, linterp=linterp, hinterp=hinterp)
    return np.abs(upp - low)/2


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB):
    return 5 / 10**(gain_dB / 20)


def epadu2dB(gain_epadu):
    return 20 * np.log10(5 / gain_epadu)
