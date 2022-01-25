
__all__ = ["REJECT_PARAMETERS_COMMON",
           'REJECT_PARAMETERS_SIGMA',
           "REJECT_RETURNS_SIGMA",
           "NDCOMB_NOT_IMPLEMENTED",
           "NDCOMB_PARAMETERS_COMMON", "NDCOMB_RETURNS_COMMON",
           "OFFSETS_LONG", "OFFSETS_SHORT", "IMCOMBINE_LINK"]


def _fix(content, indent=0):
    ''' From astropy
    https://github.com/astropy/astropy/blob/a6a350570aec6600eb907fe9903585dd18dc0874/astropy/wcs/docstrings.py#L12
    '''
    lines = content.split('\n')
    indent = '\n' + ' ' * indent
    return indent.join(lines)


def REJECT_PARAMETERS_COMMON(indent=0):
    return _fix(
        """
mask : ndarray, optional.
    The initial mask provided prior to any rejection. `arr` and `mask` must have the identical
    shape.

axis : int, optional.
    The axis to combine the image.

full : bool, optional.
    Whether to return full results. See Return.
"""
)


def REJECT_PARAMETERS_SIGMA(indent=0):
    return _fix(
        """
sigma : float-like, optional.
    The sigma-factors to be muiltiplied to the sigma values. Overridden by `sigma_lower` and/or
    `sigma_upper`, if input.

sigma_lower : float or `None`, optional
    The number of standard deviations to use as the lower bound for the clipping limit. If `None` then
    the value of `sigma` is used. The default is `None`.

sigma_upper : float or `None`, optional
    The number of standard deviations to use as the upper bound for the clipping limit. If `None` then
    the value of `sigma` is used. The default is `None`.

maxiters : int, optional.
    The maximum number of iterations to do the rejection. It is silently converted to int if it is not.

ddof : int, optional.
    The delta-degrees of freedom (see `numpy.std`). It is silently converted to int if it is not.

nkeep : float or int, optional.
    The minimum number of pixels that should be left after rejection. If ``nkeep < 1``, it is regarded
    as fraction of the total number of pixels along the axis to combine.

maxrej : float or int, optional.
    The maximum number of pixels that can be rejected during the rejection. If ``maxrej < 1``, it is
    regarded as fraction of the total number of pixels along the axis to combine.

cenfunc : str, optional.
    The centering function to be used.

        * median if  `cenfunc` in ``{'med', 'medi', 'median'}``
        * average if `cenfunc` in ``{'avg', 'average', 'mean'}``
        * lower median if `cenfunc` in ``{'lmed', 'lmd', 'lmedian'}``

    The lower median means the median which takes the lower value when even number of data is left.
    This is suggested to be robust against cosmic-ray hit according to IRAF IMCOMBINE manual.

irafmode : bool, optional.
        Whether to use IRAF-like pixel restoration scheme. Default is `True`.

""", indent)


def REJECT_RETURNS_SIGMA(indent=0):
    return _fix("""
o_mask : ndarray of bool
    The mask of the same shape as `arr` and `mask`.

o_low, o_upp : ndarray of `dtype`
    Returned only if ``full = True``. The lower and upper bounds used for sigma clipping. Data with
    ``(arr < o_low) | (o_upp < arr)`` are masked. Shape of ``arr.shape[1:]``.

o_nit : ndarray of int or int
    Returned only if `full` is `True`. The number of iterations until it is halted.

o_code : ndarray of uint8
    Returned only if `full` is `True`. Each element is a ``uint8`` value with::

        *      (0): maxiters reached without any flag below
        * 1-th (1): maxiters == 0 (no iteration happened)
        * 2-th (2): iteration finished before maxiters reached
        * 3-th (4): remaining ndata < nkeep reached
        * 4-th (8): rejected ndata > maxrej reached

    The code of 10 is, for example, 1010 in binary, so the iteration finished before `maxiters`
    (2-th flag) because pixels more than `maxrej` are rejected (4-th flag).
""", indent)


def NDCOMB_NOT_IMPLEMENTED(indent=0):
    return _fix(
        '''
.. warning::
    Few functionalities are not implemented yet:

        #. ``blank`` option
        #. ``logfile``
        #. ``statsec`` with input, output, overlap
        #. ``weight``
        #. ``scale_sample``, ``zero_sample``
        #. ``"mode"`` for ``scale``, ``zero``, ``weight``
        #. ``memlimit`` behaviour
''', indent)


def NDCOMB_PARAMETERS_COMMON(indent=0):
    return _fix('''
thresholds : 2-float list-like, optional.
    The thresholds ``(lower, upper)`` applied to all images before any rejection/combination. Default
    is no thresholding, ``(-np.inf, +np.inf)``. One possible usage is to replace bad pixel to very
    large or small numbers and use this thresholding.

zero : str or 1-d array
    The *zero* value to subtract from each image *after* thresholding, but *before* scaling/offset
    shifting/rejection/combination. If an array, it is directly subtracted from each image, (so it must
    have size identical to the number of images). If `str`, the zero-level is:

        - **Average** if ``'avg'|'average'|'mean'``
        - **Median** if ``'med'|'medi'|'median'``
        - **Sigma-clipped average** if
          ``'avg_sc'|'average_sc'|'mean_sc'``
        - **Sigma-clipped median** if
          ``'med_sc'|'medi_sc'|'median_sc'``

    For options for sigma-clipped statistics, see `zero_kw`.

    .. note::
        By using `zero` of ``"med_sc"``, the user can crudely subtract sky value from each frame before
        combining.

scale : str or 1-d array
    The way to scale each image *after* thresholding/zeroing, but *before* offset
    shifting/rejection/combination. If an array, it is directly understood as the **raw scales**, and
    it must have size identical to the number of images. If `str`, the raw scale is:

        - **Exposure time** (``exposure_key`` in header of each FITS)
          if ``'exp'|'expos'|'exposure'|'exptime'``
        - **Average** if ``'avg'|'average'|'mean'``
        - **Median** if ``'med'|'medi'|'median'``
        - **Sigma-clipped average** if
          ``'avg_sc'|'average_sc'|'mean_sc'``
        - **Sigma-clipped median** if
          ``'med_sc'|'medi_sc'|'median_sc'``

    The true scale is obtained by ``scales / scales[0]`` if `scale_to_0th` is `True`, following
    IRAF's convention. Otherwise, the absolute value from the raw scale will be used. For options for
    sigma-clipped statistics, see `scale_kw`.

    .. note::
        Using ``scale="avg_sc", scale_to_0th=False`` is useful for flat combining.

zero_to_0th : bool, optional.
    Whether to re-base the zero values such that all images have identical zero values as that of the
    0-th image (in python, ``zero - zero[0]``). This is the behavior of IRAF, so `zero_to_0th` is
    `True` by default.

scale_to_0th : bool, optional.
    Whether to re-scale the scales such that ``scale[0]`` is unity (in python, ``scale/scale[0]``).
    This is the behavior of IRAF, so `scale_to_0th` is `True` by default.

zero_section, scale_section : str, optional.
    The sections used for zeroing and scaling. These must be in FITS section
    format, and are the sections **AFTER** trimming based on `trimsec`.

zero_kw, scale_kw : dict
    Used only if `scale` or `zero` are sigma-clipped mean, median, etc (ending with ``_sc`` such as
    ``median_sc``, ``avg_sc``). The keyword arguments for `astropy.stats.sigma_clipped_stats`. By
    default, ``std_ddof=1`` (note that `~astropy.stats.sigma_clipped_stats` has default
    ``std_ddof=0``.)

    .. note::
        #. If `axis` is specified, it will be ignored.

        #. Sigma-clipping in astropy has *cumulative rejection* algorithm by default. This may give
           unwanted results especially when `sigma` in this `zero_kw` or `scale_kw` is small
           (around 1), but this is not likely a problem since both zero and scale will be determined
           from all the pixels, which is usually more than an order of a million.

sigma : 2-float list-like, optional.
    The sigma-factors to be used for sigma-clip rejeciton in ``(sigma_lower, sigma_upper)``. Defaults
    to ``(3, 3)``, which means 3-sigma clipping from the "sigma" values determined by the method
    specified by `reject`. If a single float, it will be used for both the lower and upper values.

maxiters : int, optional.
    The maximum number of iterations to do the rejection (for sigma-clipping). It is silently converted
    to `int` if it is not.

ddof : int, optional.
    The delta-degrees of freedom (see `numpy.std`). It is silently converted to `int` if it is not.

nkeep : float or int, optional.
    The minimum number of pixels that should be left after rejection. If ``nkeep < 1``, it is regarded
    as fraction of the total number of pixels along the axis to combine. This corresponds to *positive*
    `nkeep` parameter of IRAF `IMCOMBINE`_. If number of remaining non-nan value is fewer than
    `nkeep`, the masks at that position will be reverted to the previous iteration, and rejection
    code will be added by number 4.

maxrej : float or int, optional.
    The maximum number of pixels that can be rejected during the rejection. If ``maxrej < 1``, it is
    regarded as fraction of the total number of pixels along the axis to combine. This corresponds to
    *negative* `nkeep` parameter of IRAF `IMCOMBINE`_. In IRAF, only one of `nkeep` and `maxrej`
    can be set. If number of rejected pixels at a position exceeds `maxrej`, the masks at that
    position will be reverted to the previous iteration, and rejection code will be added by number 8.

cenfunc : str, optional.
    The centering function to be used in rejection algorithm.

        - median if  ``'med'|'medi'|'median'``
        - average if ``'avg'|'average'|'mean'``
        - lower median if ``'lmed'|'lmd'|'lmedian'``

    For lower median, see note in `combine`.

n_minmax : 2-float or 2-int list-like, optional.
    The number of low and high pixels to be rejected by the "minmax" algorithm. These numbers are
    converted to fractions of the total number of input images so that if no rejections have taken
    place the specified number of pixels are rejected while if pixels have been rejected by masking,
    thresholding, or non-overlap, then the fraction of the remaining pixels, truncated to an integer,
    is used.

rdnoise, gain, snoise : float, optional.
    The readnoise of the detector in the unit of electrons, electron gain of the detector in the unit
    of elctrons/DN (or electrons/ADU), and sensitivity noise as a fraction. Used only if
    ``reject="ccdclip"`` and/or ``combine="nmodel"``.

    The variance of a single pixel in an image when these are used,

    .. math::
        V_\mathrm{DN}
        = ( \mathtt{rdnoise}/\mathtt{gain} )^2
        + \mathrm{DN}/\mathtt{gain}
        + ( \mathtt{snoise} * \mathrm{DN} )^2

    .. math::
        V_\mathrm{electron}
        = (\mathtt{rdnoise})^2
        + (\mathtt{gain} * \mathrm{DN})^2
        + (\mathtt{snoise} * \mathtt{gain} * \mathrm{DN})^2

pclip : float, optional.
    The parameter for ``reject="pclip"``. If ``abs(pclip) >= 1``, then it specifies a number of pixels
    above or below the median to use for computing the clipping sigma. If ``abs(pclip) < 1``, then it
    specifies the fraction of the pixels above or below the median to use. A positive value selects a
    point above the median and a negative value selects a point below the median. The default of
    ``-0.5`` selects approximately the quartile point. Better to use negative value to avoid cosmic-ray
    contamination.

combine: str, optional.
    The function to be used for the final combining after thresholding, zeroing, scaling, rejection,
    and offset shifting.

        - median if  ``'med'|'medi'|'median'``
        - average if ``'avg'|'average'|'mean'``
        - lower median if ``'lmed'|'lmd'|'lmedian'``

    .. note::
        The lower median means the median which takes the lower value when even number of data is left.
        This is suggested to be robust against cosmic-ray hit according to IRAF `IMCOMBINE`_ manual.
        Currently there is no lmedian-alternative in bottleneck or numpy, so a custom-made version is
        used (in ``numpy_util.py``), which is nothing but a simple modification to the original numpy
        source codes, and this is much slower than bottleneck's median. I think it must be
        re-implemented in the future.
''', indent)


def NDCOMB_RETURNS_COMMON(indent=0):
    return _fix("""
err : ndarray
    The standard deviation map (if `return_variance` is `False`) or the variance map (if
    `return_variance` is `True`) of the survived pixels (with `ddof`).

mask_total : ndarray (dtype bool)
    The full mask, ``N+1``-D. Identical to original FITS files' masks propagated with ``| mask_rej |
    mask_thresh`` below. The total number of rejected pixels at each position can be obtained by
    ``np.count_nonzero(mask_total, axis=0)``.

mask_rej, mask_thresh : ndarray(dtype bool)
    The masks (``N``-D) from the rejection process and thresholding process (`thresholds`). Threshold
    is done prior to any rejection or scaling/zeroing. Number of rejected pixels at each position for
    each process can be obtained by, e.g., ``nrej = np.count_nonzero(mask_rej, axis=0)``. Note that
    `mask_rej` consumes less memory than `nrej`.

low, upp : ndarray (dtype `dtype`)
    The lower and upper bounds (``N``-D) to reject pixel values at each position (``(data < low) | (upp
    < data)`` are removed).

nit : ndarray (dtype uint8)
    The number of iterations (``N``-D) used in rejection process. I cannot think of iterations larger
    than 100, so set the dtype to ``uint8`` to reduce memory and filesize.

rejcode : ndarray (dtype uint8)
    The exit code from rejection (``N``-D). See each rejection's docstring.
""", indent)


def OFFSETS_SHORT(indent=0):
    return _fix('''
offsets : (n, m)-d array
    If given, it must have shape such that ``n`` is the number of images and ``m`` is the dimension of
    the images (offsets in x, y, z, ... order, not pythonic order), and it is directly regarded as the
    "raw offsets".

    The raw offsets are then modified such that the minimum offsets in each axis becomes zero (in
    pythonic way, ``np.max(offsets, axis=0) - offsets``). The offsets are used to determine the final
    output image's shape.
''', indent)


def OFFSETS_LONG(indent=0):
    return _fix('''
offsets : str or (n, m)-d array
    If array, it must have shape such that ``n`` is the number of images and ``m`` is the dimension of
    the images (if ``m=3``, offsets in x, y, z, ... order, not pythonic order), and it is directly
    regarded as the **raw offsets**. If ``str``, the raw offsets are obtained by the followings:

        - ``CRPIX`` values in the header if ``"wcs"|"world"``
        - ``LTV`` values in the header if ``"physical"|"phys"|"phy"``

    .. warning::
        The physical coordinate system is defined by the IRAF-like ``LTM``/``LTV`` keywords define the
        offsets. Currently, only the cases when ``LTMi_j`` is 0 or 1 can be managed. Otherwise, we
        need scaling and it is not supported now.

    For both wcs or physical cases, the raw offsets for *each* frame is nothing but an ``m``-D tuple
    consists of ``offset_raw[i] = CRPIX{m-i}`` or ``LTV{m-i}[_{m-i}]``. The reason to subtract ``i`` is
    because python has ``z, y, x`` order of indexing while WCS information is in ``x, y, z`` order. If
    it is a ``j``-th image, ``offsets[j, :] = offset_raw``, and `offsets` has shape of ``(n, m)``.

    This raw `offsets` are then modified such that the minimum offsets in each axis becomes zero (in
    pythonic way, ``np.max(offsets, axis=0) - offsets``). The offsets are used to determine the final
    output image's shape.

    .. note::
        Though IRAF `IMCOMBINE`_ says it calculates offsets from the 0-th image center if
        ``offsets="wcs"``, it seems it acutally uses ``CRPIX`` from the header... I couldn't find how
        IRAF does offset calculation for WCS, it's not reproducible using rounding. Even using WCS info
        correctly, it's not reproducible. Also, if we only use ``CRPIX``, the offset calculations are
        completely wrong if ``CRPIX`` is not centered at the identical world coordinate (e.g., RA/DEC).
        **IRAF indeed wrongly combines images** if this happens.
''', indent)


def IMCOMBINE_LINK(indent=0):
    return _fix("""
.. _IMCOMBINE: https://iraf.net/irafhelp.php?val=imcombine&help=Help+Page
""", indent)