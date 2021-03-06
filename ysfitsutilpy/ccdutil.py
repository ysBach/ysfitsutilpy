'''
Collection of functions that are quite far from headerutil.
'''
from copy import deepcopy
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData, Cutout2D
from astropy.time import Time
from astropy.wcs import WCS
from ccdproc import trim_image

from .hdrutil import (add_to_header, calc_offset_physical, get_if_none,
                      update_tlm)
from .misc import (_parse_data_header, _parse_image, binning, fitsxy2py,
                   _image_shape, inputs2list)

__all__ = [
    "set_ccd_attribute", "set_ccd_gain_rdnoise",
    "propagate_ccdmask",
    "trim_ccd", "bezel_ccd", "cutccd", "bin_ccd",
    "CCDData_astype", "make_errmap",
    "errormap"
]


def set_ccd_attribute(ccd, name, value=None, key=None, default=None, unit=None, header_comment=None,
                      update_header=True, verbose=True, wrapper=None, wrapper_kw={}):
    ''' Set attributes from given paramters.

    Parameters
    ----------
    ccd : CCDData
        The ccd to add attribute.

    value : Any, optional.
        The value to be set as the attribute. If `None`, the ``ccd.header[key]`` will be searched.

    name : str, optional.
        The name of the attribute.

    key : str, optional.
        The key in the ``ccd.header`` to be searched if ``value=None``.

    unit : astropy.Unit, optional.
        The unit that will be applied to the found value.

    header_comment : str, optional.
        The comment string to the header if ``update_header=True``. If `None` (default), search for
        existing comment in the original header by ``ccd.comments[key]`` and only overwrite the value
        by ``ccd.header[key]=found_value``. If it's not `None`, the comments will also be overwritten
        if ``update_header=True``.

    wrapper : function object, None, optional.
        The wrapper function that will be applied to the found value. Other keyword arguments should be
        given as a dict to ``wrapper_kw``.

    wrapper_kw : dict, optional.
        The keyword argument to ``wrapper``.

    Example
    -------
    >>> set_ccd_attribute(ccd, 'gain', value=2, unit='electron/adu')
    >>> set_ccd_attribute(ccd, 'ra', key='RA', unit=u.deg, default=0)

    Note
    ----
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
        add_to_header(ccd.header, 'h', s, t_ref=_t_start)

    setattr(ccd, name, value_Q)
    update_tlm(ccd.header)


# TODO: This is quite much overlapping with get_gain_rdnoise...
def set_ccd_gain_rdnoise(ccd, verbose=True, update_header=True,
                         gain=None, gain_key="GAIN", gain_unit=u.electron/u.adu,
                         rdnoise=None, rdnoise_key="RDNOISE", rdnoise_unit=u.electron):
    """ A convenience set_ccd_attribute for gain and readnoise.

    Parameters
    ----------
    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. If ``gain`` or ``readnoise`` is specified, they are interpreted
        with ``gain_unit`` and ``rdnoise_unit``, respectively. If they are not specified, this function
        will seek for the header with keywords of ``gain_key`` and ``rdnoise_key``, and interprete the
        header value in the unit of ``gain_unit`` and ``rdnoise_unit``, respectively.

    gain_key, rdnoise_key : str, optional.
        See ``gain``, ``rdnoise`` explanation above.

    gain_unit, rdnoise_unit : str, astropy.Unit, optional.
        See ``gain``, ``rdnoise`` explanation above.

    verbose : bool, optional.
        The verbose option.

    update_header : bool, optional
        Whether to update the given header.
    """
    gain_str = f"[{gain_unit:s}] Gain of the detector"
    rdn_str = f"[{rdnoise_unit:s}] Readout noise of the detector"
    set_ccd_attribute(ccd=ccd, name='gain', value=gain, key=gain_key, unit=gain_unit, default=1.0,
                      header_comment=gain_str, update_header=update_header, verbose=verbose)
    set_ccd_attribute(ccd=ccd, name='rdnoise', value=rdnoise, key=rdnoise_key, unit=rdnoise_unit, default=0.0,
                      header_comment=rdn_str, update_header=update_header, verbose=verbose)


def propagate_ccdmask(ccd, additional_mask=None):
    ''' Propagate the CCDData's mask and additional mask.

    Parameters
    ----------
    ccd : CCDData, ndarray
        The ccd to extract mask. If ndarray, it will only return a copy of ``additional_mask``.

    additional_mask : mask-like, None
        The mask to be propagated.

    Note
    ----
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
# def trim_ccd(ccd, fits_section=None, add_keyword=True, verbose=False):
#     _t = Time.now()
#     trim_str = f"Trimmed using {fits_section}"
#     trimmed_ccd = trim_image(ccd, fits_section=fits_section, add_keyword=add_keyword)
#     ny, nx = ccd.data.shape

#     if fits_section:
#         trim_slice = fitsxy2py(fits_section)
#         ltv1 = -1*trim_slice[1].indices(nx)[0]
#         ltv2 = -1*trim_slice[0].indices(ny)[0]
#     else:
#         ltv1 = 0.
#         ltv2 = 0.

#     hdr = trimmed_ccd.header
#     for k, v in zip(["LTV1", "LTV2"], [ltv1, ltv2]):
#         try:  # if LTV exists already
#             hdr[k] += v
#         except KeyError:
#             hdr[k] = v

#     add_11 = not ("LTM1_1" in hdr)
#     add_12 = not ("LTM1_2" in hdr)
#     add_21 = not ("LTM2_1" in hdr)
#     add_22 = not ("LTM2_2" in hdr)
#     if add_11:
#         if "LTM1" in hdr:
#             hdr["LTM1_1"] = hdr["LTM1"]
#         else:
#             hdr["LTM1_1"] = 1.
#     if add_12:
#         hdr["LTM1_2"] = 0.
#     if add_21:
#         hdr["LTM2_1"] = 0.
#     if add_22:
#         if "LTM2" in hdr:
#             hdr["LTM2_2"] = hdr["LTM2"]
#         else:
#             hdr["LTM2_2"] = 1.

#     add_to_header(trimmed_ccd.header, 'h', trim_str, t_ref=_t, verbose=verbose)
#     update_tlm(trimmed_ccd.header)

#     return trimmed_ccd

def trim_ccd(ccd, fits_section=None, update_header=True, verbose=False):
    _t = Time.now()
    trim_str = f"Trimmed using {fits_section}"
    trimmed_ccd = trim_image(ccd, fits_section=fits_section, add_keyword=update_header)
    ndim = ccd.data.ndim  # ndim == NAXIS keyword
    shape = ccd.data.shape

    if fits_section:
        trim_slice = fitsxy2py(fits_section)
        ltvs = []
        for i_python, npix in enumerate(shape):
            # If section is [10:X], the LTV must be -9, not -10.
            ltvs.append(-1*trim_slice[i_python].indices(npix)[0])
    else:
        ltvs = [0.]*ndim

    # python shape is in z, y, x order, so reverse it
    ltvs = ltvs[::-1]

    if update_header:
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

    if verbose:
        add_to_header(trimmed_ccd.header, 'h', trim_str, t_ref=_t, verbose=verbose)
        update_tlm(trimmed_ccd.header)

    return trimmed_ccd


def bezel_ccd(ccd, bezel_x=None, bezel_y=None, replace=np.nan, verbose=False):
    """ Replace pixel values at the edges of the image.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The data to be used.

    bezel_x, bezel_y : None, int, float, size-2 of these, optional.
        The bezel width along x and y directions. If `float`, it will be rounded to integer. If given
        as size-2 array-like, it must be ``(bezel_lower, bezel_upper)``.

    replace : None, float-like, optinoal.
        If `None`, it is identical to trimming the CCD with given bezels. If given as float-like, the
        bezel pixels will be replaced with this value. Defaults to ``np.nan`` to keep the size of the
        input ``ccd``.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData`
        The trimmed (``replace=None``) or bezel-replaced (otherwise)
        ccd.

    Example
    -------
    >>>

    Note
    ----

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
    if ccd.data.ndim != 2:
        raise ValueError("Only 2-D CCDData is supported yet.")

    ny, nx = ccd.data.shape

    if bezel_x is None and bezel_y is None:
        return ccd

    bezel_x = _sanitize_bezel(bezel_x, nx)
    bezel_y = _sanitize_bezel(bezel_y, ny)

    if replace is None:
        sl = (f"[{bezel_x[0] + 1}:{nx - bezel_x[1]},{bezel_y[0] + 1}:{ny - bezel_y[1]}]")
        nccd = trim_ccd(ccd, fits_section=sl) if isinstance(ccd, CCDData) else np.array(ccd)[fitsxy2py(sl)]
        # i.e., use trim_ccd if CCDData, and use python slice if ndarray-like
    else:
        nccd = ccd.copy()
        nccd.data[:bezel_y[0], :] = replace
        nccd.data[ny - bezel_y[1]:, :] = replace
        nccd.data[:, :bezel_x[0]] = replace
        nccd.data[:, nx - bezel_x[0]:] = replace
        add_to_header(
            nccd.header, 'h', t_ref=_t, verbose=verbose,
            s=f"Replaced pixels with bezel width {bezel_x} along x and {bezel_y} along y with {replace}."
        )

    update_tlm(nccd.header)

    return nccd


def trim_overlap(inputs, extension=None, coordinate='image'):
    ''' Trim only the overlapping regions of the two CCDs
    Parameters
    ----------
    coordinate : str, optional.
        Ways to find the overlapping region. If ``'image'`` (default), output size will be
        ``np.min([ccd.shape for ccd in ccds], axis=0)``. If ``'physical'``, overlapping region will be
        found based on the physical coordinates.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    Note
    ----
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
        offsets.append(calc_offset_physical(ccd, reference, order_xyz=False, ignore_ltm=True))

    offsets, new_shape = _image_shape(shapes, offsets, method='overlap', intify_offsets=False)


def cutccd(ccd, position, size, mode='trim', fill_value=np.nan):
    ''' Converts the Cutout2D object to proper CCDData.

    Parameters
    ----------
    ccd: CCDData
        The ccd to be trimmed.

    position : tuple or `~astropy.coordinates.SkyCoord`
        The position of the cutout array's center with respect to the ``data`` array.  The position can
        be specified either as a ``(x, y)`` tuple of pixel coordinates or a
        `~astropy.coordinates.SkyCoord`, in which case ``wcs`` is a required input.

    size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array along each axis.  If ``size`` is a scalar number or a scalar
        `~astropy.units.Quantity`, then a square cutout of ``size`` will be created.  If ``size`` has
        two elements, they should be in ``(ny, nx)`` order.  Scalar numbers in ``size`` are assumed to
        be in units of pixels.  ``size`` can also be a `~astropy.units.Quantity` object or contain
        `~astropy.units.Quantity` objects.  Such `~astropy.units.Quantity` objects must be in pixel or
        angular units.  For all cases, ``size`` will be converted to an integer number of pixels,
        rounding the the nearest integer.  See the ``mode`` keyword for additional details on the final
        cutout size.

        .. note::
            If ``size`` is in angular units, the cutout size is converted to pixels using the pixel
            scales along each axis of the image at the ``CRPIX`` location.  Projection and other
            non-linear distortions are not taken into account.

    wcs : `~astropy.wcs.WCS`, optional
        A WCS object associated with the input ``data`` array.  If ``wcs`` is not `None`, then the
        returned cutout object will contain a copy of the updated WCS for the cutout data array.

    mode : {'trim', 'partial', 'strict'}, optional
        The mode used for creating the cutout data array.  For the ``'partial'`` and ``'trim'`` modes,
        a partial overlap of the cutout array and the input ``data`` array is sufficient. For the
        ``'strict'`` mode, the cutout array has to be fully contained within the ``data`` array,
        otherwise an `~astropy.nddata.utils.PartialOverlapError` is raised.   In all modes,
        non-overlapping arrays will raise a `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'``
        mode, positions in the cutout array that do not overlap with the ``data`` array will be filled
        with ``fill_value``.  In ``'trim'`` mode only the overlapping elements are returned, thus the
        resulting cutout array may be smaller than the requested ``shape``.

    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the cutout array that do not overlap with
        the input ``data``. ``fill_value`` must have the same ``dtype`` as the input ``data`` array.
    '''
    hdr_orig = ccd.header
    w = WCS(hdr_orig)
    cutout = Cutout2D(
        data=ccd.data, position=position, size=size, wcs=w, mode=mode, fill_value=fill_value, copy=True
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
            "Since Cutout2D is for small image crop, astropy do not currently support distortion in WCS."
            + "This may result in slightly inaccurate WCS calculation in the future."
        )

    update_tlm(nccd.header)

    return nccd


def bin_ccd(ccd, factor_x=1, factor_y=1, binfunc=np.mean, trim_end=False, update_header=True):
    ''' Bins the given ccd.

    Paramters
    ---------
    ccd : CCDData
        The ccd to be binned

    factor_x, factor_y : int, optional.
        The binning factors in x, y direction.

    binfunc : funciton object, optional.
        The function to be applied for binning, such as ``np.sum``, ``np.mean``, and ``np.median``.

    trim_end : bool, optional.
        Whether to trim the end of x, y axes such that binning is done without error.

    update_header : bool, optional.
        Whether to update header. Defaults to True.

    Note
    ----
    This is ~ 20-30 to upto 10^5 times faster than astropy.nddata's block_reduce:
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
    Tested on  MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4),
    Radeon Pro 560X (4GB)]
    '''
    _t_start = Time.now()

    if not isinstance(ccd, CCDData):
        raise TypeError("ccd must be CCDData object.")

    if factor_x == 1 and factor_y == 1:
        return ccd
    _ccd = ccd.copy()
    _ccd.data = binning(_ccd.data,
                        factor_x=factor_x,
                        factor_y=factor_y,
                        binfunc=binfunc,
                        trim_end=trim_end)
    if update_header:
        _ccd.header["BINFUNC"] = (binfunc.__name__, "The function used for binning.")
        _ccd.header["XBINNING"] = (factor_x, "Binning done after the observation in X direction")
        _ccd.header["YBINNING"] = (factor_y, "Binning done after the observation in Y direction")
        # add as history
        add_to_header(
            _ccd.header, 'h', t_ref=_t_start, s=f"Binned by (xbin, ybin) = ({factor_x}, {factor_y}) "
        )
    update_tlm(_ccd.header)
    return _ccd


def CCDData_astype(ccd, dtype='float32', uncertainty_dtype=None):
    ''' Assign dtype to the CCDData object (numpy uses float64 default).

    Parameters
    ----------
    ccd : CCDData
        The ccd to be astyped.

    dtype : dtype-like
        The dtype to be applied to the data

    uncertainty_dtype : dtype-like
        The dtype to be applied to the uncertainty. Be default, use the same dtype as data
        (``uncertainty_dtype=dtype``).

    Example
    -------
    >>> from astropy.nddata import CCDData
    >>> import numpy as np
    >>> ccd = CCDData.read("image_unitygain001.fits", 0)
    >>> ccd.uncertainty = np.sqrt(ccd.data)
    >>> ccd = yfu.CCDData_astype(ccd, dtype='int16', uncertainty_dtype='float32')
    '''
    nccd = ccd.copy()
    nccd.data = nccd.data.astype(dtype)

    try:
        if uncertainty_dtype is None:
            uncertainty_dtype = dtype
        nccd.uncertainty.array = nccd.uncertainty.array.astype(uncertainty_dtype)
    except AttributeError:
        # If there is no uncertainty attribute in the input ``ccd``
        pass

    update_tlm(nccd.header)
    return nccd


def make_errmap(ccd, gain_epadu=1, rdnoise_electron=0, flat_err=0.0, subtracted_dark=None,
                return_variance=False, detail=False):
    ''' Calculate the simple error map. Use ``errormap`` instead.

    Parameters
    ----------
    ccd: array-like
        The ccd data which will be used to generate error map. It must be bias subtracted. If dark is
        subtracted, give ``subtracted_dark``. This array will be added to ``ccd.data`` and used to
        calculate the Poisson noise term. If the amount of this subtracted dark is negligible, you may
        just set ``subtracted_dark = None`` (default).

    gain_epadu, rdnoise_electron: float, array-like, or Quantity, optional.
        The effective gain factor in ``electron/ADU`` unit and the readout noise in ``electron`` unit.

    flat_err : float, array-like optional.
        The uncertainty from the flat fielding (see, e.g., eq 10 of StetsonPB 1987, PASP, 99, 191).
        Stetson used 0.0075 (0.75% fractional uncertainty), and the same is implemented to IRAF
        DAOPHOT: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daopars

    subtracted_dark: array-like
        The subtracted dark map.

    return_variance: bool, optional
        Whether to return as variance map. Default is `False`, i.e., return the square-rooted standard
        deviation map. It's better to use variance for large image size (computation speed issue).

    Example
    -------
    >>> from astropy.nddata import CCDData, StdDevUncertainty
    >>> ccd = CCDData.read("obj001.fits", 0)
    >>> hdr = ccd.header
    >>> dark = CCDData.read("master_dark.fits", 0)
    >>> params = dict(gain_epadu=hdr["GAIN"], rdnoise_electron=hdr["RDNOISE"], subtracted_dark=dark.data)
    >>> ccd.uncertainty = StdDevUncertainty(make_errmap(ccd, **params))
    '''
    print("Use ``errormap`` instead.")

    data, _ = _parse_data_header(ccd)
    data[data < 0] = 0  # make all negative pixel to 0

    if isinstance(gain_epadu, u.Quantity):
        gain_epadu = gain_epadu.to(u.electron / u.adu).value
    elif isinstance(gain_epadu, str):
        gain_epadu = float(gain_epadu)

    if isinstance(rdnoise_electron, u.Quantity):
        rdnoise_electron = rdnoise_electron.to(u.electron).value
    elif isinstance(rdnoise_electron, str):
        rdnoise_electron = float(rdnoise_electron)

    var_flat = (data * flat_err)**2

    # Get Poisson noise
    if subtracted_dark is not None:
        dark = subtracted_dark.copy()
        if isinstance(dark, CCDData):
            dark = dark.data
        # If subtracted dark is negative, this may cause negative pixel
        # in ``data``:
        data += dark

    var_Poisson = data / gain_epadu  # (data * gain) / gain**2 to make it ADU
    var_RDnoise = (rdnoise_electron / gain_epadu)**2

    varmap = var_Poisson + var_RDnoise + var_flat
    if return_variance:
        return varmap
    else:
        errmap = np.sqrt(varmap)
        return errmap


def errormap(ccd_biassub, gain_epadu=1, rdnoise_electron=0, subtracted_dark=None, flat=None,
             dark_std=None, flat_err=None, dark_std_min='rdnoise', return_variance=False):
    ''' Calculate the detailed pixel-wise error map in ADU unit.

    Parameters
    ----------
    ccd : CCDData, PrimaryHDU, ImageHDU, ndarray.
        The ccd data which will be used to generate error map. It must be **bias subtracted**. If dark
        is subtracted, give ``subtracted_dark``. This array will be added to ``ccd.data`` and used to
        calculate the Poisson noise term. If the amount of this subtracted dark is negligible, you may
        just set ``subtracted_dark = None`` (default).

    gain_epadu, rdnoise_electron : float, array-like, or Quantity, optional.
        The effective gain factor in ``electron/ADU`` unit and the readout noise in ``electron`` unit.

    subtracted_dark : array-like
        The subtracted dark map.

    flat : ndarray, optional.
        The flat field value. There is no need that flat values are normalized. If `None` (default), a
        constant flat of value ``1`` is used.

    flat_err : float, array-like optional.
        The uncertainty of the flat, which is obtained by the central limit theorem (sample standard
        deviation of the pixel divided by the square root of the number of flat frames). An example in
        IRAF and DAOPHOT: the uncertainty from the flat fielding ``flat_err/flat`` is set as a constant
        (see, e.g., eq 10 of StetsonPB 1987, PASP, 99, 191) set as Stetson used 0.0075 (0.75%
        fractional uncertainty), and the same is implemented to IRAF DAOPHOT:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daopars

    dark_std : float, array-like, optional.
        The sample standard deviation of dark pixels. It **should not be divided by the number of dark
        frames**, because we are interested in the uncertainty in the dark (prediction), not the
        confidence interval of the *mean* of the dark. If `None`, it is assumed dark has no
        uncertainty.

    dark_std_min : 'rdnoise', float, optional.
        The minimum value for ``dark_std``. Any ``dark_std`` value below this will be replaced by this
        value. If ``'rdnoise'`` (default), the ``rdnoise_electron/gain_epadu`` will be used.

    return_variance: bool, optional
        Whether to return as variance map. Default is `False`, i.e., return the square-rooted standard
        deviation map. It's better to use variance for large image size (computation speed issue).

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

    # restore dark for Poisson term calculation
    if subtracted_dark is not None:
        data += subtracted_dark

    if flat is None:
        flat = 1

    var = data / (gain_epadu*flat**2)
    var += (rdnoise_electron/(gain_epadu*flat))**2

    if dark_std is not None:
        if dark_std_min == 'rdnoise':
            dark_std_min = rdnoise_electron/gain_epadu
        dark_std[dark_std < dark_std_min] = dark_std_min
        var += (dark_std/flat)**2

    if flat_err is not None:
        var += data**2*(flat_err/flat)**2

    if return_variance:
        return var
    else:  # Sqrt is the most time-consuming part...
        return np.sqrt(var)

    # var_pois = data / (gain_epadu * flat**2)
    # var_rdn = (rdnoise_electron/(gain_epadu*flat))**2
    # var_flat_err = data**2*(flat_err/flat)**2
    # var_dark_err = (dark_err/flat)**2
