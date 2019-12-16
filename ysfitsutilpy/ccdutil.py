'''
Collection of functions that are quite far from headerutil.
'''
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, Cutout2D, StdDevUncertainty
from astropy.wcs import WCS
from ccdproc import trim_image

from .misc import binning

__all__ = ["cutccd", "bin_ccd", "load_ccd", "imcopy", "CCDData_astype",
           "make_errmap"]


# TODO: add LTV-like keys to the header.
def cutccd(ccd, position, size, mode='trim', fill_value=np.nan):
    ''' Converts the Cutout2D object to proper CCDData.
    Parameters
    ----------
    ccd: CCDData
        The ccd to be trimmed.

    position : tuple or `~astropy.coordinates.SkyCoord`
        The position of the cutout array's center with respect to the
        ``data`` array.  The position can be specified either as a ``(x,
        y)`` tuple of pixel coordinates or a
        `~astropy.coordinates.SkyCoord`, in which case ``wcs`` is a
        required input.

    size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array along each axis.  If ``size`` is a
        scalar number or a scalar `~astropy.units.Quantity`, then a
        square cutout of ``size`` will be created.  If ``size`` has two
        elements, they should be in ``(ny, nx)`` order.  Scalar numbers
        in ``size`` are assumed to be in units of pixels.  ``size`` can
        also be a `~astropy.units.Quantity` object or contain
        `~astropy.units.Quantity` objects.  Such
        `~astropy.units.Quantity` objects must be in pixel or angular
        units.  For all cases, ``size`` will be converted to an integer
        number of pixels, rounding the the nearest integer.  See the
        ``mode`` keyword for additional details on the final cutout
        size.

        .. note::
            If ``size`` is in angular units, the cutout size is
            converted to pixels using the pixel scales along each
            axis of the image at the ``CRPIX`` location.  Projection
            and other non-linear distortions are not taken into
            account.

    wcs : `~astropy.wcs.WCS`, optional
        A WCS object associated with the input ``data`` array.  If
        ``wcs`` is not `None`, then the returned cutout object will
        contain a copy of the updated WCS for the cutout data array.

    mode : {'trim', 'partial', 'strict'}, optional
        The mode used for creating the cutout data array.  For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap of the
        cutout array and the input ``data`` array is sufficient. For the
        ``'strict'`` mode, the cutout array has to be fully contained
        within the ``data`` array, otherwise an
        `~astropy.nddata.utils.PartialOverlapError` is raised.   In all
        modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'`` mode,
        positions in the cutout array that do not overlap with the
        ``data`` array will be filled with ``fill_value``.  In
        ``'trim'`` mode only the overlapping elements are returned, thus
        the resulting cutout array may be smaller than the requested
        ``shape``.

    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the cutout
        array that do not overlap with the input ``data``.
        ``fill_value`` must have the same ``dtype`` as the input
        ``data`` array.
    '''
    hdr_orig = ccd.header
    w = WCS(hdr_orig)
    cutout = Cutout2D(data=ccd.data, position=position, size=size, wcs=w,
                      mode=mode, fill_value=fill_value, copy=True)
    # Copy True just to avoid any contamination to the original ccd.

    nccd = CCDData(data=cutout.data,
                   header=hdr_orig,
                   wcs=cutout.wcs,
                   unit=ccd.unit)
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
        warn("Since Cutout2D is for small image crop, astropy do not "
             + "currently support distortion in WCS. This may result in "
             + "slightly inaccurate WCS calculation in the future.")

    return nccd


def bin_ccd(ccd, factor_x=1, factor_y=1, binfunc=np.mean, trim_end=False,
            update_header=True):
    ''' Bins the given ccd.
    Paramters
    ---------
    ccd: CCDData
        The ccd to be binned
    factor_x, factor_y: int, optional.
        The binning factors in x, y direction.
    binfunc : funciton object, optional.
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.
    trim_end: bool, optional.
        Whether to trim the end of x, y axes such that binning is done
        without error.
    update_header: bool, optional.
        Whether to update header. Defaults to True.
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
        _ccd.header.add_history(
            f"Binned by (xbin, ybin) = ({factor_x}, {factor_y})")
        _ccd.header["BINFUNC"] = (
            binfunc.__name__, "The function used for binning.")
        _ccd.header["XBINNING"] = (
            factor_x, "Binning done after the observation in X direction")
        _ccd.header["YBINNING"] = (
            factor_y, "Binning done after the observation in Y direction")
    return _ccd


# FIXME: Remove it in the future.
def load_ccd(path, extension=0, usewcs=True, hdu_uncertainty="UNCERT",
             unit='adu'):
    '''remove it when astropy updated:
    Note
    ----
    CCDData.read cannot read TPV WCS:
    https://github.com/astropy/astropy/issues/7650
    '''
    with fits.open(path) as hdul:
        hdul = fits.open(path)
        hdu = hdul[extension]
        try:
            uncdata = hdul[hdu_uncertainty].data
            unc = StdDevUncertainty(uncdata)
        except KeyError:
            unc = None

        w = None
        if usewcs:
            w = WCS(hdu.header)

        if unit not in ['adu', u.adu]:
            ccd = CCDData(data=hdu.data, header=hdu.header, wcs=w,
                          uncertainty=unc, unit=unit)
        else:
            try:
                ccd = CCDData(data=hdu.data, header=hdu.header, wcs=w,
                              uncertainty=unc)
            except ValueError:
                ccd = CCDData(data=hdu.data, header=hdu.header, wcs=w,
                              uncertainty=unc, unit=unit)
    return ccd


def imcopy(fpaths, fits_sections=None, outputs=None, return_ccd=True,
           dtype='float32', **kwargs):
    ''' Similar to IRAF imcopy
    Parameters
    ----------
    fpaths : path-like or array-like of such.
        The path(s) to the original FITS file(s).
    fits_sections : str or array-like of such, optional.
        The section specified by FITS convention, i.e., bracket
        embraced, comma separated, XY order, 1-indexing, and including
        the end index. If given as array-like format of length ``N``,
        all such sections in all FITS files will be extracted.
    outputs : path-like or array-like of such, optional.
        The output paths of each FITS file to be copied. If array-like,
        it must have the shape of ``(M, N)`` where ``M`` and ``N`` are
        the sizes of ``fpaths`` and ``fits_sections``, respectively.
    return_ccd : bool, optional.
        Whether to load the FITS files as ``CCDData`` and return it.
    dtype : dtype, optional.
        The dtype for the ``outputs`` or returning ccds.
    kwargs : optionals
        The keyword arguments for ``CCDData.write``.

    Return
    ------
    results: CCDData or list of CCDData
        Only if ``return_ccd`` is set ``True``.
        A sinlge ``CCDData`` will be returned if only one was input.
        Otherwise, the same number of ``CCDData`` will be gathered as a
        list and returned.
    Note
    ----
    Due to the memory issue, it is generally better NOT to load all the
    FITS files and pass them to this function. Therefore, as it is in
    IRAF, I made this function to accept only the file paths, not the
    pre-loaded CCDData objects. I here will load the

    All the sections will be flattened if they are higher than 1-d. I
    think it will only increase the complexity of the code if I accept
    that...?

    Example
    -------
    >>> from ysfitsutilpy import imcopy
    >>> from pathlib import Path
    >>>
    >>> datapath = Path("./data")
    >>> files = datapath.glob("*.pcr.fits")
    >>> sections = ["[50:100, 50:100]", "[50:100, 50:150]"]
    >>> outputs = [datapath/"test1.fits", datapath/"test2.fits"]
    >>>
    >>> # single file, single section
    >>> trim = imcopy(pcrfits[0], sections[0])
    >>>
    >>> # single file, multi sections
    >>> trims = imcopy(pcrfits[0], sections)
    >>>
    >>> # only save (no return to reduce memory burden) with overwrite option
    >>> imcopy(pcrfits[0], sections, outputs=outputs, overwrite=True)
    >>>
    >>> # multi file multi section
    >>> trims2d = imcopy(pcrfits[:2], fits_sections=sections, outputs=None)
    '''
    to_trim = False
    to_save = False

    str_flat = ("{} with dimension higher than 2-d is not supported yet. "
                + "Currently it's {}-d. Flattening...")
    str_save = ("If outputs is array-like, it's shape must have the shape "
                + "of (fpaths.size, fits_sections.size) = ({}, {}). "
                + "Now it's ({}, {}).")
    fpaths = np.atleast_1d(fpaths)

    if fpaths.ndim > 1:
        print(str_flat.format("fpaths", fpaths.ndim))
        fpaths = fpaths.flatten()
    m = fpaths.shape[0]

    if fits_sections is not None:
        sects = np.atleast_1d(fits_sections)
        to_trim = True
        if sects.ndim > 1:
            print(str_flat.format("fits_sections", sects.ndim))
            sects = sects.flatten()
        n = sects.shape[0]
    else:
        n = 1

    if outputs is not None:
        outputs = np.atleast_2d(outputs)
        to_save = True
        if outputs.ndim > 2:
            raise ValueError("outputs should be lower than 3-d.")
        if outputs.shape != (m, n):
            raise ValueError(str_save.format(m, n, *outputs.shape))

    if return_ccd:
        results = []

    for i, fpath in enumerate(fpaths):
        ccd = load_ccd(fpath)
        result = []
        if to_trim:  # n CCDData will be in ``result``
            for sect in sects:
                nccd = trim_image(ccd, fits_section=sect)
                nccd = CCDData_astype(nccd, dtype=dtype)
                result.append(nccd)
        else:  # only one single CCDData will be in ``result``
            nccd = CCDData_astype(ccd, dtype=dtype)
            result.append(nccd)

        if to_save:
            for j, res in enumerate(result):
                res.write(outputs[i, j], **kwargs)

        if return_ccd:
            if len(result) == 1:
                results.append(result[0])
            else:
                results.append(result)

    if return_ccd:
        if len(results) == 1:
            return results[0]
        else:
            return np.array(results, dtype=object)


def CCDData_astype(ccd, dtype='float32', uncertainty_dtype=None):
    ''' Assign dtype to the CCDData object (numpy uses float64 default).
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
    >>> ccd = CCDData.read("image_unitygain001.fits", ext=0)
    >>> ccd.uncertainty = np.sqrt(ccd.data)
    >>> ccd = yfu.CCDData_astype(ccd, dtype='int16',
    >>>                          uncertainty_dtype='float32')
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
                flat_err=0.0, subtracted_dark=None, return_variance=False):
    ''' Calculate the simple error map in ADU unit.
    Parameters
    ----------
    ccd: array-like
        The ccd data which will be used to generate error map. It must
        be bias subtracted. If dark is subtracted, give
        ``subtracted_dark``. This array will be added to ``ccd.data``
        and used to calculate the Poisson noise term. If the amount of
        this subtracted dark is negligible, you may just set
        ``subtracted_dark = None`` (default).
    gain: float, array-like, or Quantity, optional.
        The effective gain factor in ``electron/ADU`` unit.
    rdnoise: float, array-like, or Quantity, optional.
        The readout noise. Put ``rdnoise=0`` will calculate only the
        Poissonian error. This is useful when generating noise map for
        dark frames.
    flat_err : float, array-like optional.
        The uncertainty from the flat fielding (see, e.g., eq 10 of
        StetsonPB 1987, PASP, 99, 191). Stetson used 0.0075 (0.75%
        fractional uncertainty), and the same is implemented to IRAF
        DAOPHOT: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daopars

    subtracted_dark: array-like
        The subtracted dark map.

    return_variance: bool, optional
        Whether to return as variance map. Default is ``False``, i.e.,
        return the square-rooted standard deviation map. It's better to
        use variance for large image size (computation speed issue).

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
