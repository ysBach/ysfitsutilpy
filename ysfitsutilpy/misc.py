'''
Objects that are
(1) too fundamental, so used in various places,
(2) completely INDEPENDENT of all other modules of this package.
'''
import sys

import numpy as np
from astro_ndslice import is_list_like, listify
from astropy import units as u
from astropy.time import Time
import numba as nb

__all__ = ["MEDCOMB_KEYS_INT", "SUMCOMB_KEYS_INT", "MEDCOMB_KEYS_FLT32",
           "LACOSMIC_KEYS", "LACOSMIC_CRREJ", "parse_crrej_psf",
           "get_size",
           "cmt2hdr", "update_tlm", "update_process",
           "weighted_avg", "sigclip_dataerr",
           "circular_mask", "circular_mask_2d",
           "enclosing_circle_radius",
           "str_now", "change_to_quantity", "binning",
           "quantile_lh", "quantile_sigma",
           "min_max_med_1d", "mean_std_1d"]


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
                 'invar': None,
                 'inbkg': None,
                 'niter': 4,
                 'sepmed': False,
                 'cleantype': 'medmask',
                 'fsmode': 'median',
                 'psfmodel': 'gauss',
                 'psffwhm': 2.5,
                 'psfsize': 7,
                 'psfk': None,
                 'psfbeta': 4.765}

# same as above, but simplify `fsmode`, `psfmodel`, and `psfk` into `fs`
LACOSMIC_CRREJ = {'sigclip': 4.5,
                  'sigfrac': 0.5,
                  'objlim': 1.0,
                  'satlevel': np.inf,
                  'invar': None,
                  'inbkg': None,
                  'niter': 4,
                  'sepmed': False,
                  'cleantype': 'medmask',
                  'fs': 'median',
                  'psffwhm': 2.5,
                  'psfsize': 7,
                  'psfbeta': 4.765}


def get_size(obj, seen=None):
    """ Recursively finds size of objects.
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


def cmt2hdr(
        header,
        histcomm,
        s,
        precision=3,
        time_fmt="{:.>72s}",
        t_ref=None,
        dt_fmt="(dt = {:.3f} s)",
        set_kw={"after": -1},
        verbose=False,
):
    """ Automatically add timestamp as well as HISTORY or COMMENT string

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
    The timming benchmark for a reasonably long header (len(ccd.header.cards) =
    197) shows dt ~ 0.2-0.3 ms on MBP 15" [2018, macOS 11.6, i7-8850H (2.6 GHz;
    6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)]:

    %timeit ccd.header.copy()
    1.67 ms +/- 33.3 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    %timeit yfu.cmt2hdr(ccd.header.copy(), 'h', 'test')
    1.89 ms +/- 141 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    %timeit yfu.cmt2hdr(ccd.header.copy(), 'hist', 'test')
    1.89 ms +/- 144 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    %timeit yfu.cmt2hdr(ccd.header.copy(), 'histORy', 'test')
    1.95 ms +/- 146 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    """
    pall = locals()
    if histcomm.lower() in ["h", "hist"]:
        pall["histcomm"] = "HISTORY"
        return cmt2hdr(**pall)
    elif histcomm.lower() in ["c", "com", "comm"]:
        pall["histcomm"] = "COMMENT"
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

    for _s in listify(s):
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
    """ Adds the IRAF-like ``FITS-TLM`` right after ``NAXISi``.

     Timing on MBP 15" [2018, macOS 11.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB
    (2400MHz DDR4), Radeon Pro 560X (4GB)]:
    %timeit yfu.update_tlm(ccd.header)
    # 443 µs +/- 19.5 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    """
    now = Time(Time.now(), precision=0).isot
    try:
        del header["FITS-TLM"]
    except KeyError:
        pass
    try:
        header.set(
            "FITS-TLM",
            value=now,
            comment="UT of last modification of this FITS file",
            after=1,
        )
    except AttributeError:  # If header is OrderedDict
        header["FITS-TLM"] = (now, "UT of last modification of this FITS file")


def update_process(
        header,
        process=None,
        key="PROCESS",
        delimiter="",
        add_comment=True,
        additional_comment=dict(),
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
        if delimiter:
            process = header[key].split(delimiter) + process
        else:
            process = list(header[key]) + process
        # do not additionally add comment.
    elif add_comment:
        # add comment.
        cmt2hdr(
            header,
            "c",
            time_fmt=None,
            s=(
                f"[yfu.update_process] Standard items for {key} includes B=bias, D=dark, "
                + "F=flat, T=trim, W=WCS, O=Overscan, I=Illumination, C=CRrej, R=fringe, "
                + "P=fixpix, X=crosstalk."
            ),
        )

    header[key] = (delimiter.join(process), "Process (order: 1-2-3-...): see comment.")

    if additional_comment:
        addstr = ", ".join([f"{k}={v}" for k, v in additional_comment.items()])
        cmt2hdr(header, "c", f"User added items to {key}: {addstr}.", time_fmt=None)
    update_tlm(header)


def parse_crrej_psf(
        fs="median",
        psffwhm=2.5,
        psfsize=7,
        psfbeta=4.765,
        fill_with_none=True
):
    """Return a dict of minimal keyword arguments for
        `~astroscrappy.detect_cosmics`.
    fs : str, ndarray, list of such, optional.
        If it is a list-like of kernels, it must **NOT** be an ndarray of
        ``N-by-2`` or ``2-by-N``, etc. You may use `list`, `tuple`, or even
        `~pandas.Series` of ndarrays.
    fill_with_none : bool, optional.
        If `True`, the unnecessary keywords will be filled with `None`, rather
        than default parameter values (IRAF version of LACosmics). Works only
        if any of the input parmeters is list-like. If all input parameters are
        scalar (or `fs` is a single ndarray), only minimal dict is returned
        without filling with `None`.
    Notes
    -----
    assert parse_crrej_psf() == {'fsmode': 'median'}

    assert (parse_crrej_psf("gauss", psffwhm=2, psfsize=3, psfbeta=1)
            == {'fsmode': 'convolve', 'psfmodel': 'gauss', 'psffwhm': 2, 'psfsize': 3})

    assert (parse_crrej_psf("moffat", psffwhm=2, psfsize=3, psfbeta=1)
            == {'fsmode': 'convolve', 'psfmodel': 'moffat', 'psffwhm': 2, 'psfsize': 3, 'psfbeta': 1})

    assert (parse_crrej_psf("moffat", psffwhm=2, psfsize=3, psfbeta=[1, 2])
        == {'fsmode': ['convolve', 'convolve'],
 'psfmodel': ['moffat', 'moffat'],
 'psfk': [None, None],
 'psffwhm': [2, 2],
 'psfsize': [3, 3],
 'psfbeta': [1, 2]}
    )

    assert (parse_crrej_psf([np.eye(3), np.eye(5)])
    == {'fsmode': ['convolve', 'convolve'],
 'psfmodel': [None, None],
 'psfk': [np.array([[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]]),
  np.array([[1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0.],
         [0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1.]])],
 'psffwhm': [None, None],
 'psfsize': [None, None],
 'psfbeta': [None, None]}
)

    with pytest.raises(ValueError):
        parse_crrej_psf("moffat", psffwhm=2, psfsize=[3, 3, 3], psfbeta=[1, 2])

    assert (parse_crrej_psf("gaussx", fill_with_none=False)
    == {'fsmode': 'convolve', 'psfmodel': 'gaussx', 'psffwhm': 2.5, 'psfsize': 7})

    assert (parse_crrej_psf("moffat", psffwhm=[2, 3, 4], fill_with_none=False)
    == {'fsmode': ['convolve', 'convolve', 'convolve'],
 'psfmodel': ['moffat', 'moffat', 'moffat'],
 'psfk': [None, None, None],
 'psffwhm': [2, 3, 4],
 'psfsize': [7, 7, 7],
 'psfbeta': [4.765, 4.765, 4.765]}
 )
    """
    if (is_list_like(psffwhm, psfsize, psfbeta, func=any)
            or (is_list_like(fs) and not isinstance(fs, np.ndarray))):
        fs = listify(fs)
        psffwhm = listify(psffwhm)
        psfsize = listify(psfsize)
        psfbeta = listify(psfbeta)
        lengths = (len(fs), len(psffwhm), len(psfsize), len(psfbeta))
        length = max(lengths)
        if not all(_len in [1, length] for _len in lengths):
            raise ValueError(
                "`fs`, `psffwhm`, `psfsize`, and `psfbeta` must all be "
                f"length 1 or the same length (current maxlength = {length})."
            )

        fs = fs*length if len(fs) == 1 else fs
        psffwhm = psffwhm*length if len(psffwhm) == 1 else psffwhm
        psfsize = psfsize*length if len(psfsize) == 1 else psfsize
        psfbeta = psfbeta*length if len(psfbeta) == 1 else psfbeta

        def _allocate(_fs, _psffwhm, _psfsize, _psfbeta):
            if isinstance(_fs, str):
                if _fs == "median":
                    fsmode = "median"
                    psfmodel = None if fill_with_none else LACOSMIC_KEYS["psfmodel"]
                    psfk = None  # anyway, default in LACOSMIC_KEYS is `None`
                    psffwhm = None if fill_with_none else LACOSMIC_KEYS["psffwhm"]
                    psfsize = None if fill_with_none else LACOSMIC_KEYS["psfsize"]
                    psfbeta = None if fill_with_none else LACOSMIC_KEYS["psfbeta"]
                elif _fs == "moffat":
                    fsmode = "convolve"
                    psfmodel = "moffat"
                    psfk = None
                    psffwhm = _psffwhm
                    psfsize = _psfsize
                    psfbeta = _psfbeta
                elif _fs in ("gauss", "gaussx", "gaussy"):
                    fsmode = "convolve"
                    psfmodel = _fs
                    psfk = None
                    psffwhm = _psffwhm
                    psfsize = _psfsize
                    psfbeta = None if fill_with_none else LACOSMIC_KEYS["psfbeta"]
            elif isinstance(_fs, np.ndarray):
                fsmode = "convolve"
                psfmodel = None if fill_with_none else LACOSMIC_KEYS["psfmodel"]
                psfk = _fs
                psffwhm = None if fill_with_none else LACOSMIC_KEYS["psffwhm"]
                psfsize = None if fill_with_none else LACOSMIC_KEYS["psfsize"]
                psfbeta = None if fill_with_none else LACOSMIC_KEYS["psfbeta"]
            else:
                raise ValueError(f"fs ({fs}) not understood")
            return fsmode, psfmodel, psfk, psffwhm, psfsize, psfbeta

        res = dict(fsmode=[], psfmodel=[], psfk=[], psffwhm=[], psfsize=[], psfbeta=[])
        for _fs, _psffwhm, _psfsize, _psfbeta in zip(fs, psffwhm, psfsize, psfbeta):
            fsmode, psfmodel, psfk, psffwhm, psfsize, psfbeta = _allocate(
                _fs, _psffwhm, _psfsize, _psfbeta
            )
            res["fsmode"].append(fsmode)
            res["psfmodel"].append(psfmodel)
            res["psfk"].append(psfk)
            res["psffwhm"].append(psffwhm)
            res["psfsize"].append(psfsize)
            res["psfbeta"].append(psfbeta)
        return res

    elif isinstance(fs, np.ndarray):
        return dict(fsmode="convolve", psfk=fs)
    elif isinstance(fs, str):
        if fs == "median":
            return dict(fsmode=fs)
        elif fs == "moffat":
            return dict(fsmode="convolve", psfmodel="moffat",
                        psffwhm=psffwhm, psfsize=psfsize, psfbeta=psfbeta)
        elif fs in ["gauss", "gaussx", "gaussy"]:
            return dict(fsmode="convolve", psfmodel=fs,
                        psffwhm=psffwhm, psfsize=psfsize)
    else:
        raise ValueError(f"fs ({fs}) not understood")


def weighted_avg(val, err):
    # Weighted mean and standard error
    w = 1/(err**2)
    wsum = np.sum(w)
    wvg = np.sum(w*val)/wsum
    wse = 1/np.sqrt(wsum)
    return wvg, wse


# !FIXME: not finished
# TODO: add err_lower, err_upper, sigma_lower, sigma_upper
def sigclip_dataerr(val, err, cenfunc="wvg", sigma=3, maxiters=3):
    if cenfunc == "wvg":
        cenfunc = lambda val, err: weighted_avg(val, err)[0]
    elif cenfunc in ["avg", "average", "mean"]:
        cenfunc = lambda val, err: np.mean(val)[0]  # err is dummy
    else:
        raise ValueError(f"cenfunc={cenfunc} is not implemented yet.")

    val = np.ma.array(val)
    val_clipped = val.compressed()
    err_clipped = err[val.mask]
    cen = cenfunc(val_clipped, err_clipped)

    for i in range(maxiters):
        # calculate deviation for all (even masked) elements:
        deviation = np.abs(val.data - cen)
        mask = (deviation > sigma*err)

    return val, mask


def circular_mask(shape, center=None, radius=None, center_xyz=True):
    ''' Creates an N-D circular (circular, sphereical, ...) mask.

    Parameters
    ----------
    shape : tuple
        The pythonic shape, i.e., `arr.shape` (not xyz order).

    center : tuple, None, optional.
        The center of the circular mask. If `None` (default), the central
        position is used.

    radius : float, None, optional.
        The radius of the mask. If `None`, the distance to the closest edge of
        the image is used.

    center_xyz : bool, optional.
        Whether the center is in xyz order.

    Notes
    -----
    Idea copied from
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

    Note that this is slow due to the "general" N-D nature of the mask.
    If you need a 2-D mask, use `circular_mask_2d`
    '''
    if center is None:  # use the middle of the image
        center = [npix/2 for npix in shape[::-1]]

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


def circular_mask_2d(
    shape, center=None, radius=0.5, method="center", subpixels=5, maskmin=0, return_apertures=False
):
    """ Creates a 2-D circular mask using photutils CircularAperture.

    Parameters
    ----------
    shape : tuple
        The shape of the 2-D image in *pythonic* order, i.e., `arr.shape`
        (height, width).

    center : array-like, None, optional.
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

        If `None`, the center is set to the middle of the image, i.e.,
        ``(shape[0] / 2, shape[1] / 2)``.
        Default is `None`.

    radius : float, array-like optional.
        The radius (radii) of the circular mask(s).

    method : {'exact', 'center', 'subpixel'}, optional
        The method used to determine the overlap of the aperture on the pixel
        grid. Not all options are available for all aperture types. Note that
        the more precise methods are generally slower. The following methods
        are available:

        * ``'exact'`` (default):
        The exact fractional overlap of the aperture and each pixel is
        calculated. The aperture weights will contain values between 0 and 1.

        * ``'center'``:
        A pixel is considered to be entirely in or out of the aperture
        depending on whether its center is in or out of the aperture. The
        aperture weights will contain values only of 0 (out) and 1 (in).

        * ``'subpixel'``:
        A pixel is divided into subpixels (see the ``subpixels`` keyword), each
        of which are considered to be entirely in or out of the aperture
        depending on whether its center is in or out of the aperture. If
        ``subpixels=1``, this method is equivalent to ``'center'``. The
        aperture weights will contain values between 0 and 1.

    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor in each
        dimension. That is, each pixel is divided into ``subpixels**2``
        subpixels. This keyword is ignored unless ``method='subpixel'``.

    maskmin : float, optional
        The minimum value for the mask. If the aperture weights are greater
        than this value, the pixel is considered to be in the aperture. This
        keyword is ignored unless ``method='exact'`` or ``method='subpixel'``.

    return_apertures : bool, optional
        If `True`, return the `CircularAperture` objects and the masks
        instead of the 2D mask. This is useful if you want to use the
    """
    from photutils.aperture import CircularAperture

    if center is None:  # use the middle of the image
        center = (shape[0] / 2, shape[1] / 2)

    try:
        apertures = CircularAperture(center, radius)
        apmasks = apertures.to_mask(method=method, subpixels=subpixels)
    except ValueError:
        # multiple radii and "ValueError: 'r' must be a positive scalar" happens.
        if center.shape[0] != np.size(radius):
            raise ValueError(
                "If `radius` is an array-like, it must have the same length as `center`; "
                f"({center.shape[0] = }) != ({np.size(radius)} = )."
            )
        apertures = [CircularAperture(c, radius=r) for c, r in zip(center, radius)]
        apmasks = [ap.to_mask(method=method, subpixels=subpixels) for ap in apertures]

    apmask2d = np.zeros(shape, dtype=bool)

    if method == "center":
        # Use the center of the pixel to determine if it is in the aperture
        for m in apmasks:
            apmask2d |= m.to_image(shape, dtype=bool)

    elif method == "exact" or method == "subpixel":
        # Use the exact overlap of the aperture and each pixel
        for m in apmasks:
            apmask2d |= (m.to_image(shape, dtype=float) > maskmin)
    else:
        raise ValueError(f"Method {method} not supported. Use 'exact', 'center', or 'subpixel'.")

    return apmask2d


@nb.njit(fastmath=False, parallel=True)
def _enclosing_circle_radius(segm, center, segm_id, output):

    for i in nb.prange(len(segm_id)):
        _segm_id = segm_id[i]
        mask = segm == _segm_id
        y, x = np.nonzero(mask)

        # if center is None:
        #     # Calculate the centroid of the masked region
        #     center = (np.mean(x), np.mean(y))

        # Calculate the distances from the center to all non-zero pixels
        rsq_max = np.max((x - center[i][0])**2 + (y - center[i][1])**2)
        output[i] = np.sqrt(rsq_max)


def enclosing_circle_radius(segm, center, segm_id=None):
    """
    Calculate the radius of the smallest enclosing circle for a given mask.

    Parameters
    ----------
    segm : 2D array-like
        The input segmentation map (binary image) where non-zero values are
        considered as the region of interest.

    center : 2-D array, optional
        The (x, y) coordinates of the center of the circles. If not provided,
        the center will be calculated as the centroid of the masked region.

    segm_id : list of int, optional
        The list of segmentation IDs to calculate the radius for. If not provided,
        it defaults to `[1]`, which is equivalent to `True` for binary masks.

    Returns
    -------
    ndarray
        The radius of the smallest enclosing circle.

    Notes
    -----
    Since it calculates distances from the center to the pixel center, one may
    want to add ~0.5 (or sqrt(2)*0.5) to enclose the full pixel area.

    By using numba, single segmentation radius finding is ~5 times faster than
    pure numpy, and it is boosted further if `parallel=True` is used.
    """

    if segm_id is None:
        segm_id = np.array([1], dtype=segm.dtype)  # same as `True`

    center = np.atleast_2d(center)
    if center.shape[1] != 2:
        raise ValueError("Center must be a 2D array with shape (N, 2)")

    radii = np.empty(len(segm_id), dtype=np.float64)

    _enclosing_circle_radius(segm, center, segm_id, radii)

    return radii


def str_now(
    precision=3,
    fmt="{:.>72s}",
    t_ref=None,
    dt_fmt="(dt = {:.3f} s)",
    return_time=False
):
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
        Whether to return the time at the start of this function and the delta
        time (`dt`), as well as the time information string. If `t_ref` is
        `None`, `dt` is automatically set to `None`.
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
        The input to be changed to a Quantity. If a Quantity is given, `x` is
        changed to the `desired`, i.e., ``x.to(desired)``.

    desired : str or astropy Unit
        The desired unit for `x`. If `''` (default), it will be interpreted as
        `Unit(dimensionless)`.

    to_value : bool, optional.
        Whether to return as scalar value. If `True`, just the value(s) of the
        `desired` unit will be returned after conversion.

    Returns
    -------
    ux: Quantity

    Notes
    -----
    If Quantity, transform to `desired`. If `desired` is `None`, return it as
    is. If not `Quantity`, multiply the `desired`. `desired` is `None`, return
    `x` with dimensionless unscaled unit.
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
        ux = x.to(desired).value if to_value else x.to(desired)
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
        raise ValueError(
            "If you use astropy.Quantity, you should use unit convertible to `desired`."
            + f'\nYou gave "{x.unit}", unconvertible with "{desired}".'
        )

    return ux


def binning(
        arr,
        factor_x=None,
        factor_y=None,
        factors=None,
        order_xyz=True,
        binfunc=np.mean,
        trim_end=False
):
    ''' Bins the given arr frame.

    Paramters
    ---------
    arr: 2d array
        The array to be binned

    factor_x, factor_y: int or None, optional.
        The binning factors in x, y direction. This is left as legacy and for
        clarity, because mostly this function is used for 2-D CCD data. If any
        of these is given, `order_xyz` is overridden as `True`.

    factors : list-like of int, optional.
        The factors in pythonic axis order (``order_xyz=False``) or in the xyz
        order (``order_xyz=True``). If any of the tuple is `None`, that will be
        replaced by the size of the array along that axis, i.e., collapse along
        that axis.

    binfunc : funciton object
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.

    trim_end: bool
        Whether to trim the end of x, y axes such that binning is done without
        error.

    Notes
    -----
    This kind of binning is ~ 20-30 to upto 10^5 times faster than
    astropy.nddata's block_reduce:

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
    Tested on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
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
        slices = tuple(slice(None, im, None) for im in i_max)
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


def quantile_lh(
        a,
        lq,
        hq,
        axis=None,
        nanfunc=False,
        interpolation='linear',
        linterp=None,
        hinterp=None
):
    """Find quantiles for lower and higher values
    Parameters
    ----------
    a : ndarray

    lq, hq : array_like of float
        Quantile or sequence of quantiles to compute, which must be between 0
        and 1 inclusive.

    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is to
        compute the quantile(s) along a flattened version of the array.

    nanfunc : bool, optional.
        Whether to use `~np.nanquantile` instead of `~np.qualtile`.
        Default: `False`.

    interpolation, linterp, hinterp : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional.
        This optional parameter specifies the interpolation method to use when
        the desired quantile lies between two data points ``i < j``:
        * 'linear': ``i + (j - i) * fraction``, where ``fraction`` is the
          fractional part of the index surrounded by ``i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
        To tune the interpolation method for lower and higher quantiles
        individually, set `linterp` and `hinterp` separately. An idea is to use
        ``linterp='higher', hinterp='lower'`` to estimate the robust standard
        deviation estimate.
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


def quantile_sigma(
        a,
        axis=None,
        nanfunc=False,
        interpolation='linear',
        linterp=None,
        hinterp=None
):
    """ Extract "sigma" (std. dev.) from quantile to avoid bad values.

    Parameters
    ----------
    a : ndarray

    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is to
        compute the quantile(s) along a flattened version of the array.

    nanfunc : bool, optional.
        Whether to use `~np.nanquantile` instead of `~np.quantile`.
        Default: `False`.

    interpolation, linterp, hinterp : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional.
        This optional parameter specifies the interpolation method to use when
        the desired quantile lies between two data points ``i < j``:
        * 'linear': ``i + (j - i) * fraction``, where ``fraction`` is the
          fractional part of the index surrounded by ``i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
        To tune the interpolation method for lower and higher quantiles
        individually, set `linterp` and `hinterp` separately. An idea is to use
        ``linterp='higher', hinterp='lower'`` to estimate the robust standard
        deviation estimate.
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


def min_max_med_1d(arr):
    """ Return minimum, maximum and median of array.
    Tests
    -----
    up to ~ 10 times faster than numpy's min, max, median done separately (MBP
    14" [2021, macOS 13.1, M1Pro(6P+2E/G16c/N16c/32G)]) ONLY WHEN ARRAY SIZE <~
    1000:

    t1s, t2s = [], []
    for size in [10, 100, 200, 300, 500, 800, 1000, 2000, 3000]:
        a = rnd.normal(size=size)
        t1 = %timeit -o min_max_med_1d(a)
        t2 = %timeit -o a.min(), a.max(), np.median(a)
        t1s.append(t1.average)
        t2s.append(t2.average)

    [this] 984 ns ± 4.41 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    [ np ] 9.79 µs ± 47 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 1.57 µs ± 2.2 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    [ np ] 10.2 µs ± 23.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 2.54 µs ± 8.23 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [ np ] 10.6 µs ± 20.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 3.45 µs ± 15.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [ np ] 11.2 µs ± 34.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 5.44 µs ± 34.5 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [ np ] 12 µs ± 46.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 8.8 µs ± 32.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [ np ] 13 µs ± 32.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 11.7 µs ± 17.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [ np ] 14.8 µs ± 37.3 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 33.4 µs ± 4.28 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    [ np ] 19.6 µs ± 27.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    [this] 86.3 µs ± 4.78 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    [ np ] 25.6 µs ± 143 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    fig, axs = plt.subplots(1, 1, figsize=(8, 5), sharex=False, sharey=False, gridspec_kw=None)

    #axs[0].
    axs.plot( [10, 100, 200, 300, 500, 800, 1000, 2000, 3000], t1s)
    axs.plot( [10, 100, 200, 300, 500, 800, 1000, 2000, 3000], t2s)
    axs.set(
        yscale='log',
        xscale='log',
        )
    plt.tight_layout()
    plt.show();


    """
    if arr.size < 1000:
        _a = np.sort(arr)
        return _a[0], _a[-1], 0.5*(_a[_a.size//2] + _a[_a.size//2 - 1])
    else:
        return np.min(arr), np.max(arr), np.median(arr)


def mean_std_1d(arr, ddof=0, std=True, var=False):
    """ Return mean and standard deviation of array.
    Tests
    -----
    About 2.5 times faster than numpy's mean and std done separately (MBP 14"
    [2021, macOS 13.1, M1Pro(6P+2E/G16c/N16c/32G)]):

    rnd = np.random.RandomState(123)
    a = rnd.normal(size=(100))

    %timeit mean_std_1d(a)
    4.13 µs ± 15.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    %timeit np.mean(a), np.std(a)
    9.15 µs ± 41.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    print(mean_std_1d(a), "\n", np.mean(a), np.std(a))
    (0.027109073490359778, 1.1282404704779612)
     0.027109073490359778  1.128240470477961

    With ddof:
    %timeit mean_std_1d(a, ddof=1)
    4.22 µs ± 40.3 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    %timeit np.mean(a), np.std(a, ddof=1)
    9.47 µs ± 74.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    print(mean_std_1d(a, ddof=1), "\n", np.mean(a), np.std(a, ddof=1))
    (0.027109073490359778, 1.1339243375361956)
     0.027109073490359778, 1.1339243375361954

    Larger size:
    a = rnd.normal(size=(10000))
    %timeit mean_std_1d(a)
    9.38 µs ± 34.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    %timeit np.mean(a), np.std(a)
    19.2 µs ± 57.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    """
    sum_a = np.sum(arr)
    sqsum = np.sum(arr**2)
    inv_n = 1.0 / arr.size
    inv_d = 1.0 / (arr.size - ddof) if ddof > 0 else inv_n
    mean = sum_a * inv_n
    var = sqsum*inv_d - mean*sum_a*inv_d
    if var:
        if std:
            return mean, np.sqrt(var), var
        else:
            return mean, var
    else:
        if std:
            return mean, np.sqrt(var)
        else:
            raise ValueError("At least one of `std` or `var` must be True.")
