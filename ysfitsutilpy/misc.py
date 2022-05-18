'''
Objects that are
(1) too fundamental, so used in various places,
(2) completely INDEPENDENT of all other modules of this package.
'''
import sys
from collections import abc

import ccdproc
import numpy as np
from astropy import units as u
from astropy.time import Time

__all__ = ["MEDCOMB_KEYS_INT", "SUMCOMB_KEYS_INT", "MEDCOMB_KEYS_FLT32",
           "LACOSMIC_KEYS", "LACOSMIC_CRREJ", "parse_crrej_psf",
           "get_size",
           "slicefy", "bezel2slice", "is_list_like", "listify", "ndfy",
           "cmt2hdr", "update_tlm", "update_process",
           "weighted_avg", "sigclip_dataerr", "circular_mask",
           "_image_shape", "_offsets2slice",
           "str_now", "change_to_quantity", "binning",
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


# TODO: add `coord` to select whether image/physical. If physical, header is required.
def slicefy(rule, ndim=2, order_xyz=True):
    """ Parse the rule by trimsec, bezels, or slices (in this priority).

    Parameters
    ----------
    rule : str, int, list of int, list of slice, None, optional
        It can have several forms::

          * str: The FITS convention section to trim (e.g., IRAF TRIMSEC).
          * [list of] int: The number of pixels to trim from the edge of the
            image (bezel)
          * [list of] slice: The slice of each axis (`slice(start, stop,
            step)`)

        If a single int/slice is given, it will be applied to all the axes.

    order_xyz : bool, optional
        Whether the order of rule is in xyz order. Works only if the `rule` is
        bezel-like (int or list of int). If it is slice-like, `rule` must be in
        the pythonic order (i.e., ``[slice_for_axis0, slice_for_axis1, ...]``).

    >>> np.eye(5)[slicefy('[1:2,:]')]
    # array([[1., 0.],
    #       [0., 1.],
    #       [0., 0.],
    #       [0., 0.],
    #       [0., 0.]])
    >>> np.eye(5)[slicefy(1)]  # bezel by 1 pix
    # array([[1., 0., 0.],
    #    [0., 1., 0.],
    #    [0., 0., 1.]])
    >>> np.eye(5)[slicefy((1, 2))]  # bezel by (1, 1), (2, 2) pix (x/y dir)
    # array([[0., 1., 0.]])
    >>> np.eye(5)[slicefy(slice(1, -1, 2))]  # data[1:-1:2, 1:-1:2]
    # array([[1., 0.],
    #    [0., 1.]])
    """
    if rule is None:
        return tuple([slice(None, None, None) for _ in range(ndim)])
    elif isinstance(rule, str):
        fs = np.atleast_1d(rule)
        sl = [ccdproc.utils.slices.slice_from_string(sect, fits_convention=True)
              for sect in fs]
        return sl[0] if len(sl) == 1 else tuple(sl)
    elif is_list_like(rule):
        if isinstance(rule[0], slice):
            return ndfy(rule, ndim)
        else:  # bezels
            bezels = ndfy([ndfy(b, 2, default=0) for b in listify(rule)], ndim)
            return bezel2slice(bezels, order_xyz=order_xyz)
    elif isinstance(rule, slice):
        return ndfy(rule, ndim)
    elif isinstance(rule, int):  # bezels
        bezels = ndfy([ndfy(b, 2, default=0) for b in listify(rule)], ndim)
        return bezel2slice(bezels, order_xyz=order_xyz)
    else:
        raise TypeError(f"rule must be a string or a list of int/slice. Now {type(rule)=}")


def bezel2slice(bezels, order_xyz=True):
    """ Convert bezels to slice objects

    Parameters
    ----------
    bezels : list of list of int, optional.
        Must be a list of list of int. Each list of int is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.

    order_xyz : bool, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``). Due to its confusing behavior, it is intended to
        be `True` most of the time.
        Default: `True`.

    Notes
    -----
    Consider a 100x100 image.
    1. ``bezels = [[10, 20], [30, 40]], order_xyz=True`` will ignore the first
       10 columns, the last 20 columns, the **BOTTOM** 30 rows, and the **TOP**
       40 rows.
    2. ``bezels = [[10, 20], [30, 40]], order_xyz=False`` will ignore the
       **BOTTOM** 10 rows (python index of ``[:10]``), the **TOP** 20 columns
       (python index of ``[-20:]``), the first 30 columns, and the last 40
       columns.
    This confusing behavior is due to the (stupid and/or inconsistent?) way
    our world represents xy-coordinates.
    """
    bezels = np.atleast_2d(bezels)
    bezels = bezels[::-1] if order_xyz else bezels
    return tuple([slice(b[0], None if b[1] == 0 else -b[1]) for b in bezels])


def is_list_like(*objs, allow_sets=True, func=all):
    ''' Check if inputs are list-like

    Parameters
    ----------
    *objs : object
        Objects to check.
    allow_sets : bool, optional.
        If this parameter is `False`, sets will not be considered list-like.
        Default: `True`
    func : funtional object, optional.
        The function to be applied to each element. Useful ones are `all` and
        `any`.
        Default: `all`

    Notes
    -----
    Direct copy from pandas, with slight modification to accept *args and
    all/any, etc, functionality by `func`.
    https://github.com/pandas-dev/pandas/blob/bdb00f2d5a12f813e93bc55cdcd56dcb1aae776e/pandas/_libs/lib.pyx#L1026

    Note that pd.DataFrame also returns True.

    Timing on MBP 14" [2021, macOS 12.2, M1Pro(6P+2E/G16c/N16c/32G)]
    %timeit yfu.is_list_like("asdfaer.fits")
    4.32 µs +- 572 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    '''
    # I don't think we need speed boost here but...
    # if `func` is `any`, below can be reorganized by for loop and can return
    # `True` once it reaches `True` for the first time.
    return func(
        isinstance(obj, abc.Iterable)
        # we do not count strings/unicode/bytes as list-like
        and not isinstance(obj, (str, bytes))
        # exclude zero-dimensional numpy arrays, effectively scalars
        and not (isinstance(obj, np.ndarray) and obj.ndim == 0)
        # exclude sets if allow_sets is False
        and not (allow_sets is False and isinstance(obj, abc.Set))
        for obj in objs
    )


# def listify(obj):
#     """Make an object into a list.

#     Parameters
#     ----------
#     obj : None, str, list-like
#         Object to be made into a list. If `str`, it will be converted to
#         ``[obj]``. If `None`, an empty list (`[]`) is returned.
#     """
#     if obj is None:
#         return []
#     elif is_list_like(obj):
#         return list(obj)
#     else:
#         return [obj]


def listify(*objs, scalar2list=True, none2list=False):
    """Make multiple object into list of same length.

    Parameters
    ----------
    objs : None, str, list-like
        If single object, it will be converted to a list ``[obj]`` or ``obj``,
        depending on `scalar2list`. Any scalar input will be converted to a
        list of a target length (largest length among `objs`). If `None`, an
        empty list (`[]`) or ``[None]`` is returned depending on `none2list`.
        If multiple objects are given, maximum length of them is used as the
        target length.
    scalar2list : bool, optional.
        If `True`, a single scalar input will be converted to a list of a
        target length. Otherwise, it will be returned as is.

    none2list : bool, optional.
        Whether to return an empty list (`[]`). If `True`, ``[None]`` is
        returned if `objs` is `None`.
        Default: `False`

    Notes
    -----
    If any obj of `None` need to be converted to a length>1 list, it will be
    made as [None, None, ...], rather than an empty list, regardless of
    `empty_if_none`.

    Tests
    -----
    assert listify(12) == [12]
    assert listify([1]) == [1]
    assert listify(12, scalar2list=False) == 12
    assert listify([1], scalar2list=False) == [1]
    assert listify(None) == []
    assert listify(None, none2list=True) == [None]
    assert listify([1, 2]) == [1, 2]
    assert listify([1, "a"]) == [1, "a"]
    with pytest.raises(ValueError):
        listify([1, 2], "a", [3, 4, 5])
    # Below are the most important cases `listify` is useful
    assert listify("ab") == ["ab"]
    assert listify("ab", scalar2list=False) == "ab"
    assert listify([1, 2], "a") == [[1, 2], ['a', 'a']]
    assert listify([1, 2], "a", None) == [[1, 2], ['a', 'a'], [None, None]]

    Timing on MBP 14" [2021, macOS 12.2, M1Pro(6P+2E/G16c/N16c/32G)]:
    %timeit yfu.listify([12])
    8.92 µs +- 434 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    %timeit yfu.listify("asdf")
    7.08 µs +- 407 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    %timeit yfu.listify("asdf", scalar2list=False)
    7.37 µs +- 586 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)

    """
    def _listify_single(obj, none2list=True):
        if obj is None:
            return [obj] if none2list else []
        elif is_list_like(obj):
            return list(obj)
        else:
            return [obj] if scalar2list else obj

    if len(objs) == 1:
        return _listify_single(objs[0], none2list=none2list)

    objlists = [_listify_single(obj, none2list=True) for obj in objs]
    lengths = [len(obj) for obj in objlists]
    length = max(lengths)
    for objl in objlists:
        if len(objl) not in [1, length]:
            raise ValueError(f"Each input must be 1 or max(lengths)={length}.")

    return [obj*length if len(obj) == 1 else obj for obj in objlists]


def ndfy(item, length=None, default=0):
    """ Make an item to ndarray of length.

    Parameters
    ----------
    item : None, general object, list-like
        The item to be made into an ndarray. If `None`, `default` is used.

    length : int, optional.
        The length of the final ndarray. If `None`, the length of the input is
        used (if `item` is a scalar, a length-1 array is returned).

    default : general object
        The default value to be used if `item` or any element of `item` is
        `None`.

    Notes
    -----
    Useful for the cases when bezels, sigma, ... are needed. Example case when
    you want to make ((20, 20), (20, 20)) bezels from given ``bezels = 20``::
    >>> arr = np.arange(10).reshape(2, 5)  # 2-D array
    >>> bezels1 = 20
    >>> bezels2 = (20, 20)
    >>> bezels3 = ((20, 20), (20, 20))
    >>> ans = [[20, 20], [20, 20]]
    >>> assert ndfy([ndfy(b, 2, default=0) for b in listify(bezels1)], arr.ndim) == ans
    >>> assert ndfy([ndfy(b, 2, default=0) for b in listify(bezels2)], arr.ndim) == ans
    >>> assert ndfy([ndfy(b, 2, default=0) for b in listify(bezels3)], arr.ndim) == ans

    Note that 2-element bezels is ambiguous: ``[a, b]`` may mean either (1)
    both bezels in x-axis to be ``a`` and both bezels in y-axis to be ``b`` or
    (2) both x-/y-axis have bezels ``[a, b]`` for lower/upper regions. `ndfy`
    uses the first assumption::
    >>> ndfy([ndfy(b, 2, default=0) for b in listify([(30, 40)])], arr.ndim)
    >>> # [[30, 30], [40, 40]]
    Because of this, it will raise ValueError if `arr.ndim != bezels.size`.
    """
    item = listify(item)
    item = [default if i is None else i for i in item]

    if (length is None) or (len(item) == length):
        return item
    elif len(item) != 1:
        raise ValueError(f"Length of item must be 1 or {length=}. Now it is {len(item)}.")

    # Now, len(item) == 1
    return [item[0] for _ in range(length)]


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
            after=f"NAXIS{header['NAXIS']}",
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

    return val,


def circular_mask(shape, center=None, radius=None, center_xyz=True):
    ''' Creates an N-D circular (circular, sphereical, ...) mask.

    Parameters
    ----------
    shape : tuple
        The pythonic shape (not xyz order).

    center : tuple, None, optional.
        The center of the circular mask. If `None` (default), the central
        position is used.

    radius : float, None, optional.
        The radius of the mask. If `None`, the distance to the closest edge of
        the image is used.

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
    """ Makes offsets all non-negative and relative to each other.
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
        shapes,
        offsets,
        method='outer',
        offset_order_xyz=True,
        intify_offsets=False,
        pythonize_offsets=True
):
    '''shapes and offsets must be in the order of python/numpy (i.e., z, y, x order).

    Paramters
    ---------
    shapes : ndarray
        The shapes of the arrays to be processed. It must have the shape of
        ``nimage`` by ``ndim``. The order of shape must be pythonic (i.e.,
        ``shapes[i] = image[i].shape``, not in the xyz order).

    offsets : ndarray
        The offsets must be ``(cen_i - cen_ref) + const`` format, i.e., an
        offset is the position of the target frame (position can be, e.g.,
        origin or center) in the coordinate of the reference frame, with a
        possible non-zero constant offset applied. It must have the shape of
        ``nimage`` by ``ndim``.

    method : str, optional
        The method to calculate the `shape_out`::

          * ``'outer'``: To combine images, so every pixel in `shape_out` has
            at least 1 image pixel.
          * ``'inner'``: To process only where all the images have certain
            pixel (fully-overlap).

    offset_order_xyz : bool, optional
        Whether `offsets` are in xyz order. If so, those will be flipped to
        pythonic order. Default: `True`

    Returns
    -------
    _offsets : ndarray
        The *relative* offsets calculated such that at least one image per each
        dimension must have offset of 0.

    shape_out : tuple
        The shape of the array depending on the `method`.
    '''
    _offsets = _regularize_offsets(
        offsets,
        offset_order_xyz=offset_order_xyz,
        intify_offsets=intify_offsets
    )

    if method == 'outer':
        shape_out = np.around(np.max(np.array(shapes) + _offsets, axis=0)).astype(int)
        # print(_offsets, shapes, shape_out)
    # elif method == 'stack':
    #     shape_out_comb = np.around(
    #         np.max(np.array(shapes) + _offsets, axis=0)
    #     ).astype(int)
    #     shape_out = (len(shapes), *shape_out_comb)
    elif method == 'inner':
        lower_bound = np.max(_offsets, axis=0)
        upper_bound = np.min(_offsets + shapes, axis=0)
        npix = upper_bound - lower_bound
        shape_out = np.around(npix).astype(int)
        if np.any(npix < 0):
            raise ValueError("There doesn't exist fully-overlapping pixel! "
                             + f"Naïve output shape={shape_out}.")
        # print(lower_bound, upper_bound, shape_out)
    else:
        raise ValueError("method unacceptable (use one of 'inner', 'outer').")

    if offset_order_xyz and not pythonize_offsets:
        # reverse _offsets to original xyz order
        _offsets = np.flip(_offsets, -1)

    return _offsets, tuple(shape_out)


def _offsets2slice(
        shapes,
        offsets,
        method='outer',
        shape_order_xyz=False,
        offset_order_xyz=True,
        outer_for_stack=True,
        fits_convention=False
):
    """ Calculates the slices for each image when to extract overlapping parts.

    Parameters
    ----------
    shapes, offsets : ndarray
        The shape and offset of each image. If multiple images are used, it
        must have shape of ``nimage`` by ``ndim``.

    method : str, optional
        The method to calculate the `shape_out`::

          * ``'outer'``: To combine images, so every pixel in `shape_out` has
            at least 1 image pixel.
          * ``'inner'``: To process only where all the images have certain
            pixel (fully-overlap).

    shape_order_xyz, offset_order_xyz : bool, optional.
        Whether the order of the shapes or offsets are in xyz or pythonic.
        Shapes are usually in pythonic as it is obtained by
        ``image_data.shape``, but offsets are often in xyz order (e.g., if
        header ``LTVi`` keywords are loaded in their alpha-numeric order; or
        you have used `~calc_offset_wcs` or `~calc_offset_physical` with
        default ``order_xyz=True``). Default is `False` and `True`,
        respectively.

    outer_for_stack : bool, optional.
        If `True`(default), the output slice is the slice in tne ``N+1``-D
        array, which will be constructed before combining them along
        ``axis=0``. That is, ``comb = np.nan*np.ones(_image_shape(shapes,
        offsets, method='outer'))`` and ``comb[slices[i]] = images[i]``. Then a
        median combine, for example, is done by ``np.nanmedian(comb, axis=0)``.
        If ``stack_outer=False``, ``slices[i]`` will be
        ``slices_with_stack_outer_True[i][1:]``.

    fits_convention : bool, optional.
        Whether to return the slices in FITS convention (xyz order, 1-indexing,
        end index included). If `True` (default), returned list contains str;
        otherwise, slice objects will be contained.

    Returns
    -------
    slices : list of str or list of slice
        The meaning of it differs depending on `method`::

          * ``'outer'``: the slice of the **output** array where the i-th image
            should fit in.
          * ``'inner'``: the slice of the **input** array (image) where the
            overlapping region resides.

    Example
    -------
    >>>


    """
    _shapes = np.atleast_2d(shapes)
    if shape_order_xyz:
        _shapes = np.flip(_shapes, -1)

    _offsets = _regularize_offsets(
        offsets,
        offset_order_xyz=offset_order_xyz,
        intify_offsets=True
    )

    if _shapes.ndim != 2 or _offsets.ndim != 2:
        raise ValueError("Shapes and offsets must be at most 2-D.")

    if _shapes.shape != _offsets.shape:
        raise ValueError("shapes and offsets must have the identical shape.")

    if method == 'outer':
        starts = _offsets
        stops = _offsets + _shapes
        if outer_for_stack:
            def _initial_tmp(i):
                return [f"{i + 1}:{i + 1}"] if fits_convention else [slice(i, i + 1, None)]
        else:
            _initial_tmp = lambda i: []  # initialized empty list regardless of argument
    elif method == 'inner':
        offmax = np.max(_offsets, axis=0)
        if np.any(np.min(_shapes + _offsets, axis=0) <= offmax):
            raise ValueError(
                "At least 1 frame has no overlapping pixel with all others. "
                + "Check if there's any overlapping pixel for images for the given offsets."
            )

        # 1-D array +/- 2-D array:
        #   the former 1-D array is broadcast s.t. it is "tile"d along axis=-1.
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
