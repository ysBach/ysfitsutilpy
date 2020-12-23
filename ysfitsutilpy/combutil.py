from pathlib import Path
from warnings import warn

import ccdproc
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
from astropy.time import Time
from ccdproc import combine, trim_image

from .ccdutil import CCDData_astype, trim_ccd
from .filemgmt import load_if_exists, make_summary
from .hdrutil import add_to_header
from .misc import chk_keyval, load_ccd


__all__ = ["sstd", "weighted_mean", "group_FITS", "stack_FITS", "combine_ccd"]


def sstd(a, **kwargs):
    ''' Sample standard deviation function
    '''
    return np.std(a, ddof=1, **kwargs)

# FIXME: Add this to Ccdproc esp. for mem_limit
def weighted_mean(ccds, unit='adu'):
    datas = []
    ws = []  # weights = 1 / sigma**2
    for ccd in ccds:
        datas.append(ccd.data)
        ws.append(1 / ccd.uncertainty.array**2)
    wmean = np.average(np.array(datas), axis=0, weights=ws)
    wuncert = np.sqrt(1 / np.sum(np.array(ws), axis=0))
    nccd = CCDData(data=wmean, header=ccds[0].header, unit=unit)
    nccd.uncertainty = StdDevUncertainty(wuncert)
    return nccd


def group_FITS(summary_table, type_key=None, type_val=None, group_key=None):
    ''' Organize the group_by and type_key for stack_FITS
    Parameters
    ----------
    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata (header) of files. If it is in the astropy table format,
        it will be converted to `~pandas.DataFrame` object.

    type_key, type_val: None, str, list of str, optional
        The header keyword for the ccd type, and the value you want to match.

    group_key : None, str, list of str, optional
        The header keyword which will be used to make groups for the CCDs that have selected from
        ``type_key`` and ``type_val``. If `None` (default), no grouping will occur, but it will return
        the `~pandas.DataFrameGroupBy` object will be returned for the sake of consistency.

    Return
    ------
    grouped : ~pandas.DataFrameGroupBy
        The table after the grouping process.

    group_type_key : list of str
        The ``type_key`` that can directly be used for ``stack_FITS`` for each element of
        ``grouped.groups``. Basically this is ``type_key + group_key``.

    Example
    -------
    >>> allfits = list(Path('.').glob("*.fits"))
    >>> summary_table = make_summary(allfits)
    >>> type_key = ["OBJECT"]
    >>> type_val = ["dark"]
    >>> group_key = ["EXPTIME"]
    >>> gs, g_key = group_FITS(summary_table,
    ...                        type_key,
    ...                        type_val,
    ...                        group_key)
    >>> for g_val, group in gs:
    >>>     _ = combine_ccd(group["file"],
    ...                     type_key=g_key,
    ...                     type_val=g_val)
    '''
    if (not isinstance(summary_table, Table)) and (not isinstance(summary_table, pd.DataFrame)):
        raise TypeError(
            f"summary_table must be an astropy Table or Pandas DataFrame. It's now {type(summary_table)}."
        )
    elif isinstance(summary_table, Table):
        st = summary_table.to_pandas()
    else:
        st = summary_table.copy()

    type_key, type_val, group_key = chk_keyval(type_key=type_key, type_val=type_val, group_key=group_key)

    if len(group_key + type_key) == 0:
        raise ValueError("At least one of type_key and group_key should not be empty!")

    # For simplicity, crop the original data by type_key and type_val first.
    for k, v in zip(type_key, type_val):
        st = st[st[k] == v]

    group_type_key = type_key + group_key
    grouped = st.groupby(group_key)

    return grouped, group_type_key


def stack_FITS(fitslist=None, summary_table=None, extension=None,
               unit=None, table_filecol="file", trim_fits_section=None,
               ccddata=True, asccd=True, type_key=None, type_val=None,
               verbose=True):
    ''' Stacks the FITS files specified in fitslist

    Parameters
    ----------
    fitslist: None, list of path-like, or list of CCDData
        The list of path to FITS files or the list of CCDData to be stacked. It is useful to give list
        of CCDData if you have already stacked/loaded FITS file into a list by your own criteria. If
        `None` (default), you must give ``fitslist`` or ``summary_table``. If it is not `None`,
        this function will do very similar job to that of ``ccdproc.combine``. Although it is not a
        good idea, a mixed list of CCDData and paths to the files is also acceptable.

    summary_table: None, pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many FITS files and you want to
        use stacking many times, it is better to make a summary table by ``filemgmt.make_summary`` and
        use that instead of opening FITS files' headers every time you call this function. If you want
        to use ``summary_table`` instead of ``fitslist`` and have set ``ccddata=True``, you must not
        have `None` or ``NaN`` value in the ``summary_table[table_filecol]``.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    unit: Unit or str, optional
        The unit of the CCDs to be loaded.
        Used only when ``fitslist`` is not a list of ``CCDData`` and ``ccddata`` is `True`.

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to the FITS files.

    trim_fits_section : str or None, optional
        The ``fits_section`` of ``ccdproc.trim_image``. Region of ``ccd`` from which the overscan is
        extracted; see `~ccdproc.subtract_overscan` for details.
        Default is `None`.

    ccddata: bool, optional
        Whether to return file paths or loaded CCDData. If `False`, it is a function to select FITS
        files using ``type_key`` and ``type_val`` without using much memory.
        This is ignored if ``fitslist`` is given and composed of ``CCDData`` objects.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.

    asccd : bool, optional.
        Whether to load as ``astropy.nddata.CCDData``. If `False`, numpy ndarray will be used. Works
        only if ``ccddata = True``.

    Return
    ------
    matched: list of Path or list of CCDData
        list containing Path to files if ``ccddata`` is `False`. Otherwise it is a list containing
        loaded CCDData after loading the files. If ``ccdlist`` is given a priori, list of CCDData will
        be returned regardless of ``ccddata``.
    '''
    def _parse_val(value):
        val = str(value)
        if val.lstrip('+-').isdigit():  # if int
            result = int(val)
        else:
            try:
                result = float(val)
            except ValueError:
                result = str(val)
        return result

    def _check_mismatch(row):
        mismatch = False
        for k, v in zip(type_key, type_val):
            hdr_val = _parse_val(row[k])
            parse_v = _parse_val(v)
            if (hdr_val != parse_v):
                mismatch = True
                break
        return mismatch

    if ((fitslist is not None) + (summary_table is not None) != 1):
        raise ValueError("One and only one of fitslist or summary_table must be not None.")

    # Check for type_key and type_val
    type_key, type_val, _ = chk_keyval(type_key=type_key, type_val=type_val, group_key=None)

    # Setting whether we have to select a subset from the list
    selecting = True if len(type_key) > 0 else False

    if verbose:
        print("Analyzing FITS... ", end='')

    # ****************************************************************************************************** #
    # *                                   MAKE FITSLIST AND SUMMARY_TABLE                                  * #
    # ****************************************************************************************************** #
    # == If fitslist ======================================================================================= #
    if fitslist is not None:
        try:
            fitslist = list(fitslist)
        except TypeError:
            raise TypeError(f"fitslist must be convertable to list. It's now {type(fitslist)}.")

        if selecting:
            summary_table = make_summary(
                fitslist,
                extension=extension,  # extension will be parsed within make_summary (no need to care here)
                verbose=verbose,
                fname_option='relative',
                keywords=type_key,
                sort_by=None,
                pandas=True,
            )
        # else:
        #   no need to make summary_table.
    # == If summary_table ================================================================================== #
    elif summary_table is not None:
        is_astropytab = isinstance(summary_table, Table)
        is_dataframe = isinstance(summary_table, pd.DataFrame)
        if (not is_astropytab) and (not is_dataframe):
            raise TypeError("summary_table must be an astropy Table or Pandas DataFrame. "
                            + f"It's now {type(summary_table)}.")

        if is_astropytab:
            summary_table = summary_table.to_pandas()
        try:
            summary_table.reset_index(inplace=True)
        except ValueError:
            pass
        fitslist = summary_table[table_filecol].tolist()

    if verbose:
        print("Done", end='')
        if ccddata:
            print(" and loading FITS... ")
        else:
            print(".")

    # ****************************************************************************************************** #
    # *                                       SELECT AND LOAD TO MATCHED                                   * #
    # ****************************************************************************************************** #
    matched = []
    if selecting:
        # == Select FITS based on type_key and type_val ==================================================== #
        for i, row in summary_table.iterrows():
            # I intentionally used iterrows instead of making mask, because for some cases the keyword
            # (e.g., an angle) can contain both str and float among CCDs.
            #    For example, if we want to select ``angle == 0.0``, masking cannot work because the
            # column has dtype of object (``summary_table[column].dtype`` is ``object```).
            #    Instead, _check_mismatch tries to convert the value found in the header to int, and
            # if it fails, tries float, and finally uses str. This is the most natural way I could
            # think of.
            # ysBach, 2020-05-15 09:44:13 (KST: GMT+09:00)
            mismatch = _check_mismatch(row)
            if mismatch:  # skip this row (file)
                continue

            # if not skipped:
            # TODO: Is it better to remove Path here?
            if isinstance(fitslist[i], CCDData):
                if asccd:
                    matched.append(fitslist[i])
                else:
                    matched.append(fitslist[i].data)
            else:  # it must be a path to a file
                fpath = Path(fitslist[i])
                if ccddata:
                    # extension will be parsed within load_ccd (no need to care here)
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(ccd_i, fits_section=trim_fits_section)
                    if asccd:
                        matched.append(ccd_i)
                    else:
                        matched.append(ccd_i.data)
                else:
                    matched.append(fpath)
    else:
        # == Use all item in fitslist ====================================================================== #
        # summary_table is not used.
        for item in fitslist:
            if isinstance(item, CCDData):
                if asccd:
                    matched.append(item)
                else:
                    matched.append(item.data)
            else:  # it must be a path to a file
                if ccddata:
                    # extension will be parsed within load_ccd (no need to care here)
                    ccd_i = load_ccd(item, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(ccd_i, fits_section=trim_fits_section)
                    if asccd:
                        matched.append(ccd_i)
                    else:
                        matched.append(ccd_i.data)
                else:  # TODO: Is is better to remove Path here?
                    matched.append(Path(item))

    # ****************************************************************************************************** #
    # *                                     PRINT INFO MESSAGE OR WARNING                                  * #
    # ****************************************************************************************************** #

    if len(matched) == 0:
        if selecting:
            warn(f'No FITS file had "{str(type_key)} = {str(type_val)}" Maybe int/float/str confusing?')
        else:
            warn('No FITS file found')
    else:
        if selecting:
            N = len(matched)
            ks = str(type_key)
            vs = str(type_val)
            if verbose:
                if ccddata:
                    print(f'{N} FITS files with "{ks} = {vs}" are loaded.')
                else:
                    print(f'{N} FITS files with "{ks} = {vs}" are selected.')
        else:
            if verbose and ccddata:
                print('{:d} FITS files are loaded.'.format(len(matched)))

    return matched


def combine_ccd(fitslist=None, summary_table=None, table_filecol="file",
                trim_fits_section=None, output=None, unit=None,
                subtract_frame=None, combine_method='median',
                reject_method=None, normalize_exposure=False,
                normalize_average=False, normalize_median=False,
                exposure_key='EXPTIME', mem_limit=2e9,
                combine_uncertainty_function=None,
                extension=None, type_key=None, type_val=None,
                dtype="float32", uncertainty_dtype="float32",
                output_verify='fix', overwrite=False,
                verbose=True, **kwargs):
    ''' Combining images
    Slight variant from ccdproc.
    # TODO: accept the input like ``sigma_clip_func='median'``, etc.
    Parameters
    ----------
    fitslist: path-like, list of path-like, or list of CCDData
        The list of path to FITS files or the list of CCDData to be stacked. It is useful to give list
        of CCDData if you have already stacked/loaded FITS file into a list by your own criteria. If
        `None` (default), you must give ``fitslist`` or ``summary_table``. If it is not `None`, this
        function will do very similar job to that of ``ccdproc.combine``. Although it is not a good
        idea, a mixed list of CCDData and paths to the files is also acceptable.

    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many FITS files and you want to
        use stacking many times, it is better to make a summary table by ``filemgmt.make_summary`` and
        use that instead of opening FITS files' headers every time you call this function. If you want
        to use ``summary_table`` instead of ``fitslist`` and have set ``ccddata=True``, you must not
        have `None` or ``NaN`` value in the ``summary_table[table_filecol]``.

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to the FITS files.

    trim_fits_section : str or None, optional
        The ``fits_section`` of ``ccdproc.trim_image``. Region of ``ccd`` from which the overscan is
        extracted; see `~ccdproc.subtract_overscan` for details.
        Default is `None`.

    output : path-like or None, optional.
        The path if you want to save the resulting ``ccd`` object.
        Default is `None`.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is `None`.

    subtract_frame : array-like, optional.
        The frame you want to subtract from the image after the combination. It can be, e.g., dark
        frame, because it is easier to calculate Poisson error before the dark subtraction and subtract
        the dark later.
        TODO: This maybe unnecessary.
        Default is `None`.

    combine_method : str or None, optinal.
        The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median', 'sum'}
        Default is `None`.

    reject_method : str
        Made for simple use of ``ccdproc.combine``, [None, 'minmax', 'sigclip' == 'sigma_clip',
        'extrema' == 'ext']. Automatically turns on the option, e.g., ``clip_extrema = True`` or
        ``sigma_clip = True``. Leave it blank for no rejection.
        Default is `None`.

    normalize_exposure : bool, optional.
        Whether to normalize the values by the exposure time of each frame before combining.
        Default is `False`.

    normalize_average, normalize_median : bool, optional.
        Whether to normalize the values by the average or median value of each frame before combining.
        Only up to one of these must be True.
        Default is `False`.

    exposure_key : str, optional
        The header keyword for the exposure time.
        Default is ``"EXPTIME"``.

    combine_uncertainty_function : callable, None, optional
        The uncertainty calculation function of ``ccdproc.combine``. If `None` use the default
        uncertainty func when using average, median or sum combine, otherwise use the function
        provided.
        Default is `None`.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    dtype : str or `numpy.dtype` or None, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter description. If `None` it
        uses ``np.float64``.
        Default is `None`.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match. For an open HDU named
        ``hdu``, e.g., only the files which satisfies ``hdu[extension].header[type_key] == type_val``
        among all the ``fitslist`` will be used.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``, ``"ignore"``,
        ``"warn"``, or ``"exception"``. May also be any combination of ``"fix"`` or ``"silentfix"``
        with ``"+ignore"``, ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    mem_limit : float, optional
        Maximum memory which should be used while combining (in bytes).
        Default is ``2.e9``.

    **kwarg:
        kwargs for the ``ccdproc.combine``. See its documentation. This includes (RHS are the default
        values)
        ```
        weights=None,
        scale=None,
        mem_limit=16000000000.0,
        clip_extrema=False,
        nlow=1,
        nhigh=1,
        minmax_clip=False,
        minmax_clip_min=None,
        minmax_clip_max=None,
        sigma_clip=False,
        sigma_clip_low_thresh=3,
        sigma_clip_high_thresh=3,
        sigma_clip_func=<numpy.ma.core._frommethod instance>,
        sigma_clip_dev_func=<numpy.ma.core._frommethod instance>,
        combine_uncertainty_function=None, **ccdkwargs
        ```

    Returns
    -------
    master: astropy.nddata.CCDData
        Resulting combined ccd.
    '''
    # def _normalize_exptime(ccdlist, exposure_key):
    #     _ccdlist = ccdlist.copy()
    #     exptimes = []
    #     for i in range(len(_ccdlist)):
    #         exptime = _ccdlist[i].header[exposure_key]
    #         exptimes.append(exptime)
    #         _ccdlist[i] = _ccdlist[i].divide(exptime)
    #     if verbose:
    #         if len(np.unique(exptimes)) != 1:
    #             print('There are more than one exposure times:\n\t', end=' ')
    #             print(np.unique(exptimes), end=' ')
    #             print('seconds')
    #         print(f'Normalized images by exposure time ("{exposure_key}").')
    #     return _ccdlist

    def _set_reject_method(reject_method):
        ''' Convenience function for ccdproc.combine reject switches
        '''
        clip_extrema, minmax_clip, sigma_clip = False, False, False

        if reject_method in ['extrema', 'ext']:
            clip_extrema = True
        elif reject_method in ['minmax']:
            minmax_clip = True
        elif reject_method in ['sigma_clip', 'sigclip']:
            sigma_clip = True
        else:
            if reject_method not in [None, 'no']:
                raise KeyError("reject not in [None, 'minmax', sigclip'=='sigma_clip', 'extrema'=='ext']")

        return clip_extrema, minmax_clip, sigma_clip

    # def _print_info(combine_method, Nccd, reject_method, **kwargs):
    #     if reject_method is None:
    #         reject_method = 'no'

    #     info_str = ('"{:s}" combine {:d} images by "{:s}" rejection')

    #     print(info_str.format(combine_method, Nccd, reject_method))
    #     print(dict(**kwargs))
    #     return

    def _add_and_print(s, header, verbose):
        header.add_history(s)
        if verbose:
            print(s)

    # Give only one
    if ((fitslist is not None) + (summary_table is not None) != 1):
        raise ValueError("One and only one of [fitslist, summary_table] must be given.")

    # If fitslist
    if fitslist is not None:
        # == a single CCDData ============================================================================== #
        if isinstance(fitslist, CCDData):
            fitslist = [fitslist]
        else:
            # == a single path-like ======================================================================== #
            try:
                fitslist = [Path(fitslist)]
            except TypeError:
                # == a list of path-like or CCDData ======================================================== #
                try:
                    fitslist = list(fitslist)
                except TypeError:
                    raise TypeError(f"fitslist must be convertable to list. It's now {type(fitslist)}.")

    # If summary_table
    if summary_table is not None:
        if (not isinstance(summary_table, Table)) and (not isinstance(summary_table, pd.DataFrame)):
            raise TypeError(
                f"summary_table must be an astropy Table or Pandas DataFrame. It's now {type(summary_table)}."
            )

    # Check for type_key and type_val
    if ((type_key is None) ^ (type_val is None)):
        raise ValueError("type_key and type_val must be both specified or both None.")

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            if verbose:
                print(f"{output} already exists:\n\tBut will be overridden.")
        else:
            if verbose:
                print(f"{output} already exists:")
            return load_if_exists(output, loader=CCDData.read, if_not=None)

    # Do we really need to accept all three of normalize & scale?
    # if scale is None:
    #     scale = np.ones(len(ccdlist))
    if (((normalize_average) + (normalize_exposure) + (normalize_median)) > 1):
        raise ValueError(
            "Only up to one of [normalize_average, normalize_exposure, normalize_median] is acceptable.")

    # Set history messages
    str_history = ('{:d} images with {:s} = {:s} are "{:s}" combined '
                   + 'using "{:s}" rejection (additional kwargs: {})')
    str_nexp = "Each frame will be normalized by exposure time before combine."
    str_navg = "Each frame will be normalized by average before combine."
    str_nmed = "Each frame will be normalized by median before combine."
    str_subt = "Subtracted a user-provided frame"

    if reject_method is None:
        reject_method = 'no'

    # Select CCDs by
    ccdlist = stack_FITS(
        fitslist=fitslist,
        summary_table=summary_table,
        table_filecol=table_filecol,
        extension=extension,  # extension will be parsed within make_summary/load_ccd (no need to care here)
        unit=unit,
        type_key=type_key,
        type_val=type_val,
        ccddata=False,
        verbose=verbose
    )
    #  trim_fits_section=trim_fits_section,
    # ccddata=False: Loading CCD here may cause memory blast...

    try:
        header = ccdlist[0].header
    except AttributeError:
        header = fits.getheader(ccdlist[0])

    # if verbose:
    #     _print_info(
    #         combine_method=combine_method,
    #         Nccd=len(ccdlist),
    #         reject_method=reject_method,
    #         dtype=dtype,
    #         **kwargs)

    _t = Time.now()
    scale = None
    # Normalize by exposure
    # TODO: Let it accept summary table as well as fitslist
    if normalize_exposure:
        tmp = make_summary(fitslist=fitslist,
                           keywords=[exposure_key],
                           verbose=False,
                           sort_by=None)
        exptimes = tmp[exposure_key].tolist()
        scale = 1 / np.array(exptimes)
        add_to_header(header, 'h', str_nexp, verbose=verbose)

    # Normalize by pixel average
    if normalize_average:
        def invavg(a):
            return 1 / np.mean(a)
        scale = invavg
        add_to_header(header, 'h', str_navg, verbose=verbose)

    # Normalize by pixel median
    if normalize_median:
        def invmed(a):
            return 1 / np.median(a)
        scale = invmed
        add_to_header(header, 'h', str_nmed, verbose=verbose)

    # Set rejection switches
    clip_extrema, minmax_clip, sigma_clip = _set_reject_method(reject_method)

    if len(ccdlist) == 1:
        if isinstance(ccdlist[0], CCDData):
            master = ccdlist[0]
        else:
            # extension will be parsed within load_ccd (no need to care here)
            master = load_ccd(ccdlist[0], extension=extension, unit=unit)
    else:
        master = combine(
            img_list=ccdlist,
            method=combine_method,
            clip_extrema=clip_extrema,
            minmax_clip=minmax_clip,
            sigma_clip=sigma_clip,
            mem_limit=mem_limit,
            combine_uncertainty_function=combine_uncertainty_function,
            unit=unit,  # user-given unit is already applied by stack_FITS
            hdu=extension,
            scale=scale,
            dtype=dtype,
            **kwargs)

    header["COMBVER"] = (ccdproc.__version__,
                         "ccdproc version used for combine.")
    # NCOMBINE from ccdproc has no comment so I duplicate this...
    ncombine = len(ccdlist)
    header["NCOMBINE"] = (ncombine, "Number of combined images")
    header["COMBMETH"] = (combine_method, "Combining method")

    s = str_history.format(ncombine,
                           str(type_key),
                           str(type_val),
                           str(combine_method),
                           str(reject_method),
                           kwargs)
    add_to_header(header, 'h', s, verbose=verbose, t_ref=_t)
    # header.add_history(str_history.format(ncombine,
    #                                       str(type_key),
    #                                       str(type_val),
    #                                       str(combine_method),
    #                                       str(reject_method),
    #                                       kwargs))

    if subtract_frame is not None:
        _t = Time.now()
        subtract = CCDData(subtract_frame.copy())
        master.data = master.subtract(subtract).data
        add_to_header(header, 'h', str_subt, header, verbose=verbose, t_ref=_t)

    if trim_fits_section is not None:
        master = trim_ccd(master, fits_section=trim_fits_section,
                          verbose=verbose)

    master.header = header
    master = CCDData_astype(master, dtype=dtype,
                            uncertainty_dtype=uncertainty_dtype)
    # update_tlm is done incide CCDData_astype

    if output is not None:
        if verbose:
            print(f"Writing FITS to {output}... ", end='')
        master.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose:
            print("Saved.")

    return master
