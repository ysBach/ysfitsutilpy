from pathlib import Path
from warnings import warn

import ccdproc
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
from astropy.time import Time
from ccdproc import combine

from .filemgmt import load_if_exists, make_summary
from .hduutil import (CCDData_astype, _parse_extension, chk_keyval, cmt2hdr,
                      imslice, inputs2list, load_ccd)
from .misc import listify

__all__ = [
    "sstd", "weighted_mean", "group_fits", "select_fits", "stack_FITS",
    "combine_ccd"
]


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


def group_fits(
        summary_table,
        type_key=None,
        type_val=None,
        group_key=None,
        table_filecol="file",
        verbose=False
):
    ''' Organize the group_by and type_key for stack_FITS

    Parameters
    ----------
    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata (header) of files. If it is in
        the astropy table format, it will be converted to `~pandas.DataFrame`
        object.

    type_key, type_val: None, str, list of str, optional
        The header keyword for the ccd type, and the value you want to match.

    group_key : None, str, list of str, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.

    Returns
    -------
    grouped : ~pandas.DataFrameGroupBy
        The table after the grouping process.

    group_type_key : list of str
        The `type_key` that can directly be used for `stack_FITS` for each
        element of `grouped.groups`. Basically this is ``type_key +
        group_key``.

    Examples
    --------
    >>> allfits = list(Path('.').glob("*.fits"))
    >>> summary_table = make_summary(allfits)
    >>> type_key = ["OBJECT"]
    >>> type_val = ["dark"]
    >>> group_key = ["EXPTIME"]
    >>> gs, g_key = group_fits(summary_table,
    ...                        type_key,
    ...                        type_val,
    ...                        group_key)
    >>> for g_val, group in gs:
    >>>     _ = combine_ccd(group["file"],
    ...                     type_key=g_key,
    ...                     type_val=g_val)
    '''
    if isinstance(summary_table, Table):
        st = summary_table.copy().to_pandas()
    elif isinstance(summary_table, pd.DataFrame):
        st = summary_table.copy()
    else:
        raise TypeError(
            "summary_table must be an astropy Table or Pandas DataFrame. "
            + f"It's now {type(summary_table)}."
        )

    type_key, type_val, group_key = chk_keyval(type_key=type_key, type_val=type_val,
                                               group_key=group_key)

    if len(group_key + type_key) == 0:
        raise ValueError("At least one of type_key and group_key should not be empty!")

    # For simplicity, crop the original data by type_key and type_val first.
    if type_key and type_val:  # if not empty list
        fpaths = select_fits(
            st,
            table_filecol=table_filecol,
            prefer_ccddata=False,
            type_key=type_key,
            type_val=type_val,
            verbose=verbose,
            path_to_text=True
        )
        st = st[st[table_filecol].isin(fpaths)]
    group_type_key = type_key + group_key
    grouped = st.groupby(group_key)

    return grouped, group_type_key


def select_fits(
        inputs,
        extension=None,
        unit=None,
        trimsec=None,
        table_filecol="file",
        prefer_ccddata=False,
        type_key=None,
        type_val=None,
        path_to_text=False,
        verbose=True
):
    ''' Stacks the FITS files specified in fitslist

    Parameters
    ----------
    inputs : path-like, CCDData, fits.PrimaryHDU, fits.ImageHDU,pandas.DataFrame or astropy.table.Table
        If it is path-like, it must contain FITS files to extract header. If
        CCD-like, the header information will be used for selecting elements to
        select.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Ignored if `inputs` is table-like.

    unit: Unit or str, optional
        The unit of the CCDs to be loaded.
        Used only when `fitslist` is not a list of `~astropy.nddata.CCDData`
        and `prefer_ccddata` is `True`.
        Ignored if `inputs` is table-like.

    trimsec : str, [list of] int, [list of] slice, optional
        Section of the data to be extracted by `~ysfitsutilpy.hduutil.imslice`.
        Default is `None`.
        Ignored if `inputs` is table-like.

    table_filecol: str
        The column name of the `summary_table` which contains the path to the
        FITS files. Ignored if `inputs` is CCD-like.

    prefer_ccddata: bool, optional
        Whether to prefer to return CCDData objects if possible. If `True`,
        path-like, ndarray, or table-like input will return a list of CCDData.
        If `False` (default), only the paths will be returned unless the
        `inputs` is consist of CCDData. Ignored if `inputs` is already
        CCD-like.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.

    Returns
    -------
    matched: list of Path or list of CCDData
        list containing Path to files if `prefer_ccddata` is `False`. Otherwise
        it is a list containing loaded CCDData after loading the files. If
        `ccdlist` is given a priori, list of CCDData will be returned
        regardless of `prefer_ccddata`.
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

    def _check_mismatch(row, keys, values):
        mismatch = False
        for k, v in zip(keys, values):
            hdr_val = _parse_val(row[k])
            parse_v = _parse_val(v)
            if (hdr_val != parse_v):
                mismatch = True
                break
        return mismatch

    # Check for type_key and type_val
    type_key, type_val, _ = chk_keyval(type_key=type_key, type_val=type_val, group_key=None)
    # I made this but think it is unnecessary as all string type_val must be subject to
    # regex.. I am leaving it here just in case in the future I find it necessary.
    #   YPBach 2021-01-08 17:36:53 (KST: GMT+09:00)
    # regex : bool or list of bool optional.
    #     Whether to use regex for `type_val` matching. Default is `False`. If it
    #     is a list, it must have the identical length to `type_key` and
    #     `type_val`. An example is that you want to select ``OBJECT`` with regex
    #     of ``'NGC.*'``, but ``EXPTIME`` of ``120``, which is a numeric. Sometimes
    #     the header will have ``'120.0'``, which may not be easily selected by
    #     regex. In that case, an internal parser is easier to use to catch any
    #     numeric values that must be regarded as the same thing. A possible usage
    #     is: ``type_key=["OBJECT", "EXPTIME"], type_val=["NGC*", 120],
    #     regex=[True, False]``.
    # if isinstance(regex, bool):
    #     regex = [regex]*len(type_key)
    # else:
    #     try:
    #         if len(regex) != len(type_key):
    #             raise ValueError("Length of regex differ from type_key and type_val.")
    #         if not all(isinstance(r, bool) for r in regex):
    #             raise TypeError("If regex is not bool, it must be list of bool.")
    #     except TypeError:
    #         raise TypeError("If regex is not bool, it must be list of bool.")

    # Setting whether we have to select a subset from the list
    selecting = True if len(type_key) > 0 else False

    if verbose:
        print("Analyzing FITS... ", end='')

    if isinstance(inputs, Table):
        summary_table = inputs.to_pandas()
        fitslist = summary_table[table_filecol].to_list()
    elif isinstance(inputs, pd.DataFrame):
        summary_table = inputs
        fitslist = summary_table[table_filecol].to_list()
    else:
        # No need to sort here because the real "sort" will be done later in make_summary
        fitslist = inputs2list(inputs, sort=False, accept_ccdlike=True,
                               check_coherency=False, path_to_text=path_to_text)
        if selecting:
            summary_table = make_summary(
                fitslist,
                extension=extension,
                # extension will be parsed within make_summary (no need to care here)
                verbose=verbose,
                fname_option='relative',
                keywords=type_key,
                sort_by=None,
            )
        else:
            summary_table = None

    if summary_table is not None:
        try:
            summary_table.reset_index(inplace=True, drop=True)
        except ValueError:
            pass

    if verbose:
        print("Done.")

    # ************************************************************************************ #
    # *                             SELECT AND LOAD TO MATCHED                           * #
    # ************************************************************************************ #
    # == Do regex matching if type_val[i] is string ====================================== #
    _type_key = []
    _type_val = []
    if selecting:
        for k, v in zip(type_key, type_val):
            if isinstance(v, str):
                match_mask = summary_table[k].str.match(v)
                summary_table = summary_table[match_mask]
                fitslist = np.array(fitslist)[match_mask].tolist()
                # NOTE: Is there a better way to do this?
                try:
                    summary_table.reset_index(inplace=True, drop=True)
                except ValueError:
                    pass
            else:  # not used as regex
                _type_key.append(k)
                _type_val.append(v)
                continue  # need to do _check_mismatch below

    matched = []
    if selecting:
        # == Select FITS based on type_key and type_val ================================== #
        for i, row in summary_table.iterrows():
            # I intentionally used iterrows instead of making mask, because for
            # some cases the keyword (e.g., an angle) can contain both str and
            # float among CCDs.
            #   For example, if we want to select ``angle == 0.0``, masking
            # cannot work because the column has dtype of object
            # (``summary_table[column].dtype`` is `object``).
            #   Instead, _check_mismatch tries to convert the value found in
            # the header to int, and if it fails, tries float, and finally uses
            # str. This is the most natural way I could think of.
            # ysBach, 2020-05-15 09:44:13 (KST: GMT+09:00)
            mismatch = _check_mismatch(row, _type_key, _type_val)
            if mismatch:  # skip this row (file)
                continue

            # if not skipped:
            item = fitslist[i]
            if isinstance(item, CCDData):
                if trimsec is None:
                    matched.append(item)
                else:
                    matched.append(imslice(item, trimsec=trimsec))
            else:  # it must be a path to a file
                fpath = Path(item)
                if prefer_ccddata:
                    # extension will be parsed within load_ccd (no need to care here)
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trimsec is not None:
                        ccd_i = imslice(ccd_i, trimsec=trimsec)
                    matched.append(ccd_i)
                else:
                    if path_to_text:
                        matched.append(str(fpath))
                    else:
                        matched.append(fpath)
    else:
        # == Use all item in fitslist ==================================================== #
        # summary_table is not used.
        for item in fitslist:
            if isinstance(item, CCDData):
                if trimsec is None:
                    matched.append(item)
                else:
                    matched.append(imslice(item, trimsec=trimsec))
            else:  # it must be a path to a file
                if prefer_ccddata:
                    # extension will be parsed within load_ccd (no need to care here)
                    ccd_i = load_ccd(item, extension=extension, unit=unit)
                    if trimsec is not None:
                        ccd_i = imslice(ccd_i, trimsec=trimsec)
                    matched.append(ccd_i)
                else:  # TODO: Is is better to remove Path here?
                    if path_to_text:
                        matched.append(str(item))
                    else:
                        matched.append(Path(item))

    # ************************************************************************************ #
    # *                           PRINT INFO MESSAGE OR WARNING                          * #
    # ************************************************************************************ #
    if len(matched) == 0:
        if selecting:
            warn(f'No FITS file had "{str(type_key)} = {str(type_val)}". '
                 + 'Maybe int/float/str confusing?')
        else:
            warn('No FITS file found')
    else:
        if selecting:
            N = len(matched)
            ks = str(type_key)
            vs = str(type_val)
            if verbose:
                if prefer_ccddata:
                    print(f'{N} FITS files with "{ks} = {vs}" are loaded.')
                else:
                    print(f'{N} FITS files with "{ks} = {vs}" are selected.')
        else:
            if verbose and prefer_ccddata:
                print('{:d} FITS files are loaded.'.format(len(matched)))

    return matched


# !FIXME: Remove in the future
def stack_FITS(
        fitslist=None,
        summary_table=None,
        extension=None,
        unit=None,
        table_filecol="file",
        trimsec=None,
        ccddata=True,
        asccd=True,
        type_key=None,
        type_val=None,
        verbose=True
):
    ''' Stacks the FITS files specified in fitslist

    Parameters
    ----------
    fitslist: None, [list of] path-like, or [list of] CCDData
        The list of path to FITS files or the list of CCDData to be stacked. It
        is useful to give list of CCDData if you have already stacked/loaded
        FITS file into a list by your own criteria. If `None` (default), you
        must give `fitslist` or `summary_table`. If it is not `None`, this
        function will do very similar job to that of `ccdproc.combine`.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    summary_table: None, pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many FITS
        files and you want to use stacking many times, it is better to make a
        summary table by `filemgmt.make_summary` and use that instead of
        opening FITS files' headers every time you call this function. If you
        want to use `summary_table` instead of `fitslist` and have set
        ``ccddata=True``, you must not have `None` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    unit: Unit or str, optional
        The unit of the CCDs to be loaded.
        Used only when `fitslist` is not a list of `~astropy.nddata.CCDData`
        and `ccddata` is `True`.

    table_filecol: str
        The column name of the `summary_table` which contains the path to the FITS files.

    trimsec : str, [list of] int, [list of] slice, optional
        Section of the data to be extracted by `~ysfitsutilpy.hduutil.imslice`.
        Default is `None`.

    ccddata: bool, optional
        Whether to return file paths or loaded CCDData. If `False`, it is a
        function to select FITS files using `type_key` and `type_val` without
        using much memory.
        This is ignored if `fitslist` is given and composed of
        `~astropy.nddata.CCDData` objects.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.

    asccd : bool, optional.
        Whether to load as `~astropy.nddata.CCDData`. If `False`, numpy ndarray
        will be used. Works only if ``ccddata = True``.

    Returns
    -------
    matched: list of Path or list of CCDData
        list containing Path to files if `ccddata` is `False`. Otherwise it is
        a list containing loaded CCDData after loading the files. If `ccdlist`
        is given a priori, list of CCDData will be returned regardless of
        `ccddata`.
    '''
    warn("stack_FITS is deprecated; use select_fits.", DeprecationWarning)

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

    # ************************************************************************************ #
    # *                            MAKE FITSLIST AND SUMMARY_TABLE                       * #
    # ************************************************************************************ #
    # == If fitslist ===================================================================== #
    if fitslist is not None:
        fitslist = listify(fitslist)

        if selecting:
            summary_table = make_summary(
                fitslist,
                extension=extension,
                # extension will be parsed within make_summary (no need to care here)
                verbose=verbose,
                fname_option='relative',
                keywords=type_key,
                sort_by=None,
            )
        # else:
        #   no need to make summary_table.
    # == If summary_table ================================================================ #
    elif summary_table is not None:
        if isinstance(summary_table, Table):
            summary_table = summary_table.to_pandas()
        elif not isinstance(summary_table, pd.DataFrame):
            raise TypeError("summary_table must be an astropy Table or Pandas DataFrame. "
                            + f"It's now {type(summary_table)}.")
        else:
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

    # ************************************************************************************ #
    # *                             SELECT AND LOAD TO MATCHED                           * #
    # ************************************************************************************ #
    matched = []
    if selecting:
        # == Select FITS based on type_key and type_val ================================== #
        for i, row in summary_table.iterrows():
            # I intentionally used iterrows instead of making mask, because for some cases
            # the keyword (e.g., an angle) can contain both str and float among CCDs.
            #    For example, if we want to select ``angle == 0.0``, masking cannot work
            # because the column has dtype of object (``summary_table[column].dtype`` is
            # ``object```).
            #    Instead, _check_mismatch tries to convert the value found in the header to
            # int, and if it fails, tries float, and finally uses str. This is the most
            # natural way I could think of.
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
                    if trimsec is not None:
                        ccd_i = imslice(ccd_i, trimsec=trimsec)
                    if asccd:
                        matched.append(ccd_i)
                    else:
                        matched.append(ccd_i.data)
                else:
                    matched.append(fpath)
    else:
        # == Use all item in fitslist ==================================================== #
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
                    if trimsec is not None:
                        ccd_i = imslice(ccd_i, trimsec=trimsec)
                    if asccd:
                        matched.append(ccd_i)
                    else:
                        matched.append(ccd_i.data)
                else:  # TODO: Is is better to remove Path here?
                    matched.append(Path(item))

    # ************************************************************************************ #
    # *                           PRINT INFO MESSAGE OR WARNING                          * #
    # ************************************************************************************ #
    if len(matched) == 0:
        if selecting:
            warn(f'No FITS file had "{str(type_key)} = {str(type_val)}"'
                 + 'Maybe int/float/str confusing?')
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


# TODO: accept the input like ``sigma_clip_func='median'``, etc.
def combine_ccd(
        fitslist=None,
        summary_table=None,
        table_filecol="file",
        trimsec=None,
        output=None,
        unit=None,
        subtract_frame=None,
        combine_method='median',
        reject_method=None,
        normalize_exposure=False,
        normalize_average=False,
        normalize_median=False,
        exposure_key='EXPTIME',
        mem_limit=2e9,
        combine_uncertainty_function=None,
        extension=None,
        type_key=None,
        type_val=None,
        dtype="float32",
        uncertainty_dtype="float32",
        output_verify='fix',
        overwrite=False,
        verbose=True,
        **kwargs
):
    ''' Combining images -- slight variant from ccdproc.

    Parameters
    ----------
    fitslist: path-like, list of path-like, or list of CCDData
        The list of path to FITS files or the list of CCDData to be stacked. It
        is useful to give list of CCDData if you have already stacked/loaded
        FITS file into a list by your own criteria. If `None` (default), you
        must give `fitslist` or `summary_table`. If it is not `None`, this
        function will do very similar job to that of `ccdproc.combine`.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many FITS
        files and you want to use stacking many times, it is better to make a
        summary table by `filemgmt.make_summary` and use that instead of
        opening FITS files' headers every time you call this function. If you
        want to use `summary_table` instead of `fitslist` and have set
        ``ccddata=True``, you must not have `None` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    table_filecol: str
        The column name of the `summary_table` which contains the path to the
        FITS files.

    trimsec : str, [list of] int, [list of] slice, optional
        Section of the data to be extracted by `~ysfitsutilpy.hduutil.imslice`.
        Default is `None`.

    output : path-like or None, optional.
        The path if you want to save the resulting `~astropy.nddata.CCDData`
        object.
        Default is `None`.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is `None`.

    subtract_frame : array-like, optional.
        The frame you want to subtract from the image after the combination. It
        can be, e.g., dark frame, because it is easier to calculate Poisson
        error before the dark subtraction and subtract the dark later.
        TODO: This maybe unnecessary.
        Default is `None`.

    combine_method : str or None, optinal.
        The `method` for `ccdproc.combine`, i.e., {'average', 'median', 'sum'}
        Default is `None`.

    reject_method : str
        Made for simple use of `ccdproc.combine`, [None, 'minmax', 'sigclip' ==
        'sigma_clip', 'extrema' == 'ext']. Automatically turns on the option,
        e.g., ``clip_extrema = True`` or ``sigma_clip = True``. Leave it blank
        for no rejection.
        Default is `None`.

    normalize_exposure : bool, optional.
        Whether to normalize the values by the exposure time of each frame
        before combining.
        Default is `False`.

    normalize_average, normalize_median : bool, optional.
        Whether to normalize the values by the average or median value of each
        frame before combining. Only up to one of these must be `True`.
        Default is `False`.

    exposure_key : str, optional
        The header keyword for the exposure time.
        Default is ``"EXPTIME"``.

    combine_uncertainty_function : callable, None, optional
        The uncertainty calculation function of `~ccdproc.combine`. If `None`
        use the default uncertainty func when using average, median or sum
        combine, otherwise use the function provided.
        Default is `None`.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    dtype : str or `numpy.dtype` or None, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description. If `None` it uses ``np.float64``.
        Default is `None`.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.
        For an open HDU named `hdu`, e.g., only the files which satisfies
        ``hdu[extension].header[type_key] == type_val`` among all the
        `fitslist` will be used.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``. May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    mem_limit : float, optional
        Maximum memory which should be used while combining (in bytes).
        Default is ``2.e9``.

    **kwarg:
        kwargs for the `ccdproc.combine`. See its documentation. This includes
        (RHS are the default values)
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
                raise KeyError(
                    "reject must be one of "
                    + "[None, 'minmax', sigclip'=='sigma_clip', 'extrema'=='ext']"
                )

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
        # == a single CCDData ============================================================ #
        if isinstance(fitslist, CCDData):
            fitslist = [fitslist]
        else:
            # == a single path-like ====================================================== #
            try:
                fitslist = [Path(fitslist)]
            except TypeError:
                # == a list of path-like or CCDData ====================================== #
                try:
                    fitslist = list(fitslist)
                except TypeError:
                    raise TypeError(
                        f"fitslist must be list-like. It's now {type(fitslist)}."
                    )

    # If summary_table
    if summary_table is not None:
        if ((not isinstance(summary_table, Table))
                and (not isinstance(summary_table, pd.DataFrame))):
            raise TypeError(
                "summary_table must be an astropy Table or Pandas DataFrame. "
                + f"It's now {type(summary_table)}."
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
            "Only up to one of [normalize_average, normalize_exposure, normalize_median] "
            + "is acceptable."
        )

    # Set history messages
    str_history = ('{:d} images with {:s} = {:s} are "{:s}" combined '
                   + 'using "{:s}" rejection (additional kwargs: {})')
    str_nexp = "Each frame will be normalized by exposure time before combine."
    str_navg = "Each frame will be normalized by average before combine."
    str_nmed = "Each frame will be normalized by median before combine."
    str_subt = "Subtracted a user-provided frame"

    if reject_method is None:
        reject_method = 'no'

    extension = _parse_extension(extension)

    # Select CCDs by
    ccdlist = stack_FITS(
        fitslist=fitslist,
        summary_table=summary_table,
        table_filecol=table_filecol,
        extension=extension,
        # extension will be parsed within make_summary/load_ccd (no need to care here)
        unit=unit,
        type_key=type_key,
        type_val=type_val,
        ccddata=False,
        verbose=verbose
    )
    #  trimsec=trimsec,
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
        cmt2hdr(header, 'h', str_nexp, verbose=verbose)

    # Normalize by pixel average
    if normalize_average:
        def invavg(a):
            return 1 / np.mean(a)
        scale = invavg
        cmt2hdr(header, 'h', str_navg, verbose=verbose)

    # Normalize by pixel median
    if normalize_median:
        def invmed(a):
            return 1 / np.median(a)
        scale = invmed
        cmt2hdr(header, 'h', str_nmed, verbose=verbose)

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
            **kwargs
        )

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
    cmt2hdr(header, 'h', s, verbose=verbose, t_ref=_t)
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
        cmt2hdr(header, 'h', str_subt, header, verbose=verbose, t_ref=_t)

    if trimsec is not None:
        master = imslice(master, trimsec, verbose=verbose)

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
