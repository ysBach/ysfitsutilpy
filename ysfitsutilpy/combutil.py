from warnings import warn
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
from astropy.io import fits
import ccdproc
from ccdproc import combine, trim_image

from .filemgmt import load_if_exists, make_summary
from .ccdutil import CCDData_astype, load_ccd
from .misc import chk_keyval

__all__ = ["sstd", "weighted_mean", "stack_FITS", "combine_ccd"]


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
        The table which contains the metadata (header) of files. If it is in
        the astropy table format, it will be converted to `~pandas.DataFrame`
        object.

    type_key, type_val: None, str, list of str, optional
        The header keyword for the ccd type, and the value you want to match.

    group_key : None, str, list of str, optional
        The header keyword which will be used to make groups for the CCDs
        that have selected from ``type_key`` and ``type_val``.
        If ``None`` (default), no grouping will occur, but it will return
        the `~pandas.DataFrameGroupBy` object will be returned for the sake
        of consistency.

    Return
    ------
    grouped : ~pandas.DataFrameGroupBy
        The table after the grouping process.

    group_type_key : list of str
        The ``type_key`` that can directly be used for ``stack_FITS`` for each
        element of ``grouped.groups``.
        Basically this is ``type_key + group_key``.

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
    if ((not isinstance(summary_table, Table))
            and (not isinstance(summary_table, pd.DataFrame))):
        raise TypeError("summary_table must be an astropy Table or Pandas "
                        + f"DataFrame. It's now {type(summary_table)}.")
    elif isinstance(summary_table, Table):
        st = summary_table.to_pandas()
    else:
        st = summary_table.copy()

    type_key, type_val, group_key = chk_keyval(type_key=type_key,
                                               type_val=type_val,
                                               group_key=group_key)

    # For simplicity, crop the original data by type_key and type_val first.
    for k, v in zip(type_key, type_val):
        st = st[st[k] == v]

    group_type_key = type_key + group_key
    group_type_vals = []
    grouped = st.groupby(group_key)

    return grouped, group_type_key


def stack_FITS(fitslist=None, summary_table=None, extension=0,
               unit='adu', table_filecol="file", trim_fits_section=None,
               loadccd=True, type_key=None, type_val=None):
    ''' Stacks the FITS files specified in fitslist
    Parameters
    ----------
    fitslist: None, list of path-like, or list of CCDData
        The list of path to FITS files or the list of CCDData to be stacked.
        It is useful to give list of CCDData if you have already stacked/loaded
        FITS file into a list by your own criteria. If ``None`` (default),
        you must give ``fitslist`` or ``summary_table``. If it is not ``None``,
        this function will do very similar job to that of ``ccdproc.combine``.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    summary_table: None, pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many
        FITS files and you want to use stacking many times, it is better to
        make a summary table by ``filemgmt.make_summary`` and use that instead
        of opening FITS files' headers every time you call this function. If
        you want to use ``summary_table`` instead of ``fitslist`` and have set
        ``loadccd=True``, you must not have ``None`` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    extension: int or str
        The extension of FITS to be stacked. For single extension, set it as 0.

    unit: Unit or str, optional
        The unit of the CCDs to be loaded.
        Used only when ``fitslist`` is not a list of ``CCDData`` and
        ``loadccd`` is ``True``.

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to
        the FITS files.

    trim_fits_section : str or None, optional
        The ``fits_section`` of ``ccdproc.trim_image``.
        Region of ``ccd`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details.
        Default is ``None``.

    loadccd: bool, optional
        Whether to return file paths or loaded CCDData. If ``False``, it is
        a function to select FITS files using ``type_key`` and ``type_val``
        without using much memory.
        This is ignored if ``fitslist`` is given and composed of ``CCDData``
        objects.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.

    Return
    ------
    matched: list of Path or list of CCDData
        list containing Path to files if ``loadccd`` is ``False``. Otherwise
        it is a list containing loaded CCDData after loading the files. If
        ``ccdlist`` is given a priori, list of CCDData will be returned
        regardless of ``loadccd``.
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
        raise ValueError("One and only one of fitslist or summary_table must "
                         + "be not None.")

    # If fitslist
    if fitslist is not None:
        table_mode = False
        if not isinstance(fitslist, list):
            raise TypeError(
                f"fitslist must be a list. It's now {type(fitslist)}.")

    # If summary_table
    if summary_table is not None:
        table_mode = True
        if ((not isinstance(summary_table, Table))
                and (not isinstance(summary_table, pd.DataFrame))):
            raise TypeError("summary_table must be an astropy Table or Pandas "
                            + f"DataFrame. It's now {type(summary_table)}.")

    # Check for type_key and type_val
    type_key, type_val, _ = chk_keyval(type_key=type_key,
                                       type_val=type_val,
                                       group_key=None)

    # Setting whether to group
    grouping = False
    if len(type_key) > 0:
        grouping = True

    print("Analyzing FITS... ", end='')
    # Set fitslist and summary_table based on the given input and grouping.
    if table_mode:
        if isinstance(summary_table, Table):
            summary_table = summary_table.to_pandas()
        fitslist = summary_table[table_filecol].tolist()
    else:
        if grouping:
            summary_table = make_summary(fitslist,
                                         extension=extension,
                                         verbose=True,
                                         fname_option='relative',
                                         keywords=type_key,
                                         sort_by=None,
                                         pandas=True)
        # else: no need to make summary_table.

    print("Done", end='')

    if loadccd:
        print(" and loading FITS... ")
    else:
        print(".")

    matched = []

    # Append appropriate CCDs or filepaths to matched
    if grouping:  # summary_table is used.
        for i, row in summary_table.iterrows():
            mismatch = _check_mismatch(row)
            if mismatch:  # skip this row (file)
                continue

            # if not skipped:
            # TODO: Is it better to remove Path here?
            if isinstance(fitslist[i], CCDData):
                matched.append(fitslist[i])
            else:  # it must be a path to the file
                fpath = Path(fitslist[i])
                if loadccd:
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(ccd_i,
                                           fits_section=trim_fits_section)
                    matched.append(ccd_i)
                else:
                    matched.append(fpath)
    else:  # summary_table is not used.
        for item in fitslist:
            if isinstance(item, CCDData):
                matched.append(item)
            else:
                if loadccd:
                    ccd_i = load_ccd(item, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(
                            ccd_i, fits_section=trim_fits_section)
                    matched.append(ccd_i)
                else:  # TODO: Is is better to remove Path here?
                    matched.append(Path(item))

    # Generate warning OR information messages
    if len(matched) == 0:
        if grouping:
            warn('No FITS file had "{:s} = {:s}"'.format(str(type_key),
                                                         str(type_val))
                 + "Maybe int/float/str confusing?")
        else:
            warn('No FITS file found')
    else:
        if grouping:
            N = len(matched)
            ks = str(type_key)
            vs = str(type_val)
            if loadccd:
                print(f'{N} FITS files with "{ks} = {vs}" are loaded.')
            else:
                print(f'{N} FITS files with "{ks} = {vs}" are selected.')
        else:
            if loadccd:
                print('{:d} FITS files are loaded.'.format(len(matched)))

    return matched


def combine_ccd(fitslist=None, summary_table=None, table_filecol="file",
                trim_fits_section=None, output=None, unit='adu',
                subtract_frame=None, combine_method='median',
                reject_method=None, normalize_exposure=False,
                normalize_average=False,
                exposure_key='EXPTIME', mem_limit=2e9,
                combine_uncertainty_function=None,
                extension=0, type_key=None, type_val=None,
                dtype="float32", uncertainty_dtype="float32",
                output_verify='fix', overwrite=False,
                verbose=True, **kwargs):
    ''' Combining images
    Slight variant from ccdproc.
    # TODO: accept the input like ``sigma_clip_func='median'``, etc.
    Parameters
    ----------
    fitslist: path-like, list of path-like, or list of CCDData
        The list of path to FITS files or the list of CCDData to be stacked.
        It is useful to give list of CCDData if you have already stacked/loaded
        FITS file into a list by your own criteria. If ``None`` (default),
        you must give ``fitslist`` or ``summary_table``. If it is not ``None``,
        this function will do very similar job to that of ``ccdproc.combine``.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many
        FITS files and you want to use stacking many times, it is better to
        make a summary table by ``filemgmt.make_summary`` and use that instead
        of opening FITS files' headers every time you call this function. If
        you want to use ``summary_table`` instead of ``fitslist`` and have set
        ``loadccd=True``, you must not have ``None`` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to
        the FITS files.

    trim_fits_section : str or None, optional
        The ``fits_section`` of ``ccdproc.trim_image``.
        Region of ``ccd`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details.
        Default is ``None``.

    output : path-like or None, optional.
        The path if you want to save the resulting ``ccd`` object.
        Default is ``None``.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is ``'adu'``.

    subtract_frame : array-like, optional.
        The frame you want to subtract from the image after the combination.
        It can be, e.g., dark frame, because it is easier to calculate Poisson
        error before the dark subtraction and subtract the dark later.
        TODO: This maybe unnecessary.
        Default is ``None``.

    combine_method : str or None, optinal.
        The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median', 'sum'}
        Default is ``None``.

    reject_method : str
        Made for simple use of ``ccdproc.combine``,
        [None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema' == 'ext'].
        Automatically turns on the option, e.g., ``clip_extrema = True`` or
        ``sigma_clip = True``.
        Leave it blank for no rejection.
        Default is ``None``.

    normalize_exposure : bool, optional.
        Whether to normalize the values by the exposure time of each frame
        before combining.
        Default is ``False``.

    normalize_average : bool, optional.
        Whether to normalize the values by the average value of each frame
        before combining.
        Default is ``False``.

    exposure_key : str, optional
        The header keyword for the exposure time.
        Default is ``"EXPTIME"``.

    combine_uncertainty_function : callable, None, optional
        The uncertainty calculation function of ``ccdproc.combine``.
        If ``None`` use the default uncertainty func when using average,
        median or sum combine, otherwise use the function provided.
        Default is ``None``.

    extension: int or str, optional
        The extension to be used.
        Default is ``0``.

    dtype : str or `numpy.dtype` or None, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description. If ``None`` it uses ``np.float64``.
        Default is ``None``.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.
        For an open HDU named ``hdu``, e.g., only the files which satisfies
        ``hdu[extension].header[type_key] == type_val`` among all the ``fitslist``
        will be used.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``.  May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    mem_limit : float, optional
        Maximum memory which should be used while combining (in bytes).
        Default is ``2.e9``.

    **kwarg:
        kwargs for the ``ccdproc.combine``. See its documentation.
        This includes (RHS are the default values)
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
    def _set_reject_method(reject_method):
        ''' Convenience function for ccdproc.combine reject switches
        '''
        clip_extrema, minmax_clip, sigma_clip = False, False, False

        if reject_method in ['extrema', 'ext']:
            clip_extrema = True
        elif reject_method in ['minmax']:
            minmax_clip = True
        elif reject_method in ['sigma_clip' 'sigclip']:
            sigma_clip = True
        else:
            if reject_method is not None:
                raise KeyError("reject must be one of "
                               "{None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema' == 'ext}")

        return clip_extrema, minmax_clip, sigma_clip

    def _print_info(combine_method, Nccd, reject_method, **kwargs):
        if reject_method is None:
            reject_method = 'no'

        info_str = ('"{:s}" combine {:d} images by "{:s}" rejection\n')

        print(info_str.format(combine_method, Nccd, reject_method))
        print(dict(**kwargs))
        return

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

    def _add_and_print(s, header, verbose):
        header.add_history(s)
        if verbose:
            print(s)
    # Give only one
    if ((fitslist is not None) + (summary_table is not None) != 1):
        raise ValueError(
            "One and only one of [fitslist, summary_table] must be given.")

    # If fitslist
    if fitslist is not None:
        if not isinstance(fitslist, list):
            raise TypeError(
                f"fitslist must be a list. It's now {type(fitslist)}.")

    # If summary_table
    if summary_table is not None:
        if ((not isinstance(summary_table, Table))
                and (not isinstance(summary_table, pd.DataFrame))):
            raise TypeError(
                f"summary_table must be an astropy Table or Pandas DataFrame. It's now {type(summary_table)}.")

    # Check for type_key and type_val
    if ((type_key is None) ^ (type_val is None)):
        raise ValueError(
            "type_key and type_val must be both specified or both None.")

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            print(f"{output} already exists:\n\tBut will be overridden.")
        else:
            print(f"{output} already exists:")
            return load_if_exists(output, loader=CCDData.read, if_not=None)

    # Do we really need to accept all three of normalize & scale?
    # if scale is None:
    #     scale = np.ones(len(ccdlist))
    if (((normalize_average) + (normalize_exposure)) > 1):
        raise ValueError("Only up to one of [normalize_average, "
                         + "normalize_exposure] must be not None.")

    # Set history messages
    str_history = ('{:d} images with {:s} = {:s} are "{:s}" combined '
                   + 'using "{:s}" rejection with {}')
    str_nexp = "Each frame normalized by exposure time before combination."
    str_navg = "Each frame normalized by average value before combination."
    str_subt = "Subtracted a user-provided frame"
    str_trim = "Trim by FITS section {}"

    if reject_method is None:
        reject_method = 'no'

    # Select CCDs by
    ccdlist = stack_FITS(fitslist=fitslist,
                         summary_table=summary_table,
                         table_filecol=table_filecol,
                         extension=extension,
                         unit=unit,
                         type_key=type_key,
                         type_val=type_val,
                         loadccd=False)
    #  trim_fits_section=trim_fits_section,
    # loadccd=False: Loading CCD here may cause memory blast...

    try:
        header = ccdlist[0].header
    except AttributeError:
        header = fits.getheader(ccdlist[0])

    if verbose:
        _print_info(combine_method=combine_method,
                    Nccd=len(ccdlist),
                    reject_method=reject_method,
                    dtype=dtype,
                    **kwargs)

    scale = None
    # Normalize by exposure
    if normalize_exposure:
        tmp = make_summary(fitslist=fitslist,
                           keywords=[exposure_key],
                           verbose=False,
                           sort_by=None)
        exptimes = tmp[exposure_key].tolist()
        scale = 1 / np.array(exptimes)
        _add_and_print(str_nexp, header, verbose)

    # Normalize by pixel average
    if normalize_average:
        def invavg(a):
            return 1 / np.mean(a)
        scale = invavg
        _add_and_print(str_navg, header, verbose)

    # Set rejection switches
    clip_extrema, minmax_clip, sigma_clip = _set_reject_method(reject_method)

    if len(ccdlist) == 1:
        if isinstance(ccdlist[0], CCDData):
            master = ccdlist[0]
        else:
            master = load_ccd(ccdlist[0], extension=extension, unit=unit)
    else:
        master = combine(img_list=ccdlist,
                         method=combine_method,
                         clip_extrema=clip_extrema,
                         minmax_clip=minmax_clip,
                         sigma_clip=sigma_clip,
                         mem_limit=mem_limit,
                         combine_uncertainty_function=combine_uncertainty_function,
                         unit=unit,
                         hdu=extension,
                         scale=scale,
                         **kwargs)

    ncombine = len(ccdlist)
    header["COMBVER"] = (ccdproc.__version__,
                         "ccdproc version used for combine.")
    header["NCOMBINE"] = (ncombine, "Number of combined images")
    header["COMBMETH"] = (combine_method, "Combining method")

    header.add_history(str_history.format(ncombine,
                                          str(type_key),
                                          str(type_val),
                                          str(combine_method),
                                          str(reject_method),
                                          kwargs))

    if subtract_frame is not None:
        subtract = CCDData(subtract_frame.copy())
        master.data = master.subtract(subtract).data
        _add_and_print(str_subt, header, verbose)

    if trim_fits_section is not None:
        master = trim_image(master, fits_section=trim_fits_section)
        _add_and_print(str_trim.format(trim_fits_section), header, verbose)

    master.header = header
    master = CCDData_astype(master, dtype=dtype,
                            uncertainty_dtype=uncertainty_dtype)

    if output is not None:
        if verbose:
            print(f"Writing FITS to {output}... ", end='')
        master.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose:
            print("Saved.")

    return master
