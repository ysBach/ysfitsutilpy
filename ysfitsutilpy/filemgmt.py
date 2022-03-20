'''
Contians convenience funcitons which are
(1) more related to the file name or paths rather than the contents or
(2) related to the non-FITS files.
'''

from pathlib import Path
from warnings import warn

import ccdproc
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.nddata import CCDData
from astropy.time import Time

from .hduutil import (_parse_extension, cut_ccd, inputs2list, key_mapper,
                      key_remover)
from .misc import listify

__all__ = [
    "mkdir", "load_if_exists", "make_summary", "df_selector",
    "make_reduc_planner",
    # "planner_add_crrej",
    "fits_newpath", "fitsrenamer"
]


def mkdir(fpath, mode=0o777, exist_ok=True):
    ''' Convenience function for Path.mkdir()
    '''
    fpath = Path(fpath)
    Path.mkdir(fpath, mode=mode, exist_ok=exist_ok)


def load_if_exists(path, loader, if_not=None, verbose=True, **kwargs):
    ''' Load a file if it exists.

    Parameters
    ----------
    path : pathlib.Path of Path-like str
        The path to be searched.

    loader : a function
        The loader to load `path`. Can be ``CCDData.read``, ``np.loadtxt``,
        etc.

    if_not : str
        Give a python code as a str to be run if the loading failed.

    Returns
    -------
    loaded:
        The loaded file. If the file does not exist, `None` is returned.

    Example
    -------
    >>> from astropy.nddata import CCDData
    >>> from pathlib import Path
    >>> ccd = load_if_exists(
    >>>     Path(".", "test.fits"),
    >>>     loader=CCDData.read,
    >>>     unit='adu',
    >>>     if_not="print('File not found')"
    >>> )
    '''
    path = Path(path)

    if path.exists():
        if verbose:
            print(f'Loading the existing {str(path)}...', end='')
        loaded = loader(path, **kwargs)
        if verbose:
            print(" Done")
    elif if_not is not None:
        loaded = eval(if_not)
    else:
        loaded = None

    return loaded


def make_summary(
        inputs=None,
        extension=None,
        verify_fix=False,
        fname_option='relative',
        output=None,
        keywords=None,
        example_header=None,
        sort_by='file',
        sort_map=None,
        fullmatch=None,
        flags=0,
        querystr=None,
        negate_fullmatch=False,
        verbose=True
):
    """ Extracts summary from the headers of FITS files.

    Parameters
    ----------
    inputs : glob pattern, list-like of path-like, list-like of CCDData, `~pandas.DataFrame` convertible
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or list of
        files (each element must be path-like or CCDData). Although it is not a
        good idea, a mixed list of CCDData and paths to the files is also
        acceptable. If a `~pandas.DataFrame` or convertible (especially
        `~astropy.table.Table`) is given, it finds the ``"file"`` column and
        use it as the input files, make a summary table from the headers of
        those files.
        If `inputs` is `None`, any `output` is ignored and `None` is returned.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    verify_fix : bool, optional.
        Whether to do ``.verify('fix')`` to all FITS files to avoid
        VerifyError. It may take some time if turned on. Default is `False`.

    fname_option : str {'absolute', 'relative', 'name'}, optional
        Whether to save full absolute/relative path or only the filename.

    output : str or path-like, optional
        The directory and file name of the output summary file.

    keywords : list or str(``"*"``), optional
        The list of the keywords to extract (keywords should be in str).

    example_header : None or path-like, optional
        The path including the filename of the output summary text file. If
        specified, the header of the 0-th element of `inputs` will be extracted
        (if glob-pattern is given, the 0-th element is random, so be careful)
        and saved to `example_header`. Use `None` (default) to skip this.

    sort_by : str, optional
        The column name to sort the results. It can be any element of
        `keywords` or `'file'`, which sorts the table by the file name.

    sort_map: dict, optional
        A subset of `key` parameter in `pandas.DataFrame.sort_values()`. If a
        dict is given, then ``key = lambda x: x.map(sort_map)`` is passed into
        `.sort_values()`.

    fullmatch : dict, optional
        The ``{column: regex}`` style dict to be used for selecting rows by
        ``summarytab[column].str.fullmatch(regex, case=True)``.
        Default: `None`

    negate_fullmatch: bool, optional.
        Whether to negate the mask by `fullmatch`, in case the user does not
        want to think much about regex to negate it.

    flags: int, optional.
        Regex module flags, e.g. re.IGNORECASE. Default: 0

    querystr : str, optional
        The query string used for ``summarytab.query(querystr)``. See
        `~pandas.DataFrame.query`.

    Returns
    -------
    summarytab: astropy.Table

    Notes
    -----
    I want to use ccdproc.ImageFileCollection instead of this, but it is about
    4 times slower than my make_summary, so I cannot use it yet.

    Example
    -------
    >>> from pathlib import Path
    >>> import ysfitsutilpy as yfu
    >>> keys = ["OBS-TIME", "FILTER", "OBJECT"]
    >>> # actually it is case-insensitive
    >>> # The keywords you want to extract
    >>> # (from the headers of FITS files)
    >>> TOPPATH = Path(".", "observation_2018-01-01")
    >>> # The toppath
    >>> savepath = TOPPATH / "summary_20180101.csv"
    >>> # list of all the fits files in TOPPATH/rawdata:
    >>> summary = yfu.make_summary(
    >>>     TOPPATH/"rawdata/*.fits",
    >>>     keywords=keys,
    >>>     fname_option='name',
    >>>     pandas=True,
    >>>     sort_by="DATE-OBS",
    >>>     output=savepath
    >>> )

    Select all rows with ``OBJECT`` starts with "DA":
    >>> # fullmatch = {"OBJECT": "DA.*"}
    Select all rows with ``OBJECT`` starts with "Ves", ``FILTER`` is "J", and
    ``EXPTIME`` is 2 or 3:
    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"},
    >>> # querystr="EXPTIME in [2, 3]
    """
    if inputs is None:
        return None

    # Although there's no need to sort here because the real "sort" will be
    # done later based on ``sort_by`` column, I did it here because the full
    # header keys will be inferred from the 0-th element (if `keywords` is not
    # given)
    fitslist = inputs2list(inputs, sort=True, accept_ccdlike=True, check_coherency=False)

    if len(fitslist) == 0:
        if verbose:
            print("No FITS file found.")
        return None

    def _get_fname_fsize_hdr(item, idx, extension):
        if isinstance(item, CCDData):
            # NOTE: CCDData does not support extension (only available when it
            #   is being read)!
            fname = f"CCDData in fitslist[{idx:d}]"
            fsize = None
            hdr = item.header
        else:
            if fname_option == 'relative':
                fname = str(item)
            elif fname_option == 'absolute':
                fname = str(item.absolute())
            elif fname_option == 'name':
                fname = item.name
            else:
                raise ValueError(f"fname_option `{fname_option}`not understood.")
            fsize = Path(item).stat().st_size
            # Don't change to MB/GB, which will make it float...
            hdul = fits.open(item)
            if verify_fix:
                hdul.verify('fix')
            hdr = hdul[extension].header
            hdul.close()

        return fname, fsize, hdr

    skip_keys = ['COMMENT', 'HISTORY']

    if verbose and keywords is not None:
        if keywords == '*':
            print("Extracting all keywords...")
        else:
            print("Extracting keys: ", keywords)

    extension = _parse_extension(extension)

    # Save example header
    if example_header is not None:
        fname0, _, hdr0 = _get_fname_fsize_hdr(fitslist[0], 0, extension=extension)
        if verbose:
            print(f"Header of 0-th: {fname0} -> {example_header}")
        hdr0.totextfile(example_header, overwrite=True)

    # load ALL keywords for special cases
    if (keywords is None) or (keywords is not None and keywords == '*'):
        fname0, _, hdr0 = _get_fname_fsize_hdr(fitslist[0], 0, extension=extension)
        N_hkeys = len(hdr0.cards)
        keywords = []

        for i in range(N_hkeys):
            try:
                key_i = hdr0.cards[i][0]
            except VerifyError:
                raise VerifyError("Use verify_fix=True.")
            if (key_i in skip_keys):
                continue
            elif (key_i in keywords):
                warn(f"Key {key_i} is duplicated! Only the first one will be saved.")
                continue
            keywords.append(key_i)

        if verbose:
            print(f"All {len(keywords)} keywords (guessed from {fname0}) will be loaded.")

    # Initialize
    summarytab = dict(file=[], filesize=[])
    for k in keywords:
        summarytab[k] = []

    # Run through all the fits files
    for i, item in enumerate(fitslist):
        fname, fsize, hdr = _get_fname_fsize_hdr(item, i, extension=extension)
        summarytab["file"].append(fname)
        summarytab["filesize"].append(fsize)
        for k in keywords:
            try:
                summarytab[k].append(hdr[k])
            except KeyError:
                if verbose:
                    str_keyerror_fill = "Key {:s} not found for {:s}, filling with None."
                    if isinstance(item, CCDData):
                        warn(str_keyerror_fill.format(k, f"fitslist[{i}]"))
                    else:
                        warn(str_keyerror_fill.format(k, str(item)))
                summarytab[k].append(None)

    summarytab = pd.DataFrame.from_dict(summarytab)
    summarytab = df_selector(summarytab, fullmatch=fullmatch, flags=flags,
                             querystr=querystr, negate_fullmatch=negate_fullmatch)
    if sort_by is not None:
        key = None if sort_map is None else lambda x: x.map(sort_map)
        summarytab.sort_values(sort_by, inplace=True, key=key)
    summarytab.reset_index(drop=True, inplace=True)

    if output is not None:
        output = Path(output)
        if verbose:
            print(f'Saving the summary to "{str(output)}"')
        summarytab.to_csv(output, index=False)

    return summarytab


def df_selector(
        summarytab,
        fullmatch=None,
        flags=0,
        negate_fullmatch=False,
        querystr=None,
        columns=None,
        columns_drop=None,
        reset_index=True,
):
    """Select rows from a summary table.

    Parameters
    ----------
    summarytab : `~pandas.DataFrame`
        The summary table to select from. Normally the table made from header
        information.
    fullmatch : dict, optional
        The ``{column: regex}`` style dict to be used for selecting rows by
        ``summarytab[column].str.fullmatch(regex, case=True)``. An example:
        ``{"OBJECT": "Ves.*"}``. All corresponding columns must have dtype of
        `str` to apply regex.
        Default: `None`
    negate_fullmatch: bool, optional.
        Whether to negate the mask by `fullmatch`, in case the user does not
        want to think much about regex to negate it.
    flags: int, optional.
        Regex module flags, e.g. re.IGNORECASE. Default: 0
    querystr : str, optional
        The query string used for ``summarytab.query(querystr)``. See
        `~pandas.DataFrame.query`.
    columns, columns_drop: str, list, optional.
        The list of columns to be returned/dropped after selection. No need to
        setup both, but no Error will be raised even the user does so.

    Returns
    -------
    summarytab
        The final summary table after selection. If everything is `None` (the
        default), the original summary table is returned.

    Raises
    ------
    AttributeError
        The column dtype is not `str`
    TypeError
        fullmatch must be in `dict`.

    Examples
    --------
    Select all rows with ``OBJECT`` starts with "DA":
    >>> # fullmatch = {"OBJECT": "DA.*"}
    Select all rows with ``OBJECT`` starts with "Ves", ``FILTER`` is "J", and
    ``EXPTIME`` is 2 or 3:
    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"},
    >>> # querystr="EXPTIME in [2, 3]"

    """
    df = summarytab.copy()

    if fullmatch is not None:
        if not isinstance(fullmatch, dict):
            raise TypeError("fullmatch must be a dict.")

        select_mask = np.ones(len(df), dtype=bool)
        for k, v in fullmatch.items():
            try:
                select_mask &= df[k].str.fullmatch(v, flags=flags, case=True)
            except AttributeError:
                try:
                    select_mask &= (df[k] == v)
                except (ValueError, TypeError, AttributeError):
                    raise TypeError(
                        "Both ``summarytab[k].str.fullmatch(v)`` and "
                        + f"``summarytab[{k}] == {v}`` failed.\n"
                        + "Maybe use `querystr` instead?"
                    )
        df = df[~select_mask] if negate_fullmatch else df[select_mask]

    if querystr is not None:
        df = df.query(querystr)

    if columns is not None:
        df = df[listify(columns)]

    if columns_drop is not None:
        df.drop(listify(columns_drop), axis=1, inplace=True)

    if reset_index:
        df = df.reset_index(drop=True)

    return df.copy()


# def df_matcher(
#     df1,
#     df2,
#     match_by=None
# ):
#     """

#     Parameters
#     ----------
#     df1 : `~pandas.DataFrame`
#         The first table for the rows to be picked out.
#     df2 : `~pandas.DataFrame`
#         The table for the rows to be matched based on `match_by`.
#     match_by : [type], optional
#         [description], by default None
#     """
#     if match_by is None:
#         if not all(df2.columns.isin(df1.columns)):
#             raise IndexError(
#                 f"Some column of `df2` not found in `df1`. "
#                 + "You may specify `match_by` to specify the columns to match."
#             )
#         match_by = list(df2.columns)
#     match_by = np.atleast_1d(match_by)

#     for idx, row in df1.iterrows():
#         try:
#             df2.loc[df2[match_by].eq(row[match_by]).all(axis=1), :]
#         for col, cal, mat, calcol in zip(cols, cals, mats, calcols):
#             mat = np.atleast_1d(mat)
#             # If just a str, cal[mat]==row[mat] gives `Series`, not `DataFrame`,
#             # hence `axis=1` raises error.
#             try:
#                 sel = cal[(cal[mat] == row[mat]).all(axis=1)]
#                 if len(sel) == 1:
#                     df.loc[idx, col] = sel[calcol].values[0]
#                 elif len(sel) > 1:
#                     raise ValueError(
#                         f"More than one calibration frame found for {mat} = {row[mat].values}"
#                     )
#                 else:
#                     continue
#             except (IndexError):  # no match
#                 continue


def make_reduc_planner(
        summary,
        cal_summary,
        newcolname,
        match_by=None,
        output=None,
        cal_column="file",
        col_remove="REMOVEIT",
        ifmany="error",
        timecol="DATE-OBS",
        timefmt=None,
        verbose=1
):
    """Make a general purpose reducing plan table.

    keys to select calibration frames

    Parameters
    ----------
    summary : `~pandas.DataFrame` The summary table of the files to be reduced.

    cal_summary : `~pandas.DataFrame` or list of such. The summary table of the
        calibration files. If `list`, lengths of `cal_summary` and `column`
        must be the same.

        ..note::
            If the file path is relative, the reference of these paths must be
            identical to that in `summary`. If not, future reduction has no way
            to find the corresponding calibration file.

    newcolname : str or list of str The column name(s) to be added to
        `summary`, which will contain the correspondong calibration frame
        information (file name). If `list`, lengths of `cal_summary` and
        `newcolname` must be the same.

    match_by : str or list of str, optional The column name(s) to be used for
        matching calibration frames. If `list`, lengths of `match_by` and
        `newcolname` must be the same. To give multiple column names, use a
        list of lists, so that ``(len(match_by))`` is the same as
        ``len(newcolname)``. Rows that have no matching calibration frames will
        be filled with `None`. Default: `None` (all of the columns in
        `cal_summary` are used to match the calibration frames)

    cal_column : str, optional The column name which contains the "value" (file
        name) in `cal_summary`. If str, it is assumed all `cal_summary` have
        the information at that column. Default: ``"file"``

    ifmany : str, optional
        The action to take when there are more than one calibration frames
        found::

          * ``"error"``: raise a `ValueError`
          * ``"ignore"``: ignore the calibration frames for this row
          * ``"first"``: use the first calibration frame
          * ``"last"``: use the last calibration frame
          * ``"time"``: use the frame with the closest time

        If multiple frames are found for the closest time, the 0-th frame is
        used among them (the default behavior of `~numpy.argmin`). `timecol`
        must be specified if `ifmany` is ``"time"``.

    timecol : str, optional.
        The column contains the time of the observation. The FITS standard is
        `"DATE-OBS"`. **Used only if `ifmany` is ``"time"``**.

    timefmt : str, optional
        The format of the time in `timecol`. **Used only if `ifmany` is
        ``"time"``**. Usually ``"isot"`` or ``"jd"``. All possible formats are::

          >>> list(Time.FORMATS)
          ['jd', 'mjd', 'decimalyear', 'unix', 'unix_tai', 'cxcsec', 'gps', 'plot_date',
           'stardate', 'datetime', 'ymdhms', 'iso', 'isot', 'yday', 'datetime64',
           'fits', 'byear', 'jyear', 'byear_str', 'jyear_str']

        Default: `None`

    Examples
    --------
    `df1` contains all the files to be reduced. `df2` and `df3` contains all
    the dark and flat frames, respectively, and the file path is in ``"FRM"``
    column:
    >>> df1  # OBJECTS file  EXPTIME FILTER 0    h0001.fits     20.0      H 1
    >>>            h0002.fits     20.0      H
    >>> ..            ...      ...    ...
    >>> 238  k0079.fits     20.0      K 239  k0080.fits     20.0      K [240
    >>> rows x 3 columns]

    >>> df2  # DARKS
    >>> #   FILTER  EXPTIME FRM
    >>> # 0      J       20   a.fits
    >>> # 1      J       30   b.fits
    >>> # 2      H       20   c.fits
    >>> # 3      H       30   d.fits
    >>> # 4      K       20   e.fits
    >>> # 5      K       30   f.fits

    >>> df3  # FLATS
    >>> #   FILTER  EXPTIME FRM
    >>> # 0      J       30   A.fits
    >>> # 1      H       30   B.fits
    >>> # 2      K       30   C.fits

    (1) While preparing for making master calibration frames:
    >>> df_flat = make_reduc_planner( df3, cal_summary=df2,
    >>>     newcolname="DARKFRM", match_by=[["FILTER", "EXPTIME"]],
    >>>     cal_column="FRM"
    >>> )
    >>> #   FILTER  EXPTIME FRM  REMOVEIT DARKFRM
    >>> # 0      J       30   A         0       b
    >>> # 1      H       30   B         0       d
    >>> # 2      K       30   C         0       f
    After this, one may make the master flat frame by


    (1) We want to add ``"FLATFRM"`` column to `df1`, fill with the
    corresponding calibration frame's path, by using (matching by) "FILTER":
    >>> df1_d = make_reduc_planner( df1, cal_summary=df3, newcolname="FLATFRM",
    >>>     match_by="FILTER",  # also possible: [["FILTER"]] cal_column="FRM"
    >>> )

    (2) Then add ``"DARKFRM"`` column to `df1_d`, fill with the corresponding
    calibration frame's path, by using (matching by) "EXPTIME", and "FILTER":
    >>> df1_df = make_reduc_planner( df1_d, cal_summary=df2,
    >>>     newcolname="DARKFRM", match_by=[["EXPTIME", "FILTER"]],  # Note:
    >>>     list of list to avoid length mismatch cal_column="FRM"
    >>> )

    (3) The above two steps can be combined into one step:
    >>> df1_df_onestep = make_reduc_planner( df1, cal_summary=[df3, df2],
    >>>     newcolname=["FLATFRM", "DARKFRM"], match_by=["FILTER", ["EXPTIME",
    >>>     "FILTER"]], cal_column="FRM"  # also possible: ["FRM", "FRM"]
    >>> )

    >>> df1_df_onestep
    >>> #            file  EXPTIME FILTER REMOVEIT FLATFRM DARKFRM
    >>> # 0    h0001.fits     20.0      H        0 B.fits  c.fits
    >>> # 1    h0002.fits     20.0      H        0 B.fits  c.fits
    >>> # ..          ...      ...    ...      ...    ...     ...
    >>> # 238  k0079.fits     20.0      K        0 C.fits  e.fits
    >>> # 239  k0080.fits     20.0      K        0 C.fits  e.fits
    >>> # [240 rows x 5 columns]

    >>> np.all(df1_df == df1_df_onestep)
    >>> # True

    Notes
    -----
    In the above case, ~ 0.4s on MBP 16" [2021, macOS 12.0.1, M1Pro, 8P+2E
    core, GPU 16-core, RAM 16GB]
    >>> %%timeit df1_df_onestep = make_reduc_planner( df1, cal_summary=[df3,
    >>> df2], newcolname=["FLATFRM", "DARKFRM"], match_by=["FILTER",
    >>> ["EXPTIME", "FILTER"]], cal_column="FRM"  # also possible: ["FRM",
    >>> "FRM"]
    >>> )
    >>> # 403 ms +- 11.2 ms per loop (mean +- std. dev. of 7 runs, 1 loop each)
    """
    df = summary.copy()

    cals = [cal_summary] if isinstance(cal_summary, pd.DataFrame) else list(cal_summary)
    cols = listify(newcolname)  # make as a list
    if match_by is None:
        for i, _cal in enumerate(cals):
            if not all(_cal.columns.isin(df.columns)):
                raise IndexError(
                    f"Some column of `cal_summary[{i}]` not found in `summary`. "
                    + "You may specify `match_by` to specify the columns to match."
                )
        mats = [[col for col in cal.columns if col != cal_column] for cal in cals]
    else:
        mats = listify(match_by)

    calcols = listify(cal_column)
    if len(calcols) == 1:
        calcols = calcols * len(cals)

    if not len(cals) == len(cols) == len(mats) == len(calcols):
        raise ValueError(
            "Length mismatch!! `cal_summary`, `newcolname`, `match_by`, `cal_column`: "
            + f"{len(cals)}, {len(cols)}, {len(mats)}, {len(calcols)}"
        )

    # Initialize new columns
    #  - REMOVEIT: whether to remove the matched row from the summary
    if col_remove is not None:
        df[col_remove] = 0
    # - <CAL>FRM: the matched calibration frame's path
    for c in cols:
        df[c] = None

    infos = "More than one calibration frame found for {} = {}"
    for idx, row in df.iterrows():
        if ifmany == "time":
            time_row = Time(row[timecol], format=timefmt)
        for col, cal, mat, calcol in zip(cols, cals, mats, calcols):
            mat = np.atleast_1d(mat)
            # If just a str, cal[mat]==row[mat] gives `Series`, not `DataFrame`,
            # hence `axis=1` raises error.
            try:
                sel = cal[(cal[mat] == row[mat]).all(axis=1)]
                if len(sel) == 1:
                    df.loc[idx, col] = sel[calcol].values[0]
                elif len(sel) > 1:
                    if ifmany == "error":
                        raise ValueError(infos.format(mat, row[mat].values))
                    elif ifmany == "ignore":
                        continue  # leave it as None
                    elif ifmany == "first":
                        df.loc[idx, col] = sel[calcol].values[0]
                    elif ifmany == "last":
                        df.loc[idx, col] = sel[calcol].values[-1]
                    elif ifmany == "time":
                        times_cal = sel[timecol].values
                        if times_cal.dtype.kind.startswith("O"):
                            # Usu. the str time (isot format) is in dtype
                            # "object" --> Time(time_cal.values, format="isot")
                            # raise `ValueError`. So convert it to str first.
                            # 2022-01-17 19:27:06 (KST: GMT+09:00) ysBach
                            times_cal = times_cal.astype(str)
                        times_cal = Time(times_cal, format=timefmt)
                        minidx = np.argmin(np.abs(time_row - times_cal))
                        df.loc[idx, col] = sel[calcol].values[minidx]
                    else:
                        raise ValueError(f"{ifmany=} not understood")
                else:
                    continue
            except (IndexError):  # no match
                continue

    if output is not None:
        df.to_csv(Path(output), index=False)

    return df


# def planner_add_crrej(
#         plan,
#         gain=1.,
#         rdnoise=0.,
#         sigclip=4.5,
#         sigfrac=0.5,
#         objlim=1.0,
#         satlevel=1.e+9,
#         niter=4,
#         sepmed=False,
#         cleantype='medmask',
#         fs="median",
#         psffwhm=2.5,
#         psfsize=7,
#         psfbeta=4.765,
# ):
#     """ Set crrej kwargs for `~ysfitsutilpy.preproc.crrej`.

#     Notes
#     -----
#     See `~ysfitsutilpy.preproc.crrej`.

#     All parameters are optional (default to IRAF version of LACosmic, except
#     for `satlevel`, because `np.inf` is difficult to be saved to csv in the
#     planner). Because the planner is meant to be a "simple" way of makeing
#     reduction plan, `psfk` (i.e., `fs` in ndarray) is not supported due to the
#     limitation of the CSV format.

#     If any is given as list-like, it must have length equal to the number of
#     `plan` rows. Otherwise, `~pandas` will raise ValueError.
#     """
#     if isinstance(fs, np.ndarray):
#         raise TypeError("ndarray of `fs` (the `psfk` parameter) is not supported yet.")

#     df = plan.copy()

#     df["gain"] = gain
#     df["rdnoise"] = rdnoise
#     df["sigclip"] = sigclip
#     df["sigfrac"] = sigfrac
#     df["objlim"] = objlim
#     df["satlevel"] = satlevel
#     df["niter"] = niter
#     df["sepmed"] = sepmed
#     df["cleantype"] = cleantype

#     psf_keys = parse_crrej_psf(
#         fs=fs,
#         psffwhm=psffwhm,
#         psfsize=psfsize,
#         psfbeta=psfbeta
#     )
#     for k, v in psf_keys.items():
#         df[k] = v  # If ndarray, pandas will raise errors.

#     return df


def fits_newpath(
        fpath,
        rename_by,
        mkdir_by=None,
        header=None,
        delimiter='_',
        fillnan="",
        fileext='.fits'
):
    ''' Gives the new path of the FITS file from header.

    Parameters
    ----------
    fpath : path-like
        The path to the original FITS file.

    rename_by : list of str, optional
        The keywords of the FITS header to rename by.

    mkdir_by : list of str, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the
        keys.

    header : Header object, optional
        The header to extract `rename_by` and `mkdir_by`. If `None`, the
        function will do ``header = fits.getheader(fpath)``.

    delimiter : str, optional
        The delimiter for the renaming.

    fillnan : str, optional
        The string that will be inserted if the keyword is not found from the
        header.

    fileext : str, optional
        The extension of the file name to be returned. Normally it should be
        ``'.fits'`` since this function is `fits_newname`, but you may prefer,
        e.g., ``'.fit'`` for some reason. If `fileext` does not start with a
        period (``"."``), it is automatically added to the final file name in
        front of the ``fileext``.

    Returns
    -------
    newpath : path
        The new path.
    '''

    if header is None:
        hdr = fits.getheader(fpath)
    else:
        hdr = header.copy()

    # First make file name without parent path
    hdrvals = []
    for k in rename_by:
        try:
            hdrvals.append(str(hdr[k]))
        except KeyError:
            hdrvals.append(fillnan)

    if not fileext.startswith('.'):
        fileext = f".{fileext}"

    newname = delimiter.join(list(hdrvals))  # just in case, re-listify...
    newname = newname + fileext
    newpath = Path(fpath.parent)

    if mkdir_by is not None:
        for k in mkdir_by:
            newpath = newpath / hdr[k]

    newpath = newpath / newname

    return newpath


def fitsrenamer(
        fpath=None,
        header=None,
        newtop=None,
        rename_by=["OBJECT"],
        mkdir_by=None,
        delimiter='_',
        archive_dir=None,
        keymap=None,
        key_deprecation=True,
        remove_keys=None,
        overwrite=False,
        fillnan="",
        trimsec=None,
        verbose=True,
        add_header=None
):
    ''' Renames a FITS file by ``rename_by`` with delimiter.

    Parameters
    ----------
    fpath : path-like
        The path to the target FITS file.

    header : Header, optional
        The header of the fits file, especially if you want to just overwrite
        the header with this.

    newtop : path-like
        The top path for the new FITS file. If `None`, the new path will share
        the parent path with `fpath`.

    rename_by : list of str, optional
        The keywords of the FITS header to rename by.

    mkdir_by : list of str, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the
        keys.

    delimiter : str, optional
        The delimiter for the renaming.

    archive_dir : path-like or None, optional
        Where to move the original FITS file. If `None`, the original file will
        remain there. Deleting original FITS is dangerous so it is only
        supported to move the files. You may delete files manually if needed.

    keymap : dict or None, optional
        If not `None`, the keymapping is done by using the dict of `keymap` in
        the format of ``{<standard_key>:<original_key>}``.

    key_deprecation : bool, optional
        Whether to change the original keywords' comments to contain
        deprecation warning. If `True`, the original keywords' comments will
        become ``Deprecated. See <standard_key>.``.

    trimsec : str or None, optional
        Region of ``CCDData`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details. Default is `None`.

    fillnan : str, optional
        The string that will be inserted if the keyword is not found from the
        header.

    remove_keys : list of str
        The header keywords to be removed.

    add_header : header or Card object
        The header keyword, value (and comment) to add after the renaming.

    Notes
    -----
    MEF(Multi-Extension FITS) currently is not supported.

    '''

    # Load fits file
    hdul = fits.open(fpath)
    data = hdul[0].data
    if header is None:
        hdr = hdul[0].header
    else:
        hdr = header.copy()
    hdul.close()

    # add keyword
    if add_header is not None:
        if (not isinstance(add_header, fits.Header)
                and not isinstance(add_header, fits.header.Card)):
            warn("add_header is not either Header or Card. "
                 + "Be careful about possible error.")
        hdr += add_header

    # Copy keys based on KEYMAP
    if keymap is not None:
        hdr = key_mapper(hdr, keymap, deprecation=key_deprecation)

    if remove_keys is not None:
        hdr = key_remover(hdr, remove_keys, deepremove=True)

    # TODO: It is necessary to do this bothersome calculations to
    #   preserve the WCS information that may reside in the FITS (if use
    #   ``trim_image`` of ccdproc, it will not be preserved).
    # TODO: Maybe I can put some LTV-like keys to the header, rather
    #   than this crazy code...? (ysBach 2019-05-09)
    if trimsec is not None:
        slices = ccdproc.utils.slices.slice_from_string(
            trimsec,
            fits_convention=True
        )
        # initially guess start and stop indices as 0's and from shape in (ny, nx) order
        ny, nx = data[slices].shape
        starts = np.array([0, 0])   # yx order
        stops = np.array([ny, nx])  # yx order

        for i in range(2):
            if slices[i].start is not None:
                starts[i] = slices[i].start
            if slices[i].stop is not None:
                stops[i] = slices[i].stop

        cent = np.flip((stops - starts) / 2)  # xy order
        size = (ny, nx)  # yx order
        # Make CCDData instance as dummy object
        _ccd = CCDData(data, header=hdr, unit='adu')
        _ccd = cut_ccd(_ccd, cent, size)
        data = _ccd.data
        hdr = _ccd.header

    newhdul = fits.PrimaryHDU(data=data, header=hdr)

    # Set the new path
    if verbose:
        print("Renaming file by ", end='')
        form = ''
        for rn in rename_by:
            form = form + f"<{rn:s}>{delimiter:s}"
        ndelimiter = len(delimiter)
        print(form[:-ndelimiter])
        if mkdir_by is not None:
            print("Make by ", end='')
            form = ''
            for md in mkdir_by:
                form = form + f"<{md:s}>/"
            print(form[:-1])

    newpath = fits_newpath(fpath,
                           rename_by,
                           mkdir_by=mkdir_by,
                           header=hdr,
                           delimiter=delimiter,
                           fillnan=fillnan,
                           fileext='fits')
    if newtop is not None:
        newpath = Path(newtop) / newpath.name

    mkdir(newpath.parent)

    if verbose:
        print(f"Rename {fpath.name} to {newpath}")

    newhdul.writeto(newpath, output_verify='fix', overwrite=overwrite)

    if archive_dir is not None:
        archive_dir = Path(archive_dir)
        archive_path = archive_dir / fpath.name
        mkdir(archive_path.parent)
        if verbose:
            print(f"Moving {fpath.name} to {archive_path}")
        fpath.rename(archive_path)

    return newpath
