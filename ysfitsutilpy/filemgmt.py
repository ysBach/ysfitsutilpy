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
from astropy.table import Table

from .hduutil import (_parse_extension, cut_ccd, inputs2list, key_mapper,
                      key_remover)

__all__ = [
    "mkdir", "load_if_exists", "make_summary", "fits_newpath", "fitsrenamer"
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
\        keywords=None,
        example_header=None,
        sort_by='file',
        fullmatch={},
        query_str=None,
        verbose=True
):
    """ Extracts summary from the headers of FITS files.

    Parameters
    ----------
    inputs : glob pattern, list-like of path-like, list-like of CCDData
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or list of
        files (each element must be path-like or CCDData). Although it is not a
        good idea, a mixed list of CCDData and paths to the files is also
        acceptable.

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

    fullmatch : dict, optional.
        The ``{column: regex}`` dictionary for `~pandas.Series.str.fullmatch`.

    query_str : str, optional.
        The str used for `~pandas.DataFrame.query`.

    Return
    ------
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

    >>> # fullmatch = {"OBJECT": "DA.*"}
    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"}, query_str="EXPTIME in [2, 3]"
    """
    # No need to sort here because the real "sort" will be done later based on ``sort_by`` column.
    fitslist = inputs2list(inputs, sort=False, accept_ccdlike=True, check_coherency=False)

    if len(fitslist) == 0:
        if verbose:
            print("No FITS file found.")
        return None

    def _get_fname_fsize_hdr(item, idx, extension):
        if isinstance(item, CCDData):
            # NB: CCDData does not support extension (only available when it is being read)!
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
            fsize = Path(item).stat().st_size  # Don't change to MB/GB, which will make it float...
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
    if sort_by is not None:
        summarytab.sort_values(sort_by, inplace=True)
    summarytab.reset_index(drop=True, inplace=True)

    if fullmatch:
        select_mask = np.ones(len(summarytab), dtype=bool)
        for k, v in fullmatch.items():
            try:
                select_mask &= summarytab[k].str.fullmatch(v, case=True)
            except AttributeError:
                raise AttributeError(
                    f"{k} is not a string column in the dataframe! "
                    + "You may use `query_str` instead."
                )
        summarytab = summarytab[select_mask]

    if query_str is not None:
        summarytab = summarytab.query(query_str)

    if output is not None:
        output = Path(output)
        if verbose:
            print(f'Saving the summary to "{str(output)}"')
        summarytab.to_csv(output, index=False)

    return summarytab


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
        trim_fits_section=None,
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

    trim_fits_section : str or None, optional
        Region of ``CCDData`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details. Default is `None`.

    fillnan : str, optional
        The string that will be inserted if the keyword is not found from the
        header.

    remove_keys : list of str
        The header keywords to be removed.

    add_header : header or Card object
        The header keyword, value (and comment) to add after the renaming.

    Note
    ----
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
        if (not isinstance(add_header, fits.Header) and not isinstance(add_header, fits.header.Card)):
            warn("add_header is not either Header or Card. Be careful about possible error.")
        hdr += add_header

    # Copy keys based on KEYMAP
    if keymap is not None:
        hdr = key_mapper(hdr, keymap, deprecation=key_deprecation)

    if remove_keys is not None:
        hdr = key_remover(hdr, remove_keys, deepremove=True)

    # TODO: It is necessary to do this bothersome calculations to
    #   preserve the WCS information that may reside in the FITS (if use ``trim_image`` of ccdproc, it
    #   will not be preserved).
    # TODO: Maybe I can put some LTV-like keys to the header, rather
    #   than this crazy code...? (ysBach 2019-05-09)
    if trim_fits_section is not None:
        slices = ccdproc.utils.slices.slice_from_string(trim_fits_section, fits_convention=True)
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
