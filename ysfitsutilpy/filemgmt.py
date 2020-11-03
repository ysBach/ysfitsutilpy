'''
Contians convenience funcitons which are
(1) more related to the file name or paths rather than the contents or
(2) related to the non-FITS files.
'''

from pathlib import Path
import numpy as np

from astropy.table import Table
from astropy.nddata import CCDData
from astropy.io import fits
from warnings import warn
import ccdproc
from .hdrutil import key_mapper, key_remover
from .ccdutil import cutccd
from .misc import inputs2pathlist, _getext

__all__ = ["mkdir", "load_if_exists",
           "make_summary", "fits_newpath", "fitsrenamer"]


def mkdir(fpath, mode=0o777, exist_ok=True):
    ''' Convenience function for Path.mkdir()
    '''
    fpath = Path(fpath)
    Path.mkdir(fpath, mode=mode, exist_ok=exist_ok)


def load_if_exists(path, loader, if_not=None, verbose=True, **kwargs):
    ''' Load a file if it exists.
    Parameters
    ----------
    path: pathlib.Path of Path-like str
        The path to be searched.
    loader: a function
        The loader to load ``path``. Can be ``CCDData.read``,
        ``np.loadtxt``, etc.
    if_not: str
        Give a python code as a str to be run if the loading failed.
    Returns
    -------
    loaded:
        The loaded file. If the file does not exist, ``None`` is returned.

    Example
    -------
    >>> from astropy.nddata import CCDData
    >>> from pathlib import Path
    >>> ccd = load_if_exists(Path(".", "test.fits"),
    >>>                      loader=CCDData.read, ext=0,
    >>>                      if_not="print('File not found')")
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


def make_summary(inputs=None, fitslist=None, ext=None, extname=None, extver=None,
                 fname_option='relative', output=None, format='ascii.csv',
                 keywords=[], example_header=None, sort_by='file',
                 pandas=False, verbose=True):
    """ Extracts summary from the headers of FITS files.
    Parameters
    ----------
    inputs : glob pattern or list-like of path-like
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or
        list of files (each element must be path-like). One and only one
        of ``inputs`` or ``fitslist`` must be provided.

    fitslist: list of str (path-like) or list of CCDData, optional
        The list of file paths relative to the current working
        directory, or the list of ccds to be summarized. It can be
        useful to give a list of CCDData if you have already
        stacked/loaded the CCDData into a list. Although it is not a
        good idea, a mixed list of CCDData and paths to the files is
        also acceptable. One and only one of ``inputs`` or ``fitslist``
        must be provided.

    ext : int
        The extension index (0-indexing).

    extname : str
        The extension name (``XTENSION``).

    extver : int
        The version of the extension; used only if extname is given.

    fname_option: str {'absolute', 'relative', 'name'}, optional
        Whether to save full absolute/relative path or only the
        filename.

    output: str or path-like, optional
        The directory and file name of the output summary file.

    format: str, optional
        The astropy.table.Table output format. Only works if ``pandas``
        is ``False``.

    keywords: list or str(``"*"``), optional
        The list of the keywords to extract (keywords should be in str).

    example_header: None or path-like, optional
        The path including the filename of the output summary text file.
        If specified, the header of the 0-th element of ``fitslist``
        will be extracted and saved to ``example_header``.

    pandas : bool, optional
        Whether to return pandas. If ``False``, astropy table object is
        returned. It will save csv format regardless of ``format``.

    sort_by: str, optional
        The column name to sort the results. It can be any element of
        ``keywords`` or ``'file'``, which sorts the table by the file
        name.

    Return
    ------
    summarytab: astropy.Table

    Example
    -------
    >>> from pathlib import Path
    >>> import ysfitsutilpy as yfu
    >>> keys = ["OBS-TIME", "FILTER", "OBJECT"]
    >>> # actually it is case-insensitive
    >>> # The keywords you want to extract (from the headers of FITS files)
    >>> TOPPATH = Path(".", "observation_2018-01-01")
    >>> # The toppath
    >>> savepath = TOPPATH / "summary_20180101.csv"
    >>> # list of all the fits files in TOPPATH/rawdata:
    >>> summary = yfu.make_summary(TOPPATH/"rawdata/*.fits", keywords=keys,
    >>>                            fname_option='name', pandas=True,
    >>>                            sort_by="DATE-OBS", output=savepath)
    """
    if (inputs is not None) + (fitslist is not None) != 1:
        raise ValueError("Give one and only one of fitslist/fpattern.")

    if fitslist is None:
        fitslist = inputs2pathlist(inputs, sorted=True)

    if len(fitslist) == 0:
        print("No FITS file found.")
        return

    def _get_fname(path):
        if fname_option == 'relative':
            return str(path)
        elif fname_option == 'absolute':
            return str(path.absolute())
        elif fname_option == 'name':
            return path.name

    def _get_hdr(item, extension):
        ''' Gets header from ``item``.
        '''
        if isinstance(item, CCDData):
            hdr = item.header
        else:
            hdul = fits.open(item)
            hdr = hdul[extension].header
            hdul.close()
        return hdr

    _valid_options = ['absolute', 'relative', 'name']
    if fname_option not in _valid_options:
        raise KeyError(f"fname_option must be one of {_valid_options}.")

    skip_keys = ['COMMENT', 'HISTORY']
    str_example_hdr = "Extract example header from 0-th\n\tand save as {:s}"
    str_keywords = "All {:d} keywords (estimated from {:s}) will be loaded."
    str_keyerror_fill = "Key {:s} not found for {:s}, filling with None."
    str_filesave = 'Saving the summary file to "{:s}"'
    str_duplicate = ("Key {:s} is duplicated! "
                     + "Only the first one will be saved.")

    if verbose:
        if (keywords != []) and (keywords != '*'):
            print("Extracting keys: ", keywords)

    extension = _getext(ext=ext, extname=extname, extver=extver)

    # Save example header
    if example_header is not None:
        example_fits = fitslist[0]
        if verbose:
            print(str_example_hdr.format(example_header))
        ex_hdr = _get_hdr(example_fits, extension=extension)
        ex_hdr.totextfile(example_header, overwrite=True)

    # load ALL keywords for special cases
    if (keywords == []) or (keywords == '*'):
        example_fits = fitslist[0]
        ex_hdr = _get_hdr(example_fits, extension=extension)
        N_hkeys = len(ex_hdr.cards)
        keywords = []

        for i in range(N_hkeys):
            key_i = ex_hdr.cards[i][0]
            if (key_i in skip_keys):
                continue
            elif (key_i in keywords):
                warn(str_duplicate.format(key_i))
                continue
            keywords.append(key_i)

        if verbose:
            print(str_keywords.format(len(keywords), fitslist[0]))

    # Initialize
    summarytab = dict(file=[], filesize=[])
    for k in keywords:
        summarytab[k] = []

    # Run through all the fits files
    for i, item in enumerate(fitslist):
        if isinstance(item, CCDData):
            summarytab["file"].append(None)
            summarytab["filesize"].append(None)
        else:
            summarytab["file"].append(_get_fname(item))
            summarytab["filesize"].append(Path(item).stat().st_size)
            # Don't change to MB/GB, which will make it float...
        hdr = _get_hdr(item, extension=extension)
        for k in keywords:
            try:
                summarytab[k].append(hdr[k])
            except KeyError:
                if verbose:
                    if isinstance(item, CCDData):
                        warn(str_keyerror_fill.format(k, f"fitslist[{i}]"))
                    else:
                        warn(str_keyerror_fill.format(k, str(item)))
                summarytab[k].append(None)

    if pandas:
        import pandas as pd
        summarytab = pd.DataFrame.from_dict(summarytab)
        if sort_by is not None:
            summarytab.sort_values(sort_by, inplace=True)
        summarytab.reset_index(drop=True, inplace=True)

        if output is not None:
            output = Path(output)
            if verbose:
                print(str_filesave.format(str(output)))
            summarytab.to_csv(output, index=False)

    else:
        summarytab = Table(summarytab)
        if sort_by is not None:
            summarytab.sort(sort_by)

        if output is not None:
            output = Path(output)
            if verbose:
                print(str_filesave.format(str(output)))
            summarytab.write(output, format=format)

    return summarytab


def fits_newpath(fpath, rename_by, mkdir_by=None, header=None, delimiter='_',
                 fillnan="", fileext='.fits'):
    ''' Gives the new path of the FITS file from header.
    Parameters
    ----------
    fpath : path-like
        The path to the original FITS file.

    rename_by : list of str, optional
        The keywords of the FITS header to rename by.

    mkdir_by : list of str, optional
        The keys which will be used to make subdirectories to classify
        files. If given, subdirectories will be made with the header
        value of the keys.

    header : Header object, optional
        The header to extract ``rename_by`` and mkdir_by``. If ``None``,
        the function will do ``header = fits.getheader(fpath)``.

    delimiter : str, optional
        The delimiter for the renaming.

    fillnan : str, optional
        The string that will be inserted if the keyword is not found
        from the header.

    fileext : str, optional
        The extension of the file name to be returned. Normally it
        should be ``'fits'`` since this function is ``fits_newname``,
        but you may prefer, e.g., ``'fit'`` for some reason. If
        ``fileext`` does not start with ``"."``, the dot is
        automatically added to the final file name in front of the
        ``fileext``.
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


def fitsrenamer(fpath=None, header=None, newtop=None, rename_by=["OBJECT"],
                mkdir_by=None, delimiter='_', archive_dir=None, keymap=None,
                key_deprecation=True, remove_keys=None, overwrite=False,
                fillnan="", trim_fits_section=None, verbose=True,
                add_header=None):
    ''' Renames a FITS file by ``rename_by`` with delimiter.
    Note
    ----
    MEF(Multi-Extension FITS) currently is not supported.

    Parameters
    ----------
    fpath : path-like
        The path to the target FITS file.

    header : Header, optional
        The header of the fits file, especially if you want to just
        overwrite the header with this.

    newtop : path-like
        The top path for the new FITS file. If ``None``, the new path
        will share the parent path with ``fpath``.

    rename_by : list of str, optional
        The keywords of the FITS header to rename by.

    mkdir_by : list of str, optional
        The keys which will be used to make subdirectories to classify
        files. If given, subdirectories will be made with the header
        value of the keys.

    delimiter : str, optional
        The delimiter for the renaming.

    archive_dir : path-like or None, optional
        Where to move the original FITS file. If ``None``, the original
        file will remain there. Deleting original FITS is dangerous so
        it is only supported to move the files. You may delete files
        manually if needed.

    keymap : dict or None, optional
        If not ``None``, the keymapping is done by using the dict of
        ``keymap`` in the format of ``{<standard_key>:<original_key>}``.

    key_deprecation : bool, optional
        Whether to change the original keywords' comments to contain
        deprecation warning. If ``True``, the original keywords'
        comments will become ``Deprecated. See <standard_key>.``.

    trim_fits_section : str or None, optional
        Region of ``ccd`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details.
        Default is ``None``.

    fillnan : str, optional
        The string that will be inserted if the keyword is not found
        from the header.

    remove_keys : list of str
        The header keywords to be removed.

    add_header: header or Card object
        The header keyword, value (and comment) to add after the
        renaming.
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
    if trim_fits_section is not None:
        slices = ccdproc.utils.slices.slice_from_string(trim_fits_section,
                                                        fits_convention=True)
        # initially guess start and stop indices as 0's and from shape
        # in (ny, nx) order
        ny, nx = data[slices].shape
        starts = np.array([0, 0])  # yx order
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
        _ccd = cutccd(_ccd, cent, size)
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
