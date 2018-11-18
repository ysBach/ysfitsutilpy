'''
Contians convenience funcitons which are
(1) more related to the file name or paths rather than the contents or
(2) related to the non-FITS files.
'''

from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.io import fits
from .hdrutil import key_mapper

__all__ = ["mkdir", "load_if_exists", "make_summary", "fits_newpath", "fitsrenamer"]


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
        The loader to load ``path``. Can be ``CCDData.read``, ``np.loadtxt``, etc.
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
    >>> ccd = load_if_exists(Path(".", "test.fits"), loader=CCDData.read, ext=0,
    >>>       if_not="print('File not found')")
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


def make_summary(filelist, extension=0, fname_option='relative',
                 output=None, format='ascii.csv',
                 keywords=[],
                 example_header=None, sort_by='file', verbose=True):
    """ Extracts summary from the headers of FITS files.
    Parameters
    ----------
    filelist: list of str (path-like)
        The list of file paths relative to the current working directory.

    extension: int or str
        The extension to be summarized.

    fname_option: str {'absolute', 'relative', 'name'}
        Whether to save full absolute/relative path or only the filename.

    output: str or path-like
        The directory and file name of the output summary file.

    format: str
        The astropy.table.Table output format.

    keywords: list or str(``"*"``)
        The list of the keywords to extract (keywords should be in str).

    example_header: str or path-like
        The path including the filename of the output summary text file.

    sort_by: str
        The column name to sort the results. It can be any element of
        ``keywords`` or ``'file'``, which sorts the table by the file name.

    Example
    -------
    >>> from pathlib import Path
    >>> import ysfitsutilpy as yfu
    >>> keys = ["OBS-TIME", "FILTER", "O`BJECT"]  # actually it is case-insensitive
    >>> # The keywords you want to extract (from the headers of FITS files)
    >>> TOPPATH = Path(".", "observation_2018-01-01")
    >>> # The toppath
    >>> savepath = TOPPATH / "summary_20180101.csv"
    >>> # path to save summary csv file
    >>> allfits = list((TOPPATH / "rawdata").glob("*.fits"))
    >>> # list of all the fits files in Path object
    >>> summary = yfu.make_summary(allfits, keywords=keys, fname_option='name',
    >>>                         sort_by="DATE-OBS", output=savepath)
    >>> # The astropy.table.Table format.
    >>> # If you want, you may change it to pandas:
    >>> summary_pd = summary.to_pandas()`
    """

    if len(filelist) == 0:
        print("No FITS file found.")
        return

    def _get_fname(path):
        if fname_option == 'relative':
            return str(path)
        elif fname_option == 'absolute':
            return str(path.absolute())
        else:
            return path.name

    options = ['absolute', 'relative', 'name']
    if fname_option not in options:
        raise KeyError(f"fname_option must be one of {options}.")

    skip_keys = ['COMMENT', 'HISTORY']

    if verbose:
        if (keywords != []) and (keywords != '*'):
            print("Extracting keys: ", keywords)
        str_example_hdr = "Extract example header from {:s}\n\tand save as {:s}"
        str_keywords = "All {:d} keywords will be loaded."
        str_keyerror_fill = "Key {:s} not found for {:s}, filling with nan."
        str_filesave = 'Saving the summary file to "{:s}"'

    # Save example header
    if example_header is not None:
        example_fits = filelist[0]
        if verbose:
            print(str_example_hdr.format(str(example_fits), example_header))
        ex_hdu = fits.open(example_fits)
        ex_hdr = ex_hdu[extension].header
        ex_hdr.totextfile(example_header, overwrite=True)

    # load ALL keywords for special cases
    if (keywords == []) or (keywords == '*'):
        example_fits = filelist[0]
        ex_hdu = fits.open(example_fits)
        ex_hdu.verify('fix')
        ex_hdr = ex_hdu[extension].header
        N_hdr = len(ex_hdr.cards)
        keywords = []

        for i in range(N_hdr):
            key_i = ex_hdr.cards[i][0]
            if (key_i in skip_keys):
                continue
            elif (key_i in keywords):
                str_duplicate = "Key {:s} is duplicated! Only first one will be saved."
                print(str_duplicate.format(key_i))
                continue
            keywords.append(key_i)

        if verbose:
            print(str_keywords.format(len(keywords)))
#            except fits.VerifyError:
#                str_unparsable = '{:d}-th key is skipped since it is unparsable.'
#                print(str_unparsable.format(i))
#                continue

    # Initialize
    summarytab = dict(file=[])
    for k in keywords:
        summarytab[k] = []

    # Run through all the fits files
    for fpath in filelist:
        summarytab["file"].append(_get_fname(fpath))
        hdu = fits.open(fpath)
        hdu.verify('fix')
        hdr = hdu[extension].header
        for k in keywords:
            try:
                summarytab[k].append(hdr[k])
            except KeyError:
                if verbose:
                    print(str_keyerror_fill.format(k, str(fpath)))
                summarytab[k].append(np.nan)
        hdu.close()

    summarytab = Table(summarytab)
    summarytab.sort(sort_by)

    if output is not None:
        output = Path(output)
        if verbose:
            print(str_filesave.format(str(output)))
        summarytab.write(output, format=format)

    return summarytab


def fits_newpath(fpath, rename_by, mkdir_by=None, header=None, delimiter='_',
                 ext='fits'):
    ''' Gives the new path of the FITS file from header.
    Parameters
    ----------
    fpath: path-like
        The path to the original FITS file.
    rename_by: list of str, optional
        The keywords of the FITS header to rename by.
    mkdir_by: list of str, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the keys.
    header: Header object, optional
        The header to extract ``mkdir_by``. If ``None``, the function will do
        ``header = fits.getheader(fpath)``.
    delimiter: str, optional
        The delimiter for the renaming.
    ext: str, optional
        The extension of the file name to be returned. Normally it should be
        ``'fits'`` since this function is ``fits_newname``.
    '''

    if header is None:
        header = fits.getheader(fpath)

    # First make file name without parent path
    newname = ""
    for k in rename_by:
        newname += str(header[k])
        newname += delimiter

    newname = newname[:-1] + '.fits'

    newpath = Path(fpath.parent)

    if mkdir_by is not None:
        for k in mkdir_by:
            newpath = newpath / header[k]

    newpath = newpath / newname

    return newpath


def fitsrenamer(fpath=None, header=None, newtop=None, rename_by=["OBJECT"],
                mkdir_by=None, delimiter='_', archive_dir=None, keymap=None,
                key_deprecation=True,
                verbose=True, add_header=None):
    ''' Renames a FITS file by ``rename_by`` with delimiter.
    Parameters
    ----------
    fpath: path-like
        The path to the target FITS file.
    header: Header, optional
        The header of the fits file. If given, don't open ``fpath``.
    newtop: path-like
        The top path for the new FITS file. If ``None``, the new path will share
        the parent path with ``fpath``.
    rename_by: list of str, optional
        The keywords of the FITS header to rename by.
    mkdir_by: list of str, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the keys.
    delimiter: str, optional
        The delimiter for the renaming.
    archive_dir: path-like or None, optional
        Where to move the original FITS file. If ``None``, the original file
        will remain there. Deleting original FITS is dangerous so it is only
        supported to move the files. You may delete files manually if needed.
    keymap: dict or None, optional
        If not ``None``, the keymapping is done by using the dict of ``keymap``
        in the format of ``{<standard_key>:<original_key>}``.
    key_deprecation: bool, optional
        Whether to change the original keywords' comments to contain deprecation
        warning. If ``True``, the original keywords' comments will become
        ``Deprecated. See <standard_key>.``.
    add_header: header or dict
        The header keyword, value (and comment) to add after the renaming.
    '''

    # Load fits file
    hdul = fits.open(fpath)
    if header is None:
        header = hdul[0].header

    # add keyword
    if add_header is not None:
        header += add_header

    # Copy keys based on KEYMAP
    if keymap is not None:
        header = key_mapper(header, keymap, deprecation=key_deprecation)

    newhdul = fits.PrimaryHDU(data=hdul[0].data, header=header)

    # Set the new path
    newpath = fits_newpath(fpath, rename_by, mkdir_by=mkdir_by, header=header,
                           delimiter=delimiter, ext='fits')
    if newtop is not None:
        newpath = Path(newtop) / newpath

    mkdir(newpath.parent)

    if verbose:
        print(f"Rename {fpath.name} to {newpath}")

    hdul.close()
    newhdul.writeto(newpath, output_verify='fix')

    if archive_dir is not None:
        archive_dir = Path(archive_dir)
        archive_path = archive_dir / fpath.name
        mkdir(archive_path.parent)
        if verbose:
            print(f"Moving {fpath.name} to {archive_path}")
        fpath.rename(archive_path)

    return newpath