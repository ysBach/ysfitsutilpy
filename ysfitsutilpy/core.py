from pathlib import Path
import warnings
import ccdproc
import re

import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

from . import util


__all__ = ['mkdir', 'read_without_newline', 'load_ccd', 'load_if_exists',
           'make_summary',
           "fits_newpath", "fitsrenamer",
           'convert_bit', 'center_coord', "fitsxy2py", "dB2epadu", "epadu2dB",
           "CCDData_astype", "make_errmap",
           'wcsremove', 'stack_FITS']


def mkdir(fpath, mode=0o777, exist_ok=True):
    ''' Convenience function for Path.mkdir()
    '''
    fpath = Path(fpath)
    Path.mkdir(fpath, mode=mode, exist_ok=exist_ok)


def read_without_newline(fpath):
    with open(fpath, 'r') as ff:
        contents = ff.read().splitlines()
    return contents


# FIXME: remove it when astropy updated.
def load_ccd(path, extension=0, unit='adu'):
    ''' CCDData.read cannot read TPV WCS
    https://github.com/astropy/astropy/issues/7650
    '''
    hdu = fits.open(path)[extension]
    ccd = CCDData(data=hdu.data, header=hdu.header, wcs=WCS(hdu.header),
                  unit='adu')
    return ccd


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


# def get_abs_path(path=''):
#    ''' Changes input path to absolute path
#    '''
#    if path == '':
#        path = os.getcwd()
#
#    # if path is relative, change to absolute
#    if not os.path.isabs(path):
#        path = os.path.join(os.getcwd(), path)
#
#    # if the path does not exist,
#    if not os.path.isdir(path):
#        raise NameError('The path {:s} does not exist'.format(path))
#
#    # if path is absolute, do nothing.
#
#    return path


def make_summary(filelist, extension=0, fname_option='relative',
                 output=None, format='ascii.csv',
                 keywords=[], dtypes=[],
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

    dtypes: list
        The list of dtypes of keywords if you want to specify. If ``[]``,
        ``['U80'] * len(keywords)`` will be used. Otherwise, it should have
        the same length with ``keywords``.

    example_header: str or path-like
        The path including the filename of the output summary text file.

    sort_by: str
        The column name to sort the results. It can be any element of
        ``keywords`` or ``'file'``, which sorts the table by the file name.
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
    summarytab = dict(fnames=[])
    for k in keywords:
        summarytab[k] = []

    # Run through all the fits files
    for fpath in filelist:
        summarytab["fnames"].append(_get_fname(fpath))
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
#                try:
#                except ValueError:
#                    raise ValueError(str_valerror.format('U80'))

    summarytab = Table(summarytab)

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
        header = util.key_mapper(header, keymap, deprecation=key_deprecation)

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


def convert_bit(fname, original_bit=12, target_bit=16):
    ''' Converts a FIT(S) file's bit.
    In ASI1600MM, for example, the output data is 12-bit but since FITS
    standard do not accept 12-bit (but the closest integer is 16-bit), so,
    for example, the pixel values can have 0 and 15, but not any integer
    between these two. So it is better to convert to 16-bit.
    '''
    hdul = fits.open(fname)
    dscale = 2**(target_bit - original_bit)
    hdul[0].data = (hdul[0].data / dscale).astype('int')
    hdul[0].header['MAXDATA'] = (2**original_bit - 1,
                                 "maximum valid physical value in raw data")
    # hdul[0].header['BITPIX'] = target_bit
    # FITS ``BITPIX`` cannot have, e.g., 12, so the above is redundant line.
    hdul[0].header['BUNIT'] = 'ADU'

    return hdul


def center_coord(header, skycoord=False):
    ''' Gives the sky coordinate of the center of the image field of view.
    Parameters
    ----------
    header: astropy.header.Header
        The header to be used to extract WCS information (and image size)
    skycoord: bool
        Whether to return in the astropy.coordinates.SkyCoord object. If
        ``False``, a numpy array is returned.
    '''
    wcs = WCS(header)
    cx = float(header['naxis1']) / 2 - 0.5
    cy = float(header['naxis2']) / 2 - 0.5
    center_coo = wcs.wcs_pix2world(cx, cy, 0)
    if skycoord:
        return SkyCoord(*center_coo, unit='deg')
    return np.array(center_coo)


def fitsxy2py(fits_section):
    ''' Given FITS section in str, returns the slices in python convention.
    Note
    ----
    >>> np.eye(5)[ccdproc.utils.slices.slice_from_string('[1:2,:]',
        fits_convention=True)]
    # array([[1., 0.],
    #       [0., 1.],
    #       [0., 0.],
    #       [0., 0.],
    #       [0., 0.]])
    '''
    slicer = ccdproc.utils.slices.slice_from_string
    sl = slicer(fits_section, fits_convention=True)
    return sl


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB):
    return 5 / 10**(gain_dB / 20)


def epadu2dB(gain_epadu):
    return 20 * np.log10(5 / gain_epadu)


def CCDData_astype(ccd, dtype='float32', uncertainty_dtype=None):
    ''' Assign dtype to the CCDData object.
    Parameters
    ----------
    ccd: CCDData
        The ccd to be astyped.
    dtype: dtype-like
        The dtype to be applied to the data
    uncertainty_dtype: dtype-like
        The dtype to be applied to the uncertainty. Be default, use the
        same dtype as data (``uncertainty_dtype = dtype``).
    '''
    nccd = ccd.copy()
    nccd.data = nccd.data.astype(dtype)

    try:
        if uncertainty_dtype is None:
            uncertainty_dtype = dtype
        nccd.uncertainty.array = nccd.uncertainty.array.astype(dtype)
    except AttributeError:
        # If there is no uncertainty attribute in the input ``ccd``
        pass

    return nccd


def get_from_header(header, key, unit=None, verbose=True,
                    default=0):
    ''' Get a variable from the header object.
    Parameters
    ----------
    header: Header
        The header to extract the value.
    key: str
        The header keyword to extract.
    unit: astropy unit
        The unit of the value.
    default: str, int, float, ..., or Quantity
        The default if not found from the header.
    '''
    value = None

    try:
        value = header[key]
        if unit is not None:
            value = value * unit
        if verbose:
            print(f"header: {key} = {value}")

    except KeyError:
        if default is not None:
            value = util.change_to_quantity(default, desired=unit)
            warnings.warn(f"{key} not found in header: setting to {default}.")

    return value


def make_errmap(ccd, gain_epadu=1, rdnoise_electron=0,
                subtracted_dark=None):
    ''' Calculate the usual error map.
    Parameters
    ----------
    ccd: array-like
        The ccd data which will be used to generate error map. It must be bias
        subtracted. If dark is subtracted, give ``subtracted_dark``. If the
        amount of this subtracted dark is negligible, you may just set
        ``subtracted_dark = None`` (default).
    gain: float, array-like, or Quantity, optional.
        The effective gain factor in ``electron/ADU`` unit.
    rdnoise: float, array-like, or Quantity, optional.
        The readout noise. Put ``rdnoise=0`` will calculate only the Poissonian
        error. This is useful when generating noise map for dark frames.
    subtracted_dark: array-like
        The subtracted dark map.
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

    # Get Poisson noise
    if subtracted_dark is not None:
        dark = subtracted_dark.copy()
        if isinstance(dark, CCDData):
            dark = dark.data
        # If subtracted dark is negative, this may cause negative pixel in ``data``:
        data += dark

    var_Poisson = data / gain_epadu  # (data * gain) / gain**2 to make it ADU
    var_RDnoise = (rdnoise_electron / gain_epadu)**2

    errmap = np.sqrt(var_Poisson + var_RDnoise)

    return errmap


def wcsremove(filename, additional_keys=[], extension=0, output=None,
              verify='fix', overwrite=False, verbose=True):
    ''' Remove most WCS related keywords from the header.
    additional_keys : list of regex str, optional
        Additional keys given by the user to be 'reset'. It must be in regex
        expression. Of course regex accepts just string, like 'NAXIS1'.

    output: str or Path
        The output file path.
    '''

    # Define header keywords to be deleted in regex:
    re2remove = ['CD[0-9]_[0-9]',  # Coordinate Description matrix
                 'CTYPE[0-9]',  # e.g., 'RA---TAN' and 'DEC--TAN'
                 'CUNIT[0-9]',  # e.g., 'deg'
                 'CRPIX[0-9]',  # The reference pixels in image coordinate
                 'CRVAL[0-9]',  # The world cooordinate values at CRPIX[1, 2]
                 'CROTA[0-9]',  # The angle between image Y and world Y axes
                 'CDELT[0-9]',  # with CROTA, older version of CD matrix.
                 'CRDELT[0-9]',
                 'CFINT[0-9]',
                 'LTM[0-9]_[0-9]',
                 'LTV[0-9]*',
                 'PIXXMIT',
                 'PIXOFFST',
                 'WAT[0-9]_[0-9]',  # For TNX and ZPX, e.g., "WAT1_001"
                 'C0[0-9]_[0-9]',  # polynomial CD by imwcs
                 'PC[0-9]_[0-9]',
                 'PV[0-9]_[0-9]',
                 '[A,B]_[0-9]_[0-9]',  # For SIP
                 '[A,B][P]?_ORDER',   # For SIP
                 '[A,B][P]?_DMAX',    # For SIP
                 'WCS[A-Z]',          # see below
                 'AST_[A-Z]',         # astrometry.net
                 'ASTIRMS[0-9]',      # astrometry.net
                 'ASTRRMS[0-9]',      # astrometry.net
                 'FGROUPNO',          # SCAMP field group label
                 'ASTINST',           # SCAMP astrometric instrument label
                 'FLXSCALE',          # SCAMP relative flux scale
                 'MAGZEROP',          # SCAMP zero-point
                 'PHOTIRMS',          # mag dispersion RMS (internal, high S/N)
                 'PHOTINST',          # SCAMP photometric instrument label
                 'PHOTLINK',          # True if linked to a photometric field
                 'SECPIX[0-9]'
                 ]
    # WCS[A-Z] captures, WCS[DIM, RFCAT, IMCAT, MATCH, NREF, TOL, SEP],
    # but not [IM]WCS, for example. These are likely to have been inserted
    # by WCS updating tools like astrometry.net or WCSlib/WCSTools. I
    # intentionally ignored IMWCS just for future reference.

    re2remove = re2remove + list(additional_keys)

    # If following str is in comment, suggest it if verbose
    candidate_re = ['wcs', 'axis', 'axes', 'coord', 'distortion', 'reference']
    candidate_key = []

    hdul = fits.open(filename)
    hdr = hdul[extension].header

    if verbose:
        print("Removed keywords: ", end='')

    for k in list(hdr.keys()):
        com = hdr.comments[k]
        deleted = False
        for re_i in re2remove:
            if re.match(re_i, k) is not None:
                hdr.remove(k)
                deleted = True
                if verbose:
                    print(f"{k}", end=' ')
                continue
        if not deleted:
            for re_cand in candidate_re:
                if re.match(re_cand, com):
                    candidate_key.append(k)
    if verbose:
        print('\n')

    if len(candidate_key) != 0 and verbose:
        print(
            f'\nFollowing keys may be related to WCS too:\n\t{candidate_key}')

    hdul[extension].header = hdr

    if output is not None:
        hdul.writeto(output, output_verify=verify, overwrite=overwrite)

    return hdul


def stack_FITS(filelist, extension, unit='adu', trim_fits_section=None,
               type_key=None, type_val=None):
    ''' Stacks the FITS files specified in filelist
    Parameters
    ----------
    filelist: str, path-like, or list of such
        The list of FITS files to be stacked

    extension: int or str
        The extension of FITS to be stacked. For single extension, set it as 0.

    unit: Unit or str, optional

    trim_fits_section: str, optional
        Region of ``ccd`` to be trimmed; see ``ccdproc.subtract_overscan`` for
        details. Default is None.

    Return
    ------
    all_ccd: list
        list of ``CCDData``
    '''
    def _parse_val(value):
        val = str(value)
        if val.lstrip('+-').isdigit():
            result = int(val)
        else:
            try:
                result = float(val)
            except ValueError:
                result = str(val)
        return result

    iskey = False
    filelist = list(filelist)

    if ((type_key is None) ^ (type_val is None)):
        raise KeyError(
            "type_key and type_val must be both specified or both None.")

    if type_key is not None:
        iskey = True
        if isinstance(type_key, str):
            type_key = [type_key]
        if isinstance(type_val, str):
            type_val = [type_val]

        if len(type_key) != len(type_val):
            raise ValueError(
                "type_key and type_val must be of the same length.")

    all_ccd = []

    for i, fname in enumerate(filelist):
        if unit is not None:
            ccd_i = CCDData.read(fname, hdu=extension, unit=unit)
        else:
            ccd_i = CCDData.read(fname, hdu=extension)

        if iskey:
            mismatch = False
            for k, v in zip(type_key, type_val):
                hdr_val = _parse_val(ccd_i.header[k])
                if (hdr_val != v):
                    mismatch = True
                    break
            if mismatch:
                continue

        if trim_fits_section is not None:
            ccd_i = ccdproc.trim_image(ccd_i, fits_section=trim_fits_section)

        all_ccd.append(ccd_i)
#        im_i = hdu_i[extension].data
#        if (i == 0):
#            all_data = im_i
#        elif (i > 0):
#            all_data = np.dstack( (all_data, im_i) )

    if len(all_ccd) == 0:
        if iskey:
            warnings.warn('No FITS file had "{:s} = {:s}"'.format(str(type_key),
                                                                  str(type_val))
                          + "Maybe int/float confusing?")

        else:
            warnings.warn('No FITS file found')
    else:
        if iskey:
            print('{:d} FITS files with "{:s} = {:s}"'
                  ' are loaded.'.format(len(all_ccd),
                                        str(type_key),
                                        str(type_val)))
        else:
            print('{:d} FITS files are loaded.'.format(len(all_ccd)))

    return all_ccd
