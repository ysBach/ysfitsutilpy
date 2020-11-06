'''
Collection of functions that are rather header-dependent than the data.
'''
import re
from warnings import warn

import numpy as np
from astropy import units as u
from astropy import wcs as astropywcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .misc import change_to_quantity, str_now

__all__ = ["add_to_header",
           "wcs_crota", "center_radec", "key_remover", "key_mapper",
           "get_from_header", "get_if_none", "wcsremove", "fov_radius",
           "convert_bit"]


def add_to_header(header, histcomm, s, precision=3, fmt="{:.>72s}", t_ref=None,
                  dt_fmt="(dt = {:.3f} s)", verbose=False):
    ''' Automatically add timestamp as well as history string

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

    fmt : str, None, optional.
        The Python 3 format string to format the time in the header. If `None`, the timestamp string
        will not be added.

        Examples::
          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in parentheses ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with ``_``.

    t_ref : Time
        The reference time. If not `None`, delta time is calculated.

    dt_fmt : str, optional.
        The Python 3 format string to format the delta time in the header.

    verbose : bool, optional.
        Whether to print the same information on the output terminal.

    verbose_fmt : str, optional.
        The Python 3 format string to format the time in the terminal.
    '''
    if isinstance(s, str):
        s = [s]

    if histcomm.lower() in ['h', 'hist', 'history']:
        for _s in s:
            header.add_history(_s)
            if verbose:
                print(f"HISTORY {_s}")
        if fmt is not None:
            timestr = str_now(precision=precision, fmt=fmt, t_ref=t_ref, dt_fmt=dt_fmt)
            header.add_history(timestr)
            if verbose:
                print(f"HISTORY {timestr}")

    elif histcomm.lower() in ['c', 'comm', 'comment']:
        for _s in s:
            header.add_comment(s)
            if verbose:
                print(f"COMMENT {_s}")
        if fmt is not None:
            timestr = str_now(precision=precision, fmt=fmt, t_ref=t_ref, dt_fmt=dt_fmt)
            header.add_comment(timestr)
            if verbose:
                print(f"COMMENT {timestr}")


def update_tlm(header):
    header.set("FITS-TLM",
               value=Time(Time.now(), precision=0).isot,
               comment="UT of last modification of this FITS file",
               after=f"NAXIS{header['NAXIS']}")


def wcs_crota(wcs, degree=True):
    '''
    Note
    ----
    https://iraf.net/forum/viewtopic.php?showtopic=108893
    CROTA2 = arctan (-CD1_2 / CD2_2) = arctan ( CD2_1 / CD1_1)
    '''
    if isinstance(wcs, astropywcs.WCS):
        wcsprm = wcs.wcs
    elif isinstance(wcs, astropywcs.Wcsprm):
        wcsprm = wcs
    else:
        raise TypeError("wcs type not understood. It must be either astropy.wcs.WCS or astropy.wcs.Wcsprm")

    # numpy arctan2 gets y-coord (numerator) and then x-coord(denominator)
    crota = np.arctan2(wcsprm.cd[0, 0], wcsprm.cd[1, 0])
    if degree:
        crota = np.rad2deg(crota)

    return crota


def center_radec(header, center_of_image=True, ra_key="RA", dec_key="DEC",
                 equinox=None, frame=None, equinox_key="EPOCH",
                 frame_key="RADECSYS", ra_unit=u.hourangle, dec_unit=u.deg,
                 mode='all', verbose=True, plain=False):
    ''' Returns the central ra/dec from header or WCS.
    Note
    ----
    Even though RA or DEC is in sexagesimal, e.g., "20 53 20", astropy correctly reads it in such a
    form, so no worries.

    Parameters
    ----------
    header : Header
        The header to extract the central RA/DEC from keywords or WCS.

    center_of_image : bool, optional
        If `True`, WCS information will be extracted from the header, rather than relying on the
        ``ra_key`` and ``dec_key`` keywords directly. If `False`, ``ra_key`` and ``dec_key`` from the
        header will be understood as the "center" and the RA, DEC of that location will be returned.

    equinox, frame : str, optional
        The ``equinox`` and ``frame`` for SkyCoord. Default (`None`) will use the default of
        SkyCoord. Important only if ``usewcs=False``.

    XX_key : str, optional
        The header key to find XX if ``XX`` is `None`. Important only if ``usewcs=False``.

    XX_unit : Quantity, optional
        The unit of ``XX``. Important only if ``usewcs=False``.

    mode : 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or only including only the
        core WCS transformation (``'wcs'``). Important only if ``usewcs=True``.

    plain : bool
        If `True`, only the values of RA/DEC in degrees will be returned.
    '''
    if center_of_image:
        w = WCS(header)
        nx, ny = float(header["NAXIS1"]), float(header["NAXIS2"])
        centx = nx / 2 - 0.5
        centy = ny / 2 - 0.5
        coo = SkyCoord.from_pixel(centx, centy, wcs=w, origin=0, mode=mode)
    else:
        ra = get_from_header(header, ra_key, verbose=verbose)
        dec = get_from_header(header, dec_key, verbose=verbose)
        if equinox is None:
            equinox = get_from_header(header, equinox_key, verbose=verbose, default=None)
        if frame is None:
            frame = get_from_header(header, frame_key, verbose=verbose, default=None).lower()
        coo = SkyCoord(ra=ra, dec=dec, unit=(ra_unit, dec_unit), frame=frame, equinox=equinox)

    if plain:
        return coo.ra.value, coo.dec.value
    return coo


def fov_radius(header, unit=u.deg):
    ''' Calculates the rough radius (cone) of the (square) FOV using WCS.

    Parameter
    ---------
    header: Header
        The header to extract WCS information.

    Return
    ------
    radius: `~astropy.Quantity`
        The radius in degrees
    '''
    w = WCS(header)
    nx, ny = float(header["NAXIS1"]), float(header["NAXIS2"])
    # Rough calculation, so use mode='wcs'
    c1 = SkyCoord.from_pixel(0, 0, wcs=w, origin=0, mode='wcs')
    c2 = SkyCoord.from_pixel(nx, 0, wcs=w, origin=0, mode='wcs')
    c3 = SkyCoord.from_pixel(0, ny, wcs=w, origin=0, mode='wcs')
    c4 = SkyCoord.from_pixel(nx, ny, wcs=w, origin=0, mode='wcs')
    r1 = c1.separation(c3).value / 2
    r2 = c2.separation(c4).value / 2
    r = max(r1, r2) * u.deg
    return r.to(unit)


def key_remover(header, remove_keys, deepremove=True):
    ''' Removes keywords from the header.

    Parameters
    ----------
    header : Header
        The header to be modified

    remove_keys : list of str
        The header keywords to be removed.

    deepremove : True, optional
        FITS standard does not have any specification of duplication of keywords as discussed in the
        following issue: https://github.com/astropy/ccdproc/issues/464 If it is set to `True`, ALL
        the keywords having the name specified in ``remove_keys`` will be removed. If not, only the
        first occurence of each key in ``remove_keys`` will be removed. It is more sensical to set it
        `True` in most of the cases.
    '''
    nhdr = header.copy()
    if deepremove:
        for key in remove_keys:
            while True:
                try:
                    nhdr.remove(key)
                except KeyError:
                    break
    else:
        for key in remove_keys:
            try:
                nhdr.remove(key)
            except KeyError:
                continue

    return nhdr


def key_mapper(header, keymap, deprecation=False, remove=False):
    ''' Update the header to meed the standard (keymap).

    Parameters
    ----------
    header : Header
        The header to be modified

    keymap : dict
        The dictionary contains ``{<standard_key>:<original_key>}`` information

    deprecation : bool, optional
        Whether to change the original keywords' comments to contain deprecation warning. If `True`,
        the original keywords' comments will become ``Deprecated. See <standard_key>.``.

    Returns
    -------
    newhdr: Header
        The updated (key-mapped) header.
    '''
    newhdr = header.copy()
    for k_new, k_old in keymap.items():
        if k_new == k_old:
            continue

        # if k_new already in the header, only deprecate k_old.
        # if not, copy k_old to k_new and deprecate k_old.
        if k_old is not None:
            if k_new in newhdr:
                if deprecation:
                    newhdr.comments[k_old] = f"Deprecated. See {k_new}"
            else:
                try:
                    comment_ori = newhdr.comments[k_old]
                    newhdr[k_new] = (newhdr[k_old], comment_ori)
                    if deprecation:
                        newhdr.comments[k_old] = f"Deprecated. See {k_new}"
                except KeyError:
                    pass

    return newhdr


def get_from_header(header, key, unit=None, verbose=True, default=0):
    ''' Get a variable from the header object.

    Parameters
    ----------
    header : astropy.Header
        The header to extract the value.

    key : str
        The header keyword to extract.

    unit : astropy unit
        The unit of the value.

    default : str, int, float, ..., or Quantity
        The default if not found from the header.

    Returns
    -------
    q: Quantity or any object
        The extracted quantity from the header. It's a Quantity if the unit is given. Otherwise,
        appropriate type will be assigned.
    '''
    # If using q = header.get(key, default=default), we cannot give any meaningful verboses infostr.
    # Anyway the ``header.get`` sourcecode contains only 4-line:
    # ``try: return header[key] // except (KeyError, IndexError): return default.
    key = key.upper()
    try:
        q = change_to_quantity(header[key], desired=unit)
        if verbose:
            print(f"header: {key:<8s} = {q}")
    except (KeyError, IndexError):
        q = change_to_quantity(default, desired=unit)
        warn(f"The key {key} not found in header: setting to {default}.")

    return q


def get_if_none(value, header, key, unit=None, verbose=True, default=0,
                to_value=False):
    ''' Similar to get_from_header, but a convenience wrapper.
    '''
    if value is None:
        value_Q = get_from_header(header, key, unit=unit, verbose=verbose, default=default)
        value_from = f"{key} in header"
    else:
        value_Q = change_to_quantity(value, unit, to_value=False)
        value_from = "the user"

    if to_value:
        return value_Q.value, value_from
    else:
        return value_Q, value_from


# TODO: do not load data extension if not explicitly ordered
def wcsremove(filepath=None, additional_keys=[], extension=0,
              output=None, verify='fix', overwrite=False, verbose=True,
              close=True):
    ''' Remove most WCS related keywords from the header.

    Paramters
    ---------
    additional_keys : list of regex str, optional
        Additional keys given by the user to be 'reset'. It must be in regex expression. Of course
        regex accepts just string, like 'NAXIS1'.

    output: str or Path
        The output file path.
    '''
    # Define header keywords to be deleted in regex:
    re2remove = ['CD[0-9]_[0-9]',  # Coordinate Description matrix
                 'CTYPE[0-9]',      # e.g., 'RA---TAN' and 'DEC--TAN'
                 'C[0-9]YPE[0-9]',  # FOCAS
                 'CUNIT[0-9]',      # e.g., 'deg'
                 'C[0-9]NIT[0-9]',  # FOCAS
                 'CRPIX[0-9]',      # The reference pixels in image coordinate
                 'C[0-9]PIX[0-9]',  # FOCAS
                 # The world cooordinate values at CRPIX[1, 2]
                 'CRVAL[0-9]',
                 'C[0-9]VAL[0-9]',  # FOCAS
                 'CDELT[0-9]',      # with CROTA, older version of CD matrix.
                 'C[0-9]ELT[0-9]',  # FOCAS
                 # The angle between image Y and world Y axes
                 'CROTA[0-9]',
                 'CRDELT[0-9]',
                 'CFINT[0-9]',
                 'RADE[C]?SYS*'     # RA/DEC system (frame)
                 'WCS-ORIG',        # FOCAS
                 'LTM[0-9]_[0-9]',
                 'LTV[0-9]*',
                 'PIXXMIT',
                 'PIXOFFST',
                 'WAT[0-9]_[0-9]',  # For TNX and ZPX, e.g., "WAT1_001"
                 'C0[0-9]_[0-9]',   # polynomial CD by imwcs
                 'PC[0-9]_[0-9]',
                 'P[A-Z]?[0-9]?[0-9][0-9][0-9][0-9][0-9][0-9]',  # FOCAS
                 'PV[0-9]_[0-9]',
                 '[A,B][P]?_[0-9]_[0-9]',  # astrometry.net
                 '[A,B][P]?_ORDER',   # astrometry.net
                 '[A,B][P]?_DMAX',    # astrometry.net
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

    hdul = fits.open(filepath)
    hdr = hdul[extension].header

    if verbose:
        print("Removed keywords: ", end='')

    for k in list(hdr.keys()):
        com = hdr.comments[k]
        deleted = False
        for re_i in re2remove:
            if re.match(re_i, k) is not None and not deleted:
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
        print(("\nFollowing keys may be related to WCS too:"
               + f"\n\t{candidate_key}"))

    hdul[extension].header = hdr

    if output is not None:
        hdul.writeto(output, output_verify=verify, overwrite=overwrite)

    if close:
        hdul.close()
        return
    else:
        return hdul


# def center_coord(header, skycoord=False):
#     ''' Gives the sky coordinate of the center of the image field of view.
#     Parameters
#     ----------
#     header: astropy.header.Header
#         The header to be used to extract WCS information (and image size)
#     skycoord: bool
#         Whether to return in the astropy.coordinates.SkyCoord object. If
#         `False`, a numpy array is returned.
#     '''
#     wcs = WCS(header)
#     cx = float(header['naxis1']) / 2 - 0.5
#     cy = float(header['naxis2']) / 2 - 0.5
#     center_coo = wcs.wcs_pix2world(cx, cy, 0)

#     if skycoord:
#         return SkyCoord(*center_coo, unit='deg')

#     return np.array(center_coo)


def convert_bit(fname, original_bit=12, target_bit=16):
    ''' Converts a FIT(S) file's bit.
    Note
    ----
    In ASI1600MM, for example, the output data is 12-bit but since FITS standard do not accept 12-bit
    (but the closest integer is 16-bit), so, for example, the pixel values can have 0 and 15, but not
    any integer between these two. So it is better to convert to 16-bit.
    '''
    hdul = fits.open(fname)
    dscale = 2**(target_bit - original_bit)
    hdul[0].data = (hdul[0].data / dscale).astype('int')
    hdul[0].header['MAXDATA'] = (2**original_bit - 1,
                                 "maximum valid physical value in raw data")
    # hdul[0].header['BITPIX'] = target_bit
    # FITS ``BITPIX`` cannot have, e.g., 12, so the above is redundant line.
    hdul[0].header['BUNIT'] = 'ADU'
    hdul.close()
    return hdul
