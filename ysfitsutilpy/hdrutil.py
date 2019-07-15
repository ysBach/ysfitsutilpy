'''
Collection of functions that are rather header-dependent than the data.
'''
import re
from warnings import warn

import numpy as np
from astropy import units as u
from astropy import wcs as astropywcs
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.io.fits import Card
from astropy.time import Time
from astropy.wcs import WCS

from .misc import airmass_obs

__all__ = ["wcs_crota", "center_radec", "key_remover", "key_mapper",
           "get_from_header", "wcsremove", "fov_radius",
           "airmass_from_hdr", "convert_bit"]


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
        raise TypeError("wcs type not understood. It must be either "
                        + "astropy.wcs.wcs.WCS or astropy.wcs.Wcsprm")

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
    Even though RA or DEC is in sexagesimal, e.g., "20 53 20", astropy
    correctly reads it in such a form, so no worries.

    Parameters
    ----------
    header : Header
        The header to extract the central RA/DEC from keywords or WCS.

    center_of_image : bool, optional
        If ``True``, WCS information will be extracted from the header,
        rather than relying on the ``ra_key`` and ``dec_key`` keywords
        directly. If ``False``, ``ra_key`` and ``dec_key`` from the header
        will be understood as the "center" and the RA, DEC of that location
        will be returned.

    equinox, frame : str, optional
        The ``equinox`` and ``frame`` for SkyCoord. Default (``None``) will
        use the default of SkyCoord. Important only if ``usewcs=False``.

    XX_key : str, optional
        The header key to find XX if ``XX`` is ``None``. Important only if
        ``usewcs=False``.

    XX_unit : Quantity, optional
        The unit of ``XX``. Important only if ``usewcs=False``.

    mode : 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core WCS transformation (``'wcs'``). Important
        only if ``usewcs=True``.

    plain : bool
        If ``True``, only the values of RA/DEC in degrees will be returned.
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
            equinox = get_from_header(header, equinox_key,
                                      verbose=verbose, default=None)
        if frame is None:
            frame = get_from_header(header, frame_key,
                                    verbose=verbose, default=None)
            frame = frame.lower()
        coo = SkyCoord(ra=ra, dec=dec, unit=(ra_unit, dec_unit),
                       frame=frame, equinox=equinox)

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
    header: Header
        The header to be modified
    remove_keys: list of str
        The header keywords to be removed.
    deepremove: True, optional
        FITS standard does not have any specification of duplication of
        keywords as discussed in the following issue:
        https://github.com/astropy/ccdproc/issues/464
        If it is set to ``True``, ALL the keywords having the name specified
        in ``remove_keys`` will be removed. If not, only the first occurence
        of each key in ``remove_keys`` will be removed. It is more sensical to
        set it ``True`` in most of the cases.
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
        Whether to change the original keywords' comments to contain
        deprecation warning. If ``True``, the original keywords' comments will
        become ``Deprecated. See <standard_key>.``.

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

    Returns
    -------
    q: Quantity or any object
        The extracted quantity from the header. It's a Quantity if the unit is
        given. Otherwise, appropriate type will be assigned.
    '''

    def _change_to_quantity(x, unit=None):
        ''' Change the non-Quantity object to astropy Quantity.
        Parameters
        ----------
        x: object
            The input to be changed to a Quantity. If a Quantity is given,
            ``x`` is changed to the ``unit``, i.e., ``x.to(unit)``.
        unit: astropy Unit, optional
            The desired unit for ``x``.

        Returns
        -------
        ux: Quantity

        Note
        ----
        If Quantity, transform to ``unit``. If ``unit = None``, return
        it as is. If not Quantity, multiply the ``unit``.
        ``unit = None``, return ``x`` with dimensionless unscaled unit.
        '''
        if unit is None:
            ux = x  # If it were Quantity, original Quantity will be returned
        else:
            if isinstance(x, u.quantity.Quantity):
                ux = x.to(unit)
            else:
                ux = x * unit
        return ux

    q = None

    try:
        q = _change_to_quantity(header[key], unit=unit)
        if verbose:
            print(f"header: {key} = {q}")
    except KeyError:
        if default is not None:
            q = _change_to_quantity(default, unit=unit)
            warn(f"{key} not found in header: setting to {default}.")
        # else: None will be returned

    return q


# TODO: do not load data extension if not explicitly ordered
def wcsremove(filepath=None, additional_keys=[], extension=0,
              output=None, verify='fix', overwrite=False, verbose=True):
    ''' Remove most WCS related keywords from the header.
    Paramters
    ---------
    additional_keys : list of regex str, optional
        Additional keys given by the user to be 'reset'. It must be in regex
        expression. Of course regex accepts just string, like 'NAXIS1'.

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
                 '[A,B]_[0-9]_[0-9]',  # astrometry.net
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

    return hdul


# def center_coord(header, skycoord=False):
#     ''' Gives the sky coordinate of the center of the image field of view.
#     Parameters
#     ----------
#     header: astropy.header.Header
#         The header to be used to extract WCS information (and image size)
#     skycoord: bool
#         Whether to return in the astropy.coordinates.SkyCoord object. If
#         ``False``, a numpy array is returned.
#     '''
#     wcs = WCS(header)
#     cx = float(header['naxis1']) / 2 - 0.5
#     cy = float(header['naxis2']) / 2 - 0.5
#     center_coo = wcs.wcs_pix2world(cx, cy, 0)

#     if skycoord:
#         return SkyCoord(*center_coo, unit='deg')

#     return np.array(center_coo)


# TODO: change key, unit, etc as input dict.
def airmass_from_hdr(header, ra=None, dec=None, ut=None, exptime=None,
                     lon=None, lat=None, height=None, equinox=None, frame=None,
                     scale=750.,
                     ra_key="RA", dec_key="DEC", ut_key="DATE-OBS",
                     exptime_key="EXPTIME", lon_key="LONGITUD",
                     lat_key="LATITUDE", height_key="HEIGHT",
                     equinox_key="EPOCH", frame_key="RADECSYS",
                     ra_unit=u.hourangle, dec_unit=u.deg,
                     exptime_unit=u.s, lon_unit=u.deg, lat_unit=u.deg,
                     height_unit=u.m,
                     ut_format='isot', ut_scale='utc',
                     return_header=False
                     ):
    ''' Calculate airmass using the header.
    Parameters
    ----------
    ra, dec: float or Quantity, optional
        The RA and DEC of the target. If not specified, it tries to find
        them in the header using ``ra_key`` and ``dec_key``.

    ut: str or Time, optional
        The *starting* time of the observation in UT.

    exptime: float or Time, optional
        The exposure time.

    lon, lat, height: str, float, or Quantity
        The longitude, latitude, and height of the observatory. See
        astropy.coordinates.EarthLocation.

    equinox, frame: str, optional
        The ``equinox`` and ``frame`` for SkyCoord.

    scale: float, optional
        Earth radius divided by the atmospheric height (usually scale
        height) of the atmosphere.

    XX_key: str, optional
        The header key to find XX if ``XX`` is ``None``.

    XX_unit: Quantity, optional
        The unit of ``XX``

    ut_format, ut_scale: str, optional
        The ``format`` and ``scale`` for Time.

    return_header: bool, optional
        Whether to return the updated header.
    '''
    # If there is no header keyword matches the ``key``, it should give
    # KeyError. For such reason, I didn't use ``get_from_header`` here.
    def _conversion(header, val, key, unit=None):
        if val is None:
            val = header[key]  # assume it is in the unit of ``unit``.
        elif unit is not None:
            if isinstance(val, u.Quantity):
                val = val.to(unit).value
            # else: just return ``val``.
        return val

    def _cards_airmass(am_eff, alldict):
        ''' Gives airmass and alt-az related header cards.
        '''
        amstr = ("ysfitsutilpy's airmass calculation uses the same algorithm "
                 + "as IRAF: From 'Some Factors Affecting the Accuracy of "
                 + "Stellar Photometry with CCDs' by P. Stetson, DAO preprint,"
                 + " September 1988.")

        # At some times, hdr["AIRMASS"] = am, for example, did not work
        # for some reasons which I don't know.... So I used Card. -
        # YPBach 2018-05-04
        cs = [Card("AIRMASS", am_eff,
                   "Effective airmass (Stetson 1988; see COMMENT)"),
              Card("ZD", alldict["zd"][0],
                   "[deg] Zenithal distance (start of the exposure)"),
              Card("ALT", alldict["alt"][0],
                   "Altitude (start of the exposure)"),
              Card("AZ", alldict["az"][0],
                   "Azimuth (start of the exposure)"),
              Card("ALT_MID", alldict["alt"][1],
                   "Altitude (midpoint of the exposure)"),
              Card("AZ_MID", alldict["az"][1],
                   "Azimuth (midpoint of the exposure)"),
              Card("ZD_MID", alldict["zd"][1],
                   "[deg] Zenithal distance (midpoint of the exposure)"),
              Card("ALT_END", alldict["alt"][2],
                   "Altitude (end of the exposure)"),
              Card("AZ_END", alldict["az"][2],
                   "Azimuth (end of the exposure)"),
              Card("ZD_END", alldict["zd"][2],
                   "[deg] Zenithal distance (end of the exposure)"),
              Card("COMMENT", amstr),
              Card("HISTORY", "ALT-AZ calculated from ysfitsutilpy."),
              Card("HISTORY", "AIRMASS calculated from ysfitsutilpy.")]
        return cs

    ra = _conversion(header, ra, ra_key, ra_unit)
    dec = _conversion(header, dec, dec_key, dec_unit)
    exptime = _conversion(header, exptime, exptime_key, exptime_unit)
    lon = _conversion(header, lon, lon_key, lon_unit)
    lat = _conversion(header, lat, lat_key, lat_unit)
    height = _conversion(header, height, height_key, height_unit)
    equinox = _conversion(header, equinox, equinox_key)
    frame = _conversion(header, frame, frame_key)

    if ut is None:
        ut = header[ut_key]
    elif isinstance(ut, Time):
        ut = ut.isot
        # ut_format = 'isot'
        # ut_scale = 'utc'

    targetcoord = SkyCoord(ra=ra,
                           dec=dec,
                           unit=(ra_unit, dec_unit),
                           frame=frame,
                           equinox=equinox)

    try:  # It should work here but just in case I put except...
        observcoord = EarthLocation(lon=lon * lon_unit,
                                    lat=lat * lat_unit,
                                    height=height * height_unit)
    except ValueError:
        observcoord = EarthLocation(lon=lon,
                                    lat=lat,
                                    height=height)

    am_eff, alldict = airmass_obs(targetcoord=targetcoord,
                                  obscoord=observcoord,
                                  ut=ut,
                                  exptime=exptime * exptime_unit,
                                  scale=scale,
                                  full=True)

    if return_header:
        nhdr = header.copy()
        cards = _cards_airmass(am_eff, alldict)
        # Remove if there is, e.g., AIRMASS a priori to the original header:
        for c in cards:
            try:
                nhdr.remove(c.keyword)
            except KeyError:
                continue
        nhdr = nhdr + cards
        return nhdr

    else:
        return am_eff, alldict


def convert_bit(fname, original_bit=12, target_bit=16):
    ''' Converts a FIT(S) file's bit.
    Note
    ----
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
