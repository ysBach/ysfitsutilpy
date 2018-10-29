'''
Collection of functions that are rather header-dependent than the data.
'''
import re
from warnings import warn

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

from .mathutil import calc_airmass

__all__ = ["key_mapper", "get_from_header", "wcsremove", "center_coord",
           "airmass_obs", "airmass_hdr", "convert_bit"]


def key_mapper(header, keymap, deprecation=False):
    ''' Update the header to meed the standard (keymap).
    Parameters
    ----------
    header: Header
        The header to be modified
    keymap: dict
        The dictionary contains ``{<standard_key>:<original_key>}`` information
    deprecation: bool, optional
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

    def _change_to_quantity(x, desired=None):
        ''' Change the non-Quantity object to astropy Quantity.
        Parameters
        ----------
        x: object changable to astropy Quantity
            The input to be changed to a Quantity. If a Quantity is given,
            ``x`` is changed to the ``desired``, i.e., ``x.to(desired)``.
        desired: astropy Unit, optional
            The desired unit for ``x``.

        Returns
        -------
        ux: Quantity

        Note
        ----
        If Quantity, transform to ``desired``. If ``desired = None``, return
        it as is. If not Quantity, multiply the ``desired``.
        ``desired = None``, return ``x`` with dimensionless unscaled unit.
        '''
        if not isinstance(x, u.quantity.Quantity):
            if desired is None:
                ux = x * u.dimensionless_unscaled
            else:
                ux = x * desired
        else:
            if desired is None:
                ux = x
            else:
                ux = x.to(desired)
        return ux

    q = None

    try:
        q = header[key]
        if unit is not None:
            q = q * unit
        if verbose:
            print(f"header: {key} = {q}")
    except KeyError:
        if default is not None:
            q = _change_to_quantity(default, desired=unit)
            warn(f"{key} not found in header: setting to {default}.")

    return q


def wcsremove(filepath=None, additional_keys=[], extension=0,
              output=None, verify='fix', overwrite=False, verbose=True):
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

    hdul = fits.open(filepath)
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
        print(("\nFollowing keys may be related to WCS too:"
               + f"\n\t{candidate_key}"))

    hdul[extension].header = hdr

    if output is not None:
        hdul.writeto(output, output_verify=verify, overwrite=overwrite)

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


def airmass_obs(targetcoord, obscoord, ut, exptime, scale=750., full=False):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.
    Note
    ----
    Wiki:
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere
    Identical to the airmass calculation for a given observational run of
    IRAF's asutil.setairmass:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass
    Partly contributed by Kunwoo Kang (Seoul National University) in Apr 2018.

    '''
    if not isinstance(ut, Time):
        warn("ut is not Time object. Assume format='isot', scale='utc'.")
        ut = Time(ut, format='isot', scale='utc')
    if not isinstance(exptime, u.Quantity):
        warn("exptime is not astropy Quantity. Assume it is in seconds.")
        exptime = exptime * u.s

    t_start = ut
    t_mid = ut + exptime / 2
    t_final = ut + exptime

    altaz = {"alt": [], "az": [], "zd": [], "airmass": []}
    for t in [t_start, t_mid, t_final]:
        C_altaz = AltAz(obstime=t, location=obscoord)
        target = targetcoord.transform_to(C_altaz)
        alt = target.alt.to_string(unit=u.deg, sep=':')
        az = target.az.to_string(unit=u.deg, sep=':')
        zd = target.zen.to(u.deg).value
        am = calc_airmass(zd_deg=zd, scale=scale)
        altaz["alt"].append(alt)
        altaz["az"].append(az)
        altaz["zd"].append(zd)
        altaz["airmass"].append(am)

    am_simpson = (altaz["airmass"][0]
                  + 4 * altaz["airmass"][1]
                  + altaz["airmass"][2]) / 6

    if full:
        return am_simpson, altaz

    return am_simpson


# TODO: change key, unit, etc as input dict.
def airmass_hdr(header, ra=None, dec=None, ut=None, exptime=None,
                lon=None, lat=None, height=None, equinox=None, frame=None,
                scale=750.,
                ra_key="RA", dec_key="DEC", ut_key="DATE-OBS",
                exptime_key="EXPTIME", lon_key="LONGITUD", lat_key="LATITUDE",
                height_key="HEIGHT", equinox_key="EPOCH", frame_key="RADECSYS",
                ra_unit=u.hourangle, dec_unit=u.deg,
                exptime_unit=u.s, lon_unit=u.deg, lat_unit=u.deg,
                height_unit=u.m,
                ut_format='isot', ut_scale='utc',
                full=False
                ):
    ''' Calculate airmass using the header.
    Parameters
    ----------
    ra, dec: float or Quantity, optional
        The RA and DEC of the target. If not specified, it tries to find them
        in the header using ``ra_key`` and ``dec_key``.

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
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere.

    XX_key: str, optional
        The header key to find XX if ``XX`` is ``None``.

    XX_unit: Quantity, optional
        The unit of ``XX``

    ut_format, ut_scale: str, optional
        The ``format`` and ``scale`` for Time.

    full: bool, optional
        Whether to return the full calculated results. If ``False``, it returns
        the averaged (Simpson's 1/3-rule calculated) airmass only.
    '''
    # If there is no header keyword matches the ``key``, it should give
    # KeyError. For such reason, I didn't use ``get_from_header`` here.
    def _conversion(header, val, key, unit=None, instance=None):
        if val is None:
            val = header[key]
        elif (instance is not None) and (unit is not None):
            if isinstance(val, instance):
                val = val.to(unit).value

        return val

    ra = _conversion(header, ra, ra_key, ra_unit, u.Quantity)
    dec = _conversion(header, dec, dec_key, dec_unit, u.Quantity)
    exptime = _conversion(header, exptime, exptime_key,
                          exptime_unit, u.Quantity)
    lon = _conversion(header, lon, lon_key, lon_unit, u.Quantity)
    lat = _conversion(header, lat, lat_key, lat_unit, u.Quantity)
    height = _conversion(header, height, height_key, height_unit, u.Quantity)
    equinox = _conversion(header, equinox, equinox_key)
    frame = _conversion(header, frame, frame_key)

    if ut is None:
        ut = header[ut_key]
    elif isinstance(ut, Time):
        ut = ut.isot
        # ut_format = 'isot'
        # ut_scale = 'utc'

    targetcoord = SkyCoord(ra=ra, dec=dec, unit=(ra_unit, dec_unit),
                           frame=frame, equinox=equinox)

    try:
        observcoord = EarthLocation(lon=lon * lon_unit, lat=lat * lat_unit,
                                    height=height * height_unit)
    except ValueError:
        observcoord = EarthLocation(lon=lon, lat=lat,
                                    height=height * height_unit)

    result = airmass_obs(targetcoord=targetcoord, obscoord=observcoord, ut=ut,
                         exptime=exptime, scale=scale, full=full)

    return result


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

