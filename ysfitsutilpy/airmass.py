'''
Airmass related funcitons.

Don't you think these must be implemented to astropy...?
'''
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io.fits import Card
from astropy.time import Time

from .hduutil import get_if_none

__all__ = ["calc_airmass", "airmass_obs", "airmass_from_hdr"]


def calc_airmass(zd_deg=None, cos_zd=None, scale=750.):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.

    Parameters
    ----------
    zd_deg : float, optional
        The zenithal distance in degrees

    cos_zd : float, optional
        The cosine of zenithal distance. If given, ``zd_deg`` is not used.

    scale : float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere. In IRAF documentation, it is mistakenly written that
        this ``scale`` is the "scale height".

    Notes
    -----
    Wiki:

        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere

    Identical to the airmass calculation for a given observational run of
    IRAF's asutil.setairmass:

        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass

    '''
    if not ((zd_deg is None) ^ (cos_zd is None)):
        raise ValueError("One and only one of `zd_deg` or `cos_zd` should be given.")

    # NOTE: for this part, using numexpr is slower than using numpy. It is
    # because this airmass calculation is usually done for a single zd value
    # from a single exposure. If one needs a large amount of such calculations,
    # numexpr may boost the speed, but that will be like few ms to us order.
    # Using numexpr will introduce a large overhead for the calculation if this
    # is repeated for thousands of images:
    # numexpr version:
    # %timeit yfu.calc_airmass(10)
    # 21.3 µs +/- 2.44 µs per loop (mean +/- std. dev. of 7 runs, 10000 loops each)
    # numpy version:
    # %timeit yfu.calc_airmass(10)
    # 3.65 µs +/- 93.5 ns per loop (mean +/- std. dev. of 7 runs, 100000 loops each)

    if cos_zd is None:
        cos_zd = np.cos(np.deg2rad(zd_deg))

    am = np.sqrt((scale*cos_zd)**2 + 2*scale + 1) - scale*cos_zd

    return am


def airmass_obs(targetcoord, obscoord, ut, exptime, scale=750., full=False):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.

    Parameters
    ----------
    targetcoord: astropy.SkyCoord
        The target's coorndinate.

    obscoord : astropy.EarthLocation
        The observer's location.

    ut : astropy.Time
        The time when the exposure is started.

    exptime : astropy.Quantity
        The exposure time.

    scale : float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere. In IRAF documentation, it is mistakenly written that
        this `scale` is the "scale height".

    Notes
    -----
    Wiki:

        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere

    Identical to the airmass calculation for a given observational run of
    IRAF's asutil.setairmass:

        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass

    Partly contributed by Geonwoo Kang (Seoul National University) in Apr 2018.
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

    alldict = {"alt": [], "az": [], "zd": [], "airmass": []}
    for t in [t_start, t_mid, t_final]:
        C_altaz = AltAz(obstime=t, location=obscoord)
        target = targetcoord.transform_to(C_altaz)
        alt = target.alt.to_string(unit=u.deg, sep=':')
        az = target.az.to_string(unit=u.deg, sep=':')
        zd = target.zen.to(u.deg).value
        am = calc_airmass(zd_deg=zd, scale=scale)
        alldict["alt"].append(alt)
        alldict["az"].append(az)
        alldict["zd"].append(zd)
        alldict["airmass"].append(am)

    am_eff = (alldict["airmass"][0] + 4*alldict["airmass"][1] + alldict["airmass"][2]) / 6

    if full:
        return am_eff, alldict

    return am_eff


# TODO: change key, unit, etc as input dict.
def airmass_from_hdr(
        header,
        ra=None,
        dec=None,
        ut=None,
        exptime=None,
        lon=None,
        lat=None,
        height=None,
        equinox=None,
        frame=None,
        scale=750.,
        ra_key="RA",
        dec_key="DEC",
        ut_key="DATE-OBS",
        lon_key="LONGITUD",
        lat_key="LATITUDE",
        height_key="HEIGHT",
        exptime_key="EXPTIME",
        equinox_key="EPOCH",
        frame_key="RADECSYS",
        ra_unit=u.hourangle,
        dec_unit=u.deg, exptime_unit=u.s,
        lon_unit=u.deg,
        lat_unit=u.deg,
        height_unit=u.m,
        ut_format='isot',
        ut_scale='utc',
        return_header=False,
        verbose=False
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
        The `equinox` and `frame` for SkyCoord.

    scale: float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere.

    XX_key: str, optional
        The header key to find XX if ``XX`` is `None`.

    XX_unit: Quantity, optional
        The unit of ``XX``

    ut_format, ut_scale: str, optional
        The `format` and `scale` for Time.

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
        amstr = (
            "ysfitsutilpy's airmass calculation uses the same algorithm as IRAF: From "
            + "'Some Factors Affecting the Accuracy of Stellar Photometry with CCDs' by "
            + "P. Stetson, DAO preprint, September 1988.")

        # At some times, hdr["AIRMASS"] = am, for example, did not work for
        # some reasons which I don't know.... So I used Card. -
        # YPBach 2018-05-04
        cs = [
            Card("AIRMASS", am_eff,
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
            Card("HISTORY", "AIRMASS calculated from ysfitsutilpy.")
        ]
        return cs

    ra = get_if_none(ra, header, ra_key, unit=ra_unit, verbose=verbose)[0]
    dec = get_if_none(dec, header, dec_key, unit=dec_unit, verbose=verbose)[0]
    exptime = get_if_none(exptime, header, exptime_key, unit=exptime_unit,
                          verbose=verbose)[0]
    lon = get_if_none(lon, header, lon_key, lon_unit)[0]
    lat = get_if_none(lat, header, lat_key, lat_unit)[0]
    height = get_if_none(height, header, height_key, height_unit)[0]
    equinox = get_if_none(equinox, header, equinox_key)[0]
    frame = get_if_none(frame, header, frame_key)[0]

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

    targetcoord = SkyCoord(
        ra=ra, dec=dec, unit=(ra_unit, dec_unit), frame=frame, equinox=equinox
    )

    try:  # It should work here but just in case I put except...
        observcoord = EarthLocation(lon=lon*lon_unit, lat=lat*lat_unit, height=height*height_unit)
    except ValueError:
        observcoord = EarthLocation(lon=lon, lat=lat, height=height)

    am_eff, alldict = airmass_obs(
        targetcoord=targetcoord,
        obscoord=observcoord,
        ut=ut,
        exptime=exptime*exptime_unit,
        scale=scale,
        full=True
    )

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
