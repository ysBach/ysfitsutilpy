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

from .misc import _parse_extension, change_to_quantity, str_now, _parse_data_header, is_list_like

__all__ = ["add_to_header", "update_process",
           "wcs_crota", "center_radec", "calc_offset_wcs", "calc_offset_physical",
           "key_remover", "key_mapper",
           "get_from_header", "get_if_none", "wcsremove", "fov_radius",
           "convert_bit"]


def add_to_header(header, histcomm, s, precision=3, time_fmt="{:.>72s}", t_ref=None,
                  dt_fmt="(dt = {:.3f} s)", verbose=False, set_kw={'after': -1}):
    ''' Automatically add timestamp as well as HISTORY or COMMENT string

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

    time_fmt : str, None, optional.
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

    set_kw : dict, optional.
        The keyword arguments added to `Header.set()`. Default is ``{'after':-1}``, i.e., the history
        or comment will be appended to the very last part of the header.

    Note
    ----
    The timming benchmark shows that,
    %timeit add_to_header(hdu.header, 'comm', 'aadfaer sdf')
    310 µs +/- 2.93 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)

    %timeit add_to_header(hdu.header, 'comM', 'aadfaer sdf')
    309 µs +/- 2.48 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)

    %timeit add_to_header(hdu.header, 'comMent', 'aadfaer sdf')
    15.4 ms +/- 299 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    '''
    pall = locals()
    if histcomm.lower() in ['h', 'hist']:
        pall['histcomm'] = 'HISTORY'
        return add_to_header(**pall)
    elif histcomm.lower() in ['c', 'com', 'comm']:
        pall['histcomm'] = 'COMMENT'
        return add_to_header(**pall)
    # The "elif not in raise" gives large bottleneck if, e.g., ``histcomm="ComMent"``...
    # elif histcomm.lower() not in ['history', 'comment']:
    #     raise ValueError("Only HISTORY or COMMENT are supported now.")

    def _add_content(header, content):
        try:
            header.set(histcomm, content, **set_kw)
        except AttributeError:  # For a CCDData that has just initialized, header is in OrderdDict, not Header
            header[histcomm] = content

    if isinstance(s, str):
        s = [s]

    for _s in s:
        _add_content(header, _s)
        if verbose:
            print(f"{histcomm.upper():<8s} {_s}")

    if time_fmt is not None:
        timestr = str_now(precision=precision, fmt=time_fmt, t_ref=t_ref, dt_fmt=dt_fmt)
        _add_content(header, timestr)
        if verbose:
            print(f"{histcomm.upper():<8s} {timestr}")


def update_tlm(header):
    ''' Adds the IRAF-like ``FITS-TLM`` right after ``NAXISi``.
    '''
    now = Time(Time.now(), precision=0).isot
    try:
        del header["FITS-TLM"]
    except KeyError:
        pass
    try:
        header.set("FITS-TLM",
                   value=now,
                   comment="UT of last modification of this FITS file",
                   after=f"NAXIS{header['NAXIS']}")
    except AttributeError:  # If header is OrderedDict
        header["FITS-TLM"] = (now, "UT of last modification of this FITS file")


def update_process(header, process=None, key="PROCESS", delimiter='-', add_comment=True,
                   additional_comment=dict()):
    """ update the process history keyword in the header.
    Parameters
    ----------
    header : header
        The header to update the ``PROCESS`` (tunable by `key` parameter) keyword.

    process : str or list-like of str
        The additional process keys to add to the header.

    key : str, optional.
        The key for the process-related header keyword.

    delimiter : str, optional.
        The delimiter for each process. It can be null string (``''``). The best is to match it with
        the pre-existing delimiter of the ``header[key]``.

    additional_comment : dict, optional.
        The additional comment to add. For instance, ``dict(v="vertical pattern", f="fourier
        pattern")`` will add a new line of comment which reads "User added items for `key`:
        v=vertical pattern, f=fourier pattern."
    """
    if isinstance(process, str):
        process = [process]
    elif not is_list_like(process):
        raise TypeError("additional_process must be str or list-like.")
    else:
        process = list(process)

    haskey = key in header
    if haskey:
        process = [header[key]] + process
        # do not additionally add comment.
    elif add_comment:
        # add comment.
        add_to_header(
            header, 'c',
            f"Standard items for {key} includes B=bias, D=dark, F=flat, T=trim, W=WCS, C=CRrej, Fr=fringe."
        )

    header[key] = (delimiter.join(process), "Process (order: 1-2-3-...): see comment.")

    if additional_comment:
        addstr = [f"{k}={v}" for k, v in additional_comment.items()]
        addstr = ', '.join(addstr)
        add_to_header(header, 'c', f"User added items to {key}: {addstr}.")


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
        `ra_key` and `dec_key` keywords directly. If `False`, `ra_key` and `dec_key` from the
        header will be understood as the "center" and the RA, DEC of that location will be returned.

    equinox, frame : str, optional
        The `equinox` and `frame` for SkyCoord. Default (`None`) will use the default of
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


def calc_offset_wcs(target, reference, loc_target='center', loc_reference='center', order_xyz=True,
                    intify_offset=False):
    ''' The pixel offset of target's location when using WCS in referene.

    Parameters
    ----------

    target : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like, WCS
        The object to extract header to calculate the position

    reference : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like, WCS
        The object to extract reference WCS (or header to extract WCS) to calculate the position
        *from*.

    loc_target : str (center, origin) or ndarray optional.
        The location to calculate the position (in pixels and in xyz order). Default is ``'center'``
        (half of ``NAXISi`` keys in `target`). The `location`'s world coordinate is calculated from
        the WCS information in `target`. Then it will be transformed to the image coordinate of
        `reference`.

    loc_reference : str (center, origin) or ndarray optional.
        The location of the reference point (in pixels and in xyz order) in `reference`'s coordinate
        to calculate the offset.

    order_xyz : bool, optional.
        Whether to return the position in xyz order or not (python order: ``[::-1]`` of the former).
        Default is `True`.
    '''
    def _parse_loc(loc, obj):
        if isinstance(obj, WCS):
            w = obj
        else:
            _, hdr = _parse_data_header(obj, parse_data=False, copy=False)
            w = WCS(hdr)

        if loc == 'center':
            _loc = np.atleast_1d(w._naxis)/2
        elif loc == 'origin':
            _loc = [0.]*w.naxis
        else:
            _loc = np.atleast_1d(loc)

        return w, _loc

    w_targ, _loc_target = _parse_loc(loc_target, target)
    w_ref, _loc_ref = _parse_loc(loc_reference, reference)

    _loc_target_coo = w_targ.all_pix2world(*_loc_target, 0)
    _loc_target_pix_ref = w_ref.all_world2pix(*_loc_target_coo, 0)

    offset = _loc_target_pix_ref - _loc_ref

    if intify_offset:
        offset = np.around(offset).astype(int)

    if order_xyz:
        return offset
    else:
        return offset[::-1]


def calc_offset_physical(target, reference=None, order_xyz=True, ignore_ltm=True, intify_offset=False):
    ''' The pixel offset by physical-coordinate information in referene.

    Parameters
    ----------

    target : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like
        The object to extract header to calculate the position

    reference : CCDData, PrimaryHDU, ImageHDU, HDUList, Header, ndarray, number-like, path-like
        The reference to extract header to calculate the position *from*. If `None`, it is basically
        identical to extract the LTV values from `target`.
        Default is `None`.

    order_xyz : bool, optional.
        Whether to return the position in xyz order or not (python order: ``[::-1]`` of the former).
        Default is `True`.

    ignore_ltm : bool, optional.
        Whether to assuem the LTM matrix is identity. If it is not and ``ignore_ltm=False``, a
        `NotImplementedError` will be raised, i.e., non-identity LTM matrices are not supported.

    Notes
    -----
    Similar to `calc_offset_wcs`, but with locations fixed to origin (as non-identity LTM matrix is
    not supported). Also, input of WCS is not accepted because astropy's wcs module does not parse
    LTV/LTM from header.
    '''
    def _check_ltm(hdr):
        ndim = hdr["NAXIS"]
        for i in range(ndim):
            for j in range(ndim):
                try:
                    assert float(hdr["LTM{i}_{j}"]) == 1.0*(i == j)
                except (KeyError, IndexError):
                    continue
                except (AssertionError):
                    raise NotImplementedError("Non-identity LTM matrix is not supported.")

            try:  # Sometimes LTM matrix is saved as ``LTMi``, not ``LTMi_j``.
                assert float(target["LTM{i}"]) == 1.0
            except (KeyError, IndexError):
                continue
            except (AssertionError):
                raise NotImplementedError("Non-identity LTM matrix is not supported.")

    do_ref = reference is not None
    _, target = _parse_data_header(target, parse_data=False)
    if do_ref:
        _, reference = _parse_data_header(reference, parse_data=False)

    if not ignore_ltm:
        _check_ltm(target)
        if do_ref:
            _check_ltm(reference)

    ndim = target["NAXIS"]
    ltvs_obj = []
    for i in range(ndim):
        try:
            ltvs_obj.append(target[f"LTV{i + 1}"])
        except (KeyError, IndexError):
            ltvs_obj.append(0)

    if do_ref:
        ltvs_ref = []
        for i in range(ndim):
            try:
                ltvs_ref.append(reference[f"LTV{i + 1}"])
            except (KeyError, IndexError):
                ltvs_ref.append(0)
        offset = np.array(ltvs_obj) - np.array(ltvs_ref)
    else:
        offset = np.array(ltvs_obj)

    if intify_offset:
        offset = np.around(offset).astype(int)

    if order_xyz:
        return offset  # This is already xyz order!
    else:
        return offset[::-1]


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

    # TODO: Can't we just do ``return max(r1, r2).to(unit)``??? Why did I do this? I can't remember...
    # 2020-11-09 14:29:29 (KST: GMT+09:00) ysBach
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
        the keywords having the name specified in `remove_keys` will be removed. If not, only the
        first occurence of each key in `remove_keys` will be removed. It is more sensical to set it
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


def key_mapper(header, keymap=None, deprecation=False, remove=False):
    ''' Update the header to meed the standard (keymap).

    Parameters
    ----------
    header : Header
        The header to be modified

    keymap : dict
        The dictionary contains ``{<standard_key>:<original_key>}`` information. If it is `None`
        (default), the copied version of the header is returned without any change.

    deprecation : bool, optional
        Whether to change the original keywords' comments to contain deprecation warning. If `True`,
        the original keywords' comments will become ``DEPRECATED. See <standard_key>.``. It has no
        effect if ``remove=True``.
        Default is `False`.

    remove : bool, optional.
        Whether to remove the original keyword. `deprecation` is ignored if ``remove=True``.
        Default is `False`.

    Returns
    -------
    newhdr: Header
        The updated (key-mapped) header.

    Notes
    -----
    If the new keyword already exist in the given header, virtually nothing will happen. If
    ``deprecation=True``, the old one's comment will be changed, and if ``remove=True``, the old one
    will be removed; the new keyword will never be changed or overwritten.
    '''
    def _rm_or_dep(hdr, old, new):
        if remove:
            hdr.remove(old)
        elif deprecation:  # do not remove but deprecate
            hdr.comments[old] = f"DEPRECATED. See {new}"

    newhdr = header.copy()
    if keymap is not None:
        for k_new, k_old in keymap.items():
            if k_new == k_old:
                continue

            if k_old is not None:
                if k_new in newhdr:  # if k_new already in the header, JUST deprecate k_old.
                    _rm_or_dep(newhdr, k_old, k_new)
                else:  # if not, copy k_old to k_new and deprecate k_old.
                    try:
                        comment_ori = newhdr.comments[k_old]
                        newhdr[k_new] = (newhdr[k_old], comment_ori)
                        _rm_or_dep(newhdr, k_old, k_new)
                    except (KeyError, IndexError):
                        # don't even warn
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
    # Anyway the `header.get` sourcecode contains only 4-line:
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


def get_if_none(value, header, key, unit=None, verbose=True, default=0, to_value=False):
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
def wcsremove(path=None, additional_keys=[], extension=None, output=None, output_verify='fix', overwrite=False,
              verbose=True, close=True):
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

    hdul = fits.open(path)
    hdr = hdul[_parse_extension(extension)].header

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
        if len(candidate_key) != 0:
            print(f"\nFollowing keys may be related to WCS too:\n\t{candidate_key}")

    hdul[extension].header = hdr

    if output is not None:
        hdul.writeto(output, output_verify=output_verify, overwrite=overwrite)

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
