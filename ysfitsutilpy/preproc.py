from warnings import warn

import astroscrappy
import ccdproc
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astroscrappy import detect_cosmics
from ccdproc import flat_correct, subtract_bias, subtract_dark

from .ccdutil import (CCDData_astype, errormap, propagate_ccdmask,
                      set_ccd_gain_rdnoise, trim_ccd)
from .hdrutil import add_to_header, update_process, update_tlm
from .misc import (LACOSMIC_KEYS, _parse_data_header, _parse_image,
                   change_to_quantity, fitsxy2py, load_ccd)

__all__ = [
    "crrej", "medfilt_bpm", "bdf_process"]

# Set strings for header history & print (if verbose)
str_bias = "Bias subtracted (see BIASPATH)"
str_dark = "Dark subtracted (see DARKPATH)"
str_dscale = "Dark scaling using {}"
str_flat = "Flat corrected by image/flat*flat_norm_value (see FLATPATH; FLATNORM)"
str_fringe_noscale = "Fringe subtracted (see FRINPATH)"
str_fringe_scale = ("Finge subtracted with scaling (image - scale*fringe)"
                    + "(see FRINPATH, FRINSECT, FRINFUNC and FRINSCAL)")
str_trim = "Trim by FITS section {} (see LTV, LTM, TRIMIM)"
str_e0 = "Readnoise propagated with Poisson noise (using gain above) of source."
str_ed = "Poisson noise from subtracted dark was propagated."
str_ef = "Flat uncertainty was propagated."
str_nexp = "Normalized by the exposure time."
str_navg = "Normalized by the average value of the frame."
str_nmed = "Normalized by the median value of the frame."
str_cr = ("Cosmic-Ray rejected by astroscrappy (v {}), with parameters: {}")


# def _add_and_print(s, header, verbose, update_header=True, t_ref=None):
#     if update_header:
#         # add as history
#         add_to_header(header, 'h', s, t_ref=t_ref)
#     if verbose:
#         if isinstance(s, str):
#             print(str_now(fmt='{}'), s)
#         else:
#             for _s in s:
#                 print(str_now(fmt='{}'), _s)


# # TODO: This is quite much overlapping with set_ccd_gain_rdnoise...
# def get_gain_readnoise(ccd, gain=None, gain_key="GAIN",
#                        gain_unit=u.electron/u.adu,
#                        rdnoise=None, rdnoise_key="RDNOISE",
#                        rdnoise_unit=u.electron, verbose=True,
#                        update_header=True):
#     ''' Get gain and readnoise from given paramters.
#     gain, rdnoise : None, float, astropy.Quantity, optional.
#         The gain and readnoise value. If ``gain`` or ``readnoise`` is
#         specified, they are interpreted with ``gain_unit`` and
#         ``rdnoise_unit``, respectively. If they are not specified, this
#         function will seek for the header with keywords of ``gain_key``
#         and ``rdnoise_key``, and interprete the header value in the unit
#         of ``gain_unit`` and ``rdnoise_unit``, respectively.

#     gain_key, rdnoise_key : str, optional.
#         See ``gain``, ``rdnoise`` explanation above.

#     gain_unit, rdnoise_unit : str, astropy.Unit, optional.
#         See ``gain``, ``rdnoise`` explanation above.

#     verbose : bool, optional.
#         The verbose option.

#     update_header : bool, optional
#         Whether to update the given header.

#     Note
#     ----
#     If gain and readout noise are not found properly, the default values
#     of 1.0 and 0.0 with corresponding units will be returned.
#     '''
#     gain_Q, gain_from = get_if_none(
#         gain, ccd.header, key=gain_key, unit=gain_unit,
#         verbose=verbose, default=1.0)
#     rdnoise_Q, rdnoise_from = get_if_none(
#         rdnoise, ccd.header, key=rdnoise_key, unit=rdnoise_unit,
#         verbose=False, default=0.0)

#     _add_and_print(str_grd.format(gain_from,
#                                   "gain",
#                                   gain_Q.value,
#                                   gain_Q.unit),
#                    ccd.header, verbose, update_header=update_header)
#     _add_and_print(str_grd.format(rdnoise_from,
#                                   "rdnoise",
#                                   rdnoise_Q.value,
#                                   rdnoise_Q.unit),
#                    ccd.header, verbose, update_header=update_header)
#     return gain_Q, rdnoise_Q


def crrej(ccd, mask=None, propagate_crmask=False, update_header=True,
          add_process=True, gain=None, rdnoise=None, verbose=True,
          sigclip=4.5, sigfrac=0.5, objlim=1.0, satlevel=np.inf,
          pssl=0.0, niter=4, sepmed=False, cleantype='medmask',
          fsmode='median', psfmodel='gauss', psffwhm=2.5, psfsize=7,
          psfk=None, psfbeta=4.765):
    """ Do cosmic-ray rejection using L.A.Cosmic default parameters.
    Parameters
    ----------
    ccd : CCDData
        The ccd to be processed. The data must be in ADU, not electrons.

    propagate_crmask : bool, optional.
        Whether to save (propagate) the mask from CR rejection (``astroscrappy``) to the CCD's mask.
        Default is `False`.

    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. If not ``Quantity``, they must be in electrons per adu and
        electron unit, respectively.

    sigclip : float, optional
        Laplacian-to-noise limit for cosmic ray detection. Lower values will flag more pixels as cosmic
        rays.
        Default: 4.5.

    sigfrac : float, optional
        Fractional detection limit for neighboring pixels. For cosmic ray neighbor pixels, a
        lapacian-to-noise detection limit of sigfrac * sigclip will be used.
        Default: 0.5.

    objlim : float, optional
        Minimum contrast between Laplacian image and the fine structure image. Increase this value if
        cores of bright stars are flagged as cosmic rays.
        Default: 1.0.

    pssl : float, optional
        Previously subtracted sky level in ADU. We always need to work in electrons for cosmic ray
        detection, so we need to know the sky level that has been subtracted so we can add it back in.
        Default: 0.0.

    satlevel : float, optional
        Saturation of level of the image (electrons). This value is used to detect saturated stars and
        pixels at or above this level are added to the mask.
        Default: ``np.inf``.

    niter : int, optional
        Number of iterations of the LA Cosmic algorithm to perform.
        Default: 4.

    sepmed : boolean, optional
        Use the separable median filter instead of the full median filter. The separable median is not
        identical to the full median filter, but they are approximately the same and the separable
        median filter is significantly faster and still detects cosmic rays well.
        Default: `True`

    cleantype : {'median', 'medmask', 'meanmask', 'idw'}, optional
        Set which clean algorithm is used:

        * ``'median'``: An umasked 5x5 median filter
        * ``'medmask'``: A masked 5x5 median filter
        * ``'meanmask'``: A masked 5x5 mean filter
        * ``'idw'``: A masked 5x5 inverse distance weighted interpolation

        Default: ``"meanmask"``.

    fsmode : {'median', 'convolve'}, optional
        Method to build the fine structure image:

          * ``'median'``: Use the median filter in the standard LA Cosmic algorithm
          * ``'convolve'``: Convolve the image with the psf kernel

        to calculate the fine structure image.
        Default: ``'median'``.

    psfmodel : {'gauss', 'gaussx', 'gaussy', 'moffat'}, optional
        Model to use to generate the psf kernel if ``fsmode='convolve'`` and ``psfk`` is `None`. The
        current choices are Gaussian and Moffat profiles. ``'gauss'`` and ``'moffat'`` produce circular
        PSF kernels. The ``'gaussx'`` and ``'gaussy'`` produce Gaussian kernels in the x and y
        directions respectively.
        Default: ``"gauss"``.

    psffwhm : float, optional
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.

    psfsize : int, optional
        Size of the kernel to calculate. Returned kernel will have size psfsize x psfsize. psfsize
        should be odd.
        Default: 7.

    psfk : float numpy array, optional
        PSF kernel array to use for the fine structure image if ``fsmode='convolve'``. If `None` and
        ``fsmode == 'convolve'``, we calculate the psf kernel using ``'psfmodel'``.
        Default: `None`.

    psfbeta : float, optional
        Moffat beta parameter. Only used if ``fsmode='convolve'`` and ``psfmodel='moffat'``.
        Default: 4.765.

    verbose : boolean, optional
        Print to the screen or not. Default: `False`.

    Returns
    -------
    _ccd : CCDData
        The cosmic-ray cleaned CCDData in ADU. ``astroscrappy`` automatically does a gain correction,
        so I divided the ``astroscrappy`` result by gain to restore to ADU (not to surprise the users).

    crmask : ndarray (mask)
        The cosmic-ray mask from ``astroscrappy``, propagated by the original mask of the ccd (if
        ``ccd.mask`` is not `None`)and ``mask`` given by the user.

    update_header : bool, optional.
        Whether to update the header if there is any.

    add_process : bool, optional.
        Whether to add ``PROCESS`` key to the header.

    Notes
    -----
    All defaults are based on IRAF version of L.A. Cosmic (Note the default parameters of L.A. Cosmic
    differ from version to version, so I took the IRAF version written by van Dokkum.)
    See the docstring of astroscrappy by
    >>> import astroscrappy
    >>> astroscrappy.detect_cosmics?

    Example
    -------
    >>> yfu.ccdutil.set_ccd_gain_rdnoise(ccd)
    >>> nccd, mask = crrej(ccd)
    """
    _t = Time.now()

    if gain is None:
        try:
            gain = ccd.gain
        except AttributeError:
            raise ValueError(
                "Gain must be given or accessible as ``ccd.gain``. Use, e.g., yfu.set_ccd_gain_rdnoise(ccd)."
            )

    if rdnoise is None:
        try:
            rdnoise = ccd.rdnoise
        except AttributeError:
            raise ValueError(
                "Readnoise must be given or accessible as ``ccd.rdnoise``. "
                + "Use, e.g., yfu.set_ccd_gain_rdnoise(ccd)."
            )

    _ccd = ccd.copy()
    data, hdr = _parse_data_header(_ccd)
    inmask = propagate_ccdmask(_ccd, additional_mask=mask)

    # The L.A. Cosmic accepts only the gain in e/adu and rdnoise in e.
    gain = change_to_quantity(gain, u.electron/u.adu, to_value=True)
    rdnoise = change_to_quantity(rdnoise, u.electron, to_value=True)

    # remove the fucxing cosmic rays
    crrej_kwargs = dict(
        gain=gain,
        readnoise=rdnoise,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        satlevel=satlevel,
        pssl=pssl,
        niter=niter,
        sepmed=sepmed,
        cleantype=cleantype,
        fsmode=fsmode,
        psfmodel=psfmodel,
        psffwhm=psffwhm,
        psfsize=psfsize,
        psfk=psfk,
        psfbeta=psfbeta
    )
    crmask, cleanarr = detect_cosmics(
        data,
        inmask=inmask,
        verbose=verbose,
        **crrej_kwargs
    )

    # create the new ccd data object
    #   astroscrappy automatically does the gain correction, so return
    #   back to avoid confusion.
    _ccd.data = cleanarr / gain
    if propagate_crmask:
        _ccd.mask = propagate_ccdmask(_ccd, additional_mask=crmask)

    if add_process and hdr is not None:
        try:
            hdr["PROCESS"] += "C"
        except KeyError:
            hdr["PROCESS"] = "C"

    if update_header and hdr is not None:
        add_to_header(
            hdr, 'h', s=str_cr.format(astroscrappy.__version__, crrej_kwargs), verbose=verbose, t_ref=_t
        )
    else:
        if verbose:
            print(str_cr.format(astroscrappy.__version__, crrej_kwargs))

    update_tlm(hdr)
    _ccd.header = hdr

    return _ccd, crmask


# TODO: put niter
# TODO: put medfilt_min
#   to get std at each pixel by medfilt[<medfilt_min] = 0, and std = sqrt((1+snoise)*medfilt/gain +
#   rdn**2)
def medfilt_bpm(ccd, cadd=1.e-10, std_model="std", gain=1., rdnoise=0., snoise=0.,
                sigclip_kw=dict(sigma=3., maxiters=5, std_ddof=1), std_section=None,
                size=5, mode='reflect', cval=0.0, origin=0, med_sub_clip=None, med_rat_clip=[0.5, 2],
                std_rat_clip=[-5, 5], dtype='float32', update_header=True,
                verbose=False, logical='and', full=False):
    ''' Find bad pixels from median filtering technique (non standard..?)
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The CCD to find the bad pixels.

    cadd : float, optional.
        A very small const to be added to the input array to avoid resulting value of 0.0 in the median
        filtered image which raises zero-division in median ratio (image/|median_filtered|).

    model : str, optional.
        The model used to calculate the std (standard deviation) map.

          * ``'std'``: Simple standard deviation is calculated.
          * ``'ccd'``: Using CCD noise model (``sqrt{(1 + snoise)*med_filt/gain + (rdnoise/gain)**2}``)

        For ``'std'``, the arguments ``std_section`` and ``sigclip_kw`` are used, while if ``'ccd'``,
        arguments ``gain``, ``rdnoise``, ``snoise`` will be used.

    sigclip_kw : dict, optional.
        The paramters used for `astropy.stats.sigma_clipped_stats` when estimating the sky standard
        deviation at ``std_section``. This is **ignored** if ``std_model='ccd'``.

    std_section : str, optinal.
        The region in FITS standard (1-indexing, end-inclusive, xyz order) to estimate the sky standard
        deviation to obtain the ``std_ratio``. If `None` (default), the full region of the given array
        is used, which is many times not desirable due to the celestial objects in the FOV and
        computational cost. This is **ignored** if ``std_model='ccd'``.

    size, mode, cval, origin : optional.
        The parameters to obtain the median-filtered map. See `scipy.ndimage.median_filter`.

    med_sub_clip : list of two float or `None`, optional.
        The thresholds to find bad pixel by ``med_sub = ccd.data - median_filter(ccd.data)``. The
        clipping will be turned off if it is `None` (default). If a list, must be in the order of
        ``[lower, upper]`` and at most two of these can be `None`.

    med_rat_clip : list of two float or `None`, optional.
        The thresholds to find bad pixel by ``med_ratio = ccd.data/np.abs(median_filter(ccd.data))``.
        The clipping will be turned off if it is `None` (default). If a list, must be in the order of
        ``[lower, upper]`` and at most two of these can be `None`.

    std_rat_clip : list of two float or `None`, optional.
        The thresholds to find bad pixel by ``std_ratio = (ccd - median_filter(ccd))/std``. The
        clipping will be turned off if it is `None` (default). If a list, must be in the order of
        ``[lower, upper]`` and at most two of these can be `None`.

    logical : str of ``['and', '&', 'or', '|']`` or list, optional.
        The logic to propagate masks determined by the ``_clip``'s. The mask is propagated such as
        ``posmask = med_sub > med_sub_clip[1] &/| med_ratio > med_rat_clip[1] &/| std_ratio >
        std_rat_clip[1]``. If a list, it must contain two str of these, in the order of
        ``[logical_negmask, logical_posmask]``.

    Returns
    -------
    ccd : CCDData
        The badpixel removed result.

    The followings are returned as ``dict`` only if ``full=True``.

    posmask, negmask : ndarry of bool
        The masked pixels by positive/negative criteria.

    sky_std : float
        The (sigma-clipped) sky standard deviation. Returned only if ``full=True``.

    Notes
    -----
    ``med_sub_clips`` is usually not necessary but useful to detect hot pixels in dark frames (no
    light) for some special circumstances. ::

      1. Median additive difference (data-medfilt) generated,
      2. Median ratio (data/|medfilt|) generated,
      3. Stddev ratio ((data-medfilt)/std) generated,
      4. posmask and negmask calculated by clips MB_[ADD/RAT/STD]_[U/L] and logic MB_[N/P]LOG (see keywords),
      5. Pixels of (posmask | negmask) are repleced with median filtered frame.

    '''
    from scipy.ndimage import median_filter

    def _sanitize_clips(clips):
        clips = np.atleast_1d(clips)
        if clips.size == 1:
            clips = np.repeat(clips, 2)
        return clips

    if ((med_sub_clip is None) and (med_rat_clip is None) and (std_rat_clip is None)):
        warn("No BPM is found because all clips are None.", end=' ')
        if full:
            return ccd, dict(posmask=None, negmask=None, med_filt=None, med_sub=None, med_rat=None,
                             std_rat=None, std=None)
        else:
            return ccd

    logical = np.array(logical)
    if logical.size == 1:
        logical = np.repeat(logical, 2)
    elif logical.size > 2:
        raise ValueError("logical must be at most size 2.")

    _LOGICAL_AND = []
    _LOGICAL_STR = []
    for i, _logical in enumerate(logical):
        _logical_and = _logical in ['and', '&']
        if not _logical_and and _logical not in ['or', '|']:
            raise ValueError("logical not understood.")

        _LOGICAL_AND.append(_logical_and)
        _LOGICAL_STR.append("and" if _logical_and else "or")

    def _set_masks(arr2test, clips):
        if clips[0] is None:  # let lower clip does not affect final mask
            _negmask = _LOGICAL_AND[0]  # isinstance bool
        else:
            _negmask = arr2test < clips[0]  # isinstance ndarray

        if clips[1] is None:  # let upper clip does not affect final mask
            _posmask = _LOGICAL_AND[1]  # isinstance bool
        else:
            _posmask = arr2test > clips[1]  # isinstance ndarray

        return _negmask, _posmask

    if not isinstance(ccd, CCDData):
        raise TypeError("ccd should be CCDData")

    nccd = ccd.copy()
    arr = nccd.data.astype(dtype)
    hdr = nccd.header

    # add very small const to avoid resulting value of 0.0 in med_filt
    # which results in zero-division in med_ratio below.
    arr += cadd

    if std_section is not None:
        slices = fitsxy2py(std_section)
    else:
        slices = [slice(None, None, None)]*arr.ndim

    medfilt_kw = dict(size=size, mode=mode, cval=cval, origin=origin)

    _t = Time.now()
    med_filt = median_filter(arr, **medfilt_kw)

    if update_header:
        add_to_header(
            hdr, 'h', verbose=verbose, t_ref=_t,
            s=f"Median filtered (convolved) frame calculated with {medfilt_kw}"
        )

    if std_model == 'ccd':
        _t = Time.now()
        gain = change_to_quantity(gain, u.electron/u.adu, to_value=True)
        rdnoise = change_to_quantity(rdnoise, u.electron, to_value=True)

        std = np.sqrt((1 + snoise)*med_filt/gain + (rdnoise/gain)**2)
        if update_header:
            add_to_header(
                hdr, 'h', verbose=verbose, t_ref=_t,
                s=("Stdev map is generated from median filtered frame by "
                   + "sqrt{(1 + snoise)*med_filt/gain + (rdnoise/gain)**2}")
            )
            hdr['MB_MODEL'] = (std_model, "Method used for getting stdev map")
            hdr["MB_GAIN"] = (gain, "gain used for stdev map in MBPM")
            hdr["MB_RDN"] = (rdnoise, "rdnoise used for stdev map in MBPM")
            hdr["MB_SSN"] = (snoise, "snoise used for stdev map in MBPM")

    elif std_model == 'std':
        _t = Time.now()
        _, _, std = sigma_clipped_stats(arr[tuple(slices)], **sigclip_kw)

        if update_header:
            if std_section is None:
                std_section = '[' + ','.join([':'] * arr.ndim) + ']'
            hdr['MB_MODEL'] = (std_model, "Method used for getting stdev map")
            hdr["MB_SSKY"] = (std, "Sky stdev for median filter BPM (MBPM) algorithm")
            hdr["MB_SSECT"] = (f"{std_section}", "Sky stdev calculation section in MBPM algorithm")
            add_to_header(
                hdr, 'h', verbose=verbose, t_ref=_t,
                s=("Sky standard deviation (MB_SSKY) calculated by sigma clipping at MB_SSECT with "
                   + f"{sigclip_kw}; used for std_ratio map calculation.")
            )

    elif std_model is None:
        hdr['MB_MODEL'] = ('None', "Method used for getting stdev map")
        std = 1  # so that med_ratio is nothing but med_sub itself below.
        std_rat_clip = None  # turn off clipping using std_ratio

    else:
        raise ValueError("std_model not understood.")

    med_sub_clip = _sanitize_clips(med_sub_clip)
    med_rat_clip = _sanitize_clips(med_rat_clip)
    std_rat_clip = _sanitize_clips(std_rat_clip)

    _t = Time.now()
    npmask = []
    for msc, mrc, src in zip(med_sub_clip, med_rat_clip, std_rat_clip):
        if (isinstance(msc, bool) and isinstance(mrc, bool) and isinstance(src, bool)):
            npmask.append(np.zeros_like(arr, dtype=bool))

    med_ratio = arr/np.abs(med_filt)
    # Above is identical to sign(arr)*abs(arr/med_filt)
    med_sub = arr - med_filt
    std_ratio = med_sub/std

    # mask in the order of negative and positive cases
    mask_ms = _set_masks(med_sub, med_sub_clip)
    mask_mr = _set_masks(med_ratio, med_rat_clip)
    mask_sr = _set_masks(std_ratio, std_rat_clip)

    masks = []
    for i, (ms, mr, sr) in enumerate(zip(mask_ms, mask_mr, mask_sr)):
        if (isinstance(ms, bool) and isinstance(mr, bool) and isinstance(sr, bool)):
            # i.e., if all of neg or pos were None
            masks.append(np.zeros_like(arr, dtype=bool))  # all False

        else:  # if at least one was not None:
            if _LOGICAL_AND[i]:
                masks.append(ms & mr & sr)
            else:
                masks.append(ms | mr | sr)

    replace_mask = masks[0] | masks[1]
    arr[replace_mask] = med_filt[replace_mask]

    if update_header:
        hdr["MB_NLOGI"] = (_LOGICAL_STR[0], "The logic used for negative MBPM masks (and/or)")
        hdr["MB_PLOGI"] = (_LOGICAL_STR[1], "The logic used for positive MBPM masks (and/or)")
        hdr["MB_RAT_U"] = (med_rat_clip[1], "Upper clip of (data/|medfilt|) map (MBPM)")
        hdr["MB_RAT_L"] = (med_rat_clip[0], "Lower clip of (data/|medfilt|) map (MBPM)")
        hdr["MB_SUB_U"] = (med_sub_clip[1], "Upper clip of (data-medfilt) map (MBPM)")
        hdr["MB_SUB_L"] = (med_sub_clip[0], "Lower clip of (data-medfilt) map (MBPM)")
        hdr["MB_STD_U"] = (std_rat_clip[1], "Upper clip of (data-medfilt)/std map (MBPM)")
        hdr["MB_STD_L"] = (std_rat_clip[0], "Lower clip of (data-medfilt)/std map (MBPM)")

        add_to_header(
            hdr, 'h', verbose=verbose, t_ref=_t,
            s="Median-filter based Bad-Pixel Masking (MBPM) applied."
        )
        # add_to_header(
        #     hdr, 'h', verbose=verbose, t_ref=_t,
        #     s=("(1) Median additive difference (data-medfilt) generated, "
        #        + "(2) Median ratio (data/|medfilt|) generated, "
        #        + "(3) Stddev ratio ((data-medfilt)/std) generated, "
        #        + "(4) posmask and negmask calculated by clips "
        #        + "MB_[ADD/RAT/STD]_[U/L] and logic MB_[N/P]LOG (see keywords),"
        #        + "(5) Pixels of (posmask | negmask) are repleced with median "
        #        + "filtered frame."
        #        ))

    nccd = CCDData(data=arr - cadd, header=hdr, unit=nccd.unit)

    if full:
        return nccd, dict(negmask=masks[0], posmask=masks[1],
                          med_filt=med_filt, med_sub=med_sub, med_rat=med_ratio,
                          std_rat=std_ratio, std=std)
    else:
        return nccd


# NOTE: crrej should be done AFTER bias/dark and flat correction:
# http://www.astro.yale.edu/dokkum/lacosmic/notes.html
def bdf_process(ccd, output=None, extension=None,
                mbiaspath=None, mdarkpath=None, mflatpath=None, mfringepath=None,
                mbias=None, mdark=None, mflat=None, mfringe=None,
                fringe_scale_fun=np.mean, fringe_scale_section=None,
                trim_fits_section=None, calc_err=False, unit=None,
                gain=None, gain_key="GAIN", gain_unit=u.electron/u.adu,
                rdnoise=None, rdnoise_key="RDNOISE", rdnoise_unit=u.electron,
                exposure_key="EXPTIME", exposure_unit=u.s, dark_exposure=None, data_exposure=None,
                dark_scale=False, normalize_exposure=False, normalize_average=False, normalize_median=False,
                flat_min_value=None, flat_norm_value=1.,
                do_crrej=False, crrej_kwargs=None, propagate_crmask=False,
                verbose_crrej=False, verbose_bdf=True, output_verify='fix',
                overwrite=True, dtype="float32", uncertainty_dtype="float32"):
    ''' Do bias, dark and flat process.
    Parameters
    ----------
    ccd : CCDData
        The ccd to be processed.

    output : path-like or None, optional.
        The path if you want to save the resulting ``ccd`` object. Default is `None`.

    mbiaspath, mdarkpath, mflatpath, mfringepath : path-like, optional.
        The path to master bias, dark, flat, and fringe FITS files. If `None`, the corresponding
        process is not done. These can be provided in addition to ``mbias``, ``mdark``, ``mflat``,
        and/or ``mfringe``.

    mbias, mdark, mflat, mfringe : CCDData, optional.
        The master bias, dark, and flat in `~astropy.nddata.CCDData`. If this is given, the files
        provided by ``mbiaspath``, ``mdarkpath``, ``mflatpath`` and/or ``mfringe`` are **not** loaded,
        but these paths will be used for header (``BIASPATH``, ``DARKPATH``, ``FLATPATH`` and/or
        ``FRINPATH``). If the paths are not given, the header values will be ``<User>``.

    fringe_scale_fun : function object, optional.
        The function to be used to scale the fringe before subtraction, specified by the region
        ``fringe_scale_section``. This scaling is not done if ``fringe_scale_section`` is `None`.

    fringe_scale_section : str, optional.
        The FITS-convention section of the fringe and object (science) frames to match the fringe
        pattern before the subtraction. If `None`, this scaling is turned off. To use all region, use
        such as ``'[:, :]`` for 2-D.

    trim_fits_section: str, optional.
        Region of ``ccd`` to be trimmed; see ``ccdproc.subtract_overscan`` for details. Default is
        `None`.

    calc_err : bool, optional.
        Whether to calculate the error map based on Poisson and readnoise error propagation.

        ..note::
            Currently it's encouraged to make error-map manually, as the API is not stable.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is `None`.

    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. These are all ignored if ``calc_err=False`` and
        ``do_crrej=False``. If ``calc_err=True``, it automatically seeks for suitable gain and
        readnoise value. If ``gain`` or ``readnoise`` is specified, they are interpreted with
        ``gain_unit`` and ``rdnoise_unit``, respectively. If they are not specified, this function will
        seek for the header with keywords of ``gain_key`` and ``rdnoise_key``, and interprete the
        header value in the unit of ``gain_unit`` and ``rdnoise_unit``, respectively.

    gain_key, rdnoise_key : str, optional.
        See ``gain``, ``rdnoise`` explanation above.
        These are all ignored if ``calc_err=False``.

    gain_unit, rdnoise_unit : str, astropy.Unit, optional.
        See ``gain``, ``rdnoise`` explanation above.
        These are all ignored if ``calc_err=False``.

    dark_exposure, data_exposure : None, float, astropy Quantity, optional.
        The exposure times of dark and data frame, respectively. They should both be specified or both
        `None`. These are all ignored if ``mdarkpath=None``. If both are not specified while
        ``mdarkpath`` is given, then the code automatically seeks for header's ``exposure_key``. Then
        interprete the value as the quantity with unit ``exposure_unit``. If ``mdkarpath`` is not
        `None`, then these are passed to ``ccdproc.subtract_dark``.

    exposure_key : str, optional.
        The header keyword for exposure time.
        Ignored if ``mdarkpath=None``.

    exposure_unit : astropy Unit, optional.
        The unit of the exposure time.
        Ignored if ``mdarkpath=None``.

    normalize_exposure : bool, optional.
        Whether to normalize the values by the exposure time of each frame. Maybe useful for long
        exposure darks to make 1-sec darks.
        Default is `False`.

    normalize_average, normalize_median : bool, optional.
        Whether to normalize the values by the average or median value of each frame before combining.
        Only up to one of these must be True. Maybe useful for flat.
        Default is `False`.

    flat_min_value : float or None, optional.
        min_value of `ccdproc.flat_correct`. Minimum value for flat field. The value can either be None
        and no minimum value is applied to the flat or specified by a float which will replace all
        values in the flat by the min_value.
        Default is `None`.

    flat_norm_value : float or None, optional.
        The norm_value of `ccdproc.flat_correct`. If `None`, the flat is internally normalized by its
        mean before the flat correction, i.e., the flat correction will be like
        ``image/flat*mean(flat)``.
        If not `None`, the flat correction will be like ``image/flat*flat_norm_value``.
        Default is 1 (**different** from `ccdproc` which uses `None` as default).

    crrej_kwargs : dict or None, optional.
        If `None` (default), uses some default values (see ``crrej``). It is always discouraged to use
        default except for quick validity-checking, because even the official L.A. Cosmic codes in
        different versions (IRAF, IDL, Python, etc) have different default parameters, i.e., there is
        nothing which can be regarded as _the default_. To see all possible keywords, do
        ``print(astroscrappy.detect_cosmics.__doc__)`` Also refer to
        https://nbviewer.jupyter.org/github/ysbach/AO2019/blob/master/Notebooks/07-Cosmic_Ray_Rejection.ipynb

    propagate_crmask : bool, optional.
        Whether to save (propagate) the mask from CR rejection (``astroscrappy``) to the CCD's mask.
        Default is `False`.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``, ``"ignore"``,
        ``"warn"``, or ``"exception"``. May also be any combination of ``"fix"`` or ``"silentfix"``
        with ``"+ignore"``, ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below: http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    dtype : str or `numpy.dtype` or None, optional.
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter description. If `None` it uses
        ``np.float64``.
        Default is `None`.
    '''
    def _load_master(path, master):
        if path is None and master is None:
            do = False
            master = None
        else:  # at least one is given
            do = True
            if master is None:
                master = load_ccd(path, unit=unit, ccddata=True, use_wcs=False)

            # Make master as CCDData
            master, _, _ = _parse_image(master, force_ccddata=True)

            if path is None:
                path = "<User>"

            if not calc_err:
                master.uncertainty = None

        return do, master, path

    # Initial setting
    # if not isinstance(ccd, CCDData):
    #     raise TypeError(f"ccd must be CCDData (now it is {type(ccd)})")
    ccd, _, _ = _parse_image(ccd, extension=extension, force_ccddata=True)
    PROCESS = []
    proc = ccd.copy()

    # Add PROCESS key
    if "PROCESS" not in proc.header:
        proc.header["PROCESS"] = ("", "Process (order: 1-2-3-...): see comment.")
        add_to_header(proc.header, 'c',
                      ("Standard PROCESS key includes B=bias, D=dark, F=flat, "
                       + "T=trim, W=WCS (astrometry), C=CRrej, Fr=fringe.")
                      )

    # Log the CCDPROC version
    if "CCDPROCV" in proc.header:
        if str(proc.header["CCDPROCV"]) != str(ccdproc.__version__):
            add_to_header(
                proc.header, "h",
                f"The ccdproc version prior to this modification was {proc.header['CCDPROCV']}.")
            proc.header["CCDPROCV"] = (ccdproc.__version__, "ccdproc version used for processing.")
        # else (no version change): do nothing.
    else:
        proc.header["CCDPROCV"] = (ccdproc.__version__, "ccdproc version used for processing.")

    # Set for BIAS
    do_bias, mbias, mbiaspath = _load_master(mbiaspath, mbias)
    if do_bias:
        PROCESS.append("B")
        proc.header["BIASPATH"] = (str(mbiaspath), "Path to the used bias file")

    # Set for DARK
    do_dark, mdark, mdarkpath = _load_master(mdarkpath, mdark)
    if do_dark:
        proc.header.append("D")
        proc.header["DARKPATH"] = (str(mdarkpath), "Path to the used dark file")

        if dark_scale:
            # TODO: what if dark_exposure, data_exposure are given explicitly?
            add_to_header(proc.header, 'h', str_dscale.format(exposure_key), verbose=verbose_bdf)

    # Set for FLAT
    do_flat, mflat, mflatpath = _load_master(mflatpath, mflat)
    if do_flat:
        PROCESS.append("F")
        proc.header["FLATPATH"] = (str(mflatpath), "Path to the used flat file")
        proc.header["FLATNORM"] = (flat_norm_value, "flat_norm_value (none = mean of input flat)")

    # set for FRINGE
    do_fringe, mfringe, mfringepath = _load_master(mfringepath, mfringe)
    if do_fringe:
        PROCESS.append("Fr")
        proc.header["FRINPATH"] = (str(mfringepath), "Path to the used fringe")
        if fringe_scale_section is not None:
            proc.header["FRINSECT"] = (fringe_scale_section, "FITS section used for scaling fringe")
            proc.header["FRINFUNC"] = (fringe_scale_fun.__name__, "Function used for fringe scaling")

    # Set gain and rdnoise if at least one of calc_err and do_crrej is True.
    if calc_err or do_crrej:
        set_ccd_gain_rdnoise(proc,
                             gain=gain,
                             gain_key=gain_key,
                             gain_unit=gain_unit,
                             rdnoise=rdnoise,
                             rdnoise_key=rdnoise_key,
                             rdnoise_unit=rdnoise_unit,
                             verbose=verbose_bdf,
                             update_header=True
                             )
        gain_Q = proc.gain
        rdnoise_Q = proc.rdnoise

    # Do TRIM
    if trim_fits_section is not None:
        _t = Time.now()
        sect = dict(fits_section=trim_fits_section)
        proc = trim_ccd(proc, **sect)
        mbias = trim_ccd(mbias, **sect) if do_bias else None
        mdark = trim_ccd(mdark, **sect) if do_dark else None
        mflat = trim_ccd(mflat, **sect) if do_flat else None
        mfringe = trim_ccd(mfringe, **sect) if do_fringe else None
        PROCESS.append("T")

        add_to_header(proc.header, 'h', str_trim.format(trim_fits_section), verbose=verbose_bdf, t_ref=_t)

    # Do BIAS
    if do_bias:
        _t = Time.now()
        proc = subtract_bias(proc, mbias)
        add_to_header(proc.header, 'h', str_bias.format(mbiaspath), verbose=verbose_bdf, t_ref=_t)

    # Do DARK
    if do_dark:
        _t = Time.now()
        proc = subtract_dark(proc,
                             mdark,
                             dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
        add_to_header(proc.header, 'h', str_dark.format(mdarkpath), verbose=verbose_bdf, t_ref=_t)

    # Make UNCERT extension before doing FLAT and FRINGE
    #   It is better to make_errmap a priori because of mathematical and
    #   computational convenience. See ``if do_flat:`` clause below.
    if calc_err:
        _t = Time.now()
        err = errormap(proc, gain_epadu=gain, subtracted_dark=mdark)
        proc.uncertainty = StdDevUncertainty(err)

        s = [str_e0]
        if do_dark:
            s.append(str_ed)
        add_to_header(proc.header, 'h', s, verbose=verbose_bdf, t_ref=_t)

    # Do FLAT
    if do_flat:
        # Flat error propagation is done automatically by
        # ``ccdproc.flat_correct``if it has the uncertainty attribute.
        _t = Time.now()
        proc = flat_correct(proc,
                            mflat,
                            min_value=flat_min_value,
                            norm_value=flat_norm_value)
        s = [str_flat.format(mflatpath)]
        if calc_err and mflat.uncertainty is not None:
            s.append(str_ef)

        add_to_header(proc.header, 'h', s, verbose=verbose_bdf, t_ref=_t)

    # Normalize by the exposure time (e.g., ADU per sec)
    if normalize_exposure:
        _t = Time.now()
        if data_exposure is None:
            data_exposure = proc.header[exposure_key]
        proc.data = proc.data/data_exposure  # uncertainty will also be..
        add_to_header(proc.header, 'h', str_nexp, verbose=verbose_bdf, t_ref=_t)

    # Normalize by the mean value
    if normalize_average:
        _t = Time.now()
        avg = np.mean(proc.data)
        proc.data = proc.data/avg
        add_to_header(proc.header, 'h', str_navg, verbose=verbose_bdf, t_ref=_t)

    # Normalize by the median value
    if normalize_median:
        _t = Time.now()
        med = np.median(proc.data)
        proc.data = proc.data/med
        add_to_header(proc.header, 'h', str_nmed, verbose=verbose_bdf, t_ref=_t)

    # Do CRREJ
    if do_crrej:
        if crrej_kwargs is None:
            crrej_kwargs = LACOSMIC_KEYS.copy()
            warn("You are not specifying CR-rejection parameters! It can be"
                 + " dangerous to use defaults blindly.")

        if (("B" in proc.header["PROCESS"])
            + ("D" in proc.header["PROCESS"])
                + ("F" in proc.header["PROCESS"])) < 2:
            warn("L.A. Cosmic should be run AFTER bias, dark, flat process. You have only done"
                 + f" {proc.header['PROCESS']}. See http://www.astro.yale.edu/dokkum/lacosmic/notes.html")

        proc, _ = crrej(
            proc,
            propagate_crmask=propagate_crmask,
            update_header=True,
            gain=gain_Q,        # already parsed from local variables above
            rdnoise=rdnoise_Q,  # already parsed from local variables above
            verbose=verbose_crrej,
            **crrej_kwargs)

    # To avoid ``pssl`` in cr rejection, subtract fringe AFTER the CRREJ.
    # Do FRINGE
    if do_fringe:
        _t = Time.now()
        if fringe_scale_section is not None:
            sl = fitsxy2py(fringe_scale_section)
            scale = fringe_scale_fun(proc.data[sl]) / fringe_scale_fun(mfringe.data[sl])
            proc.header["FRINSCAL"] = (scale, "Scale factor multiplied to fringe")
            s = str_fringe_scale
        else:
            scale = 1  # Not 1.0 so that (ccd in int) - (mfringe in int) remains int.
            s = str_fringe_noscale
        mfringe.data *= scale
        proc.data = proc.subtract(mfringe).data  # should I calc. error..?
        add_to_header(proc.header, 'h', s, verbose=verbose_bdf, t_ref=_t)

    proc = CCDData_astype(proc, dtype=dtype, uncertainty_dtype=uncertainty_dtype)
    update_process(proc.header, PROCESS, key="PROCESS", delimiter='-', add_comment=True)
    update_tlm(proc.header)

    if output is not None:
        if verbose_bdf:
            print(f"Writing FITS to {output}... ", end='')
        proc.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose_bdf:
            print("Saved.")
    return proc
