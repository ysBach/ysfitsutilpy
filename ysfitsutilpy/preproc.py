from warnings import warn

import astroscrappy
import ccdproc
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astroscrappy import detect_cosmics
from ccdproc import flat_correct, subtract_bias, subtract_dark

from .hduutil import (CCDData_astype, _parse_data_header, _parse_image,
                      cmt2hdr, errormap, fixpix, listify, load_ccd,
                      propagate_ccdmask, set_ccd_gain_rdnoise, trim_ccd,
                      update_process, update_tlm)
from .misc import change_to_quantity, fitsxy2py, parse_crrej_psf

__all__ = [
    "crrej", "medfilt_bpm", "bdf_process", "run_reduc_plan"
]


# def _add_and_print(s, header, verbose, update_header=True, t_ref=None):
#     if update_header:
#         # add as history
#         cmt2hdr(header, 'h', s, t_ref=t_ref)
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
#     """ Get gain and readnoise from given paramters.
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
#     """
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


def crrej(
        ccd,
        mask=None,
        inbkg=None,
        invar=None,
        propagate_crmask=False,
        update_header=True,
        add_process=True,
        gain=None,
        rdnoise=None,
        sigclip=4.5,
        sigfrac=0.5,
        objlim=1.0,
        satlevel=np.inf,
        niter=4,
        sepmed=False,
        cleantype='medmask',
        fs="median",
        psffwhm=2.5,
        psfsize=7,
        psfbeta=4.765,
        verbose=True
):
    """ Do cosmic-ray rejection using L.A.Cosmic default parameters.
    Parameters
    ----------
    ccd : CCDData
        The ccd to be processed. The data must be in ADU, not electrons.

    propagate_crmask : bool, optional.
        Whether to save (propagate) the mask from CR rejection
        (`~astroscrappy`) to the CCD's mask. Default is `False`.

    inbkg : float, ndarray, path-like to FITS, optional
        A pre-determined background image, to be subtracted from `ccd` before
        running the main detection algorithm. This is used primarily with
        spectroscopic data, to remove sky lines and the cross-section of an
        object continuum during iteration, "protecting" them from spurious
        rejection (see the above paper). This background is not removed from
        the final, cleaned output (`cleanarr`). This should be in units of
        "counts", the same units of `ccd`. `inbkg` should be free from cosmic
        rays. When estimating the cosmic-ray free noise of the image, we will
        treat `inbkg` as a constant Poisson contribution to the variance.
        Default: None.

        ..note:
            Originally `pssl`, which stood for "previously subtracted sky
            level" in ADU (in `astroscrappy` < 1,.1.0 or original L.A.Cosmic).
            Since `astroscrappy` ver > 1.1.0, a 2-D sky level is supported by
            `inbkg` (it was `bkg` in == 1.1.0, which is a hasty bug in argument
            naming).

    invar : float numpy array, path-like to FITS, optional
        A pre-determined estimate of the data variance (ie. noise squared) in
        each pixel, generated by previous processing of `ccd`. If provided,
        this is used in place of an internal noise model based on `ccd`, `gain`
        and `readnoise`. This still gets median filtered and cleaned
        internally, to estimate what the noise in each pixel *would* be in the
        absence of cosmic rays. This should be in units of "counts" squared.
        (it was `var` in == 1.1.0, which is a hasty bug in argument naming)

    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. If not ``Quantity``, they must be in
        electrons per adu and electron unit, respectively.

    sigclip : float, optional
        Laplacian-to-noise limit for cosmic ray detection. Lower values will
        flag more pixels as cosmic rays.
        Default: 4.5.

    sigfrac : float, optional
        Fractional detection limit for neighboring pixels. For cosmic ray
        neighbor pixels, a lapacian-to-noise detection limit of sigfrac *
        sigclip will be used.
        Default: 0.5.

    objlim : float, optional
        Minimum contrast between Laplacian image and the fine structure image.
        Increase this value if cores of bright stars are flagged as cosmic
        rays.
        Default: 1.0.

    satlevel : float, optional
        Saturation of level of the image (electrons). This value is used to
        detect saturated stars and pixels at or above this level are added to
        the mask.
        Default: ``np.inf``.

    niter : int, optional
        Number of iterations of the LA Cosmic algorithm to perform.
        Default: 4.

    sepmed : boolean, optional
        Use the separable median filter instead of the full median filter. The
        separable median is not identical to the full median filter, but they
        are approximately the same and the separable median filter is
        significantly faster and still detects cosmic rays well.
        Default: `True`

    cleantype : {'median', 'medmask', 'meanmask', 'idw'}, optional
        Set which clean algorithm is used:

        * ``'median'``: An umasked 5x5 median filter
        * ``'medmask'``: A masked 5x5 median filter
        * ``'meanmask'``: A masked 5x5 mean filter
        * ``'idw'``: A masked 5x5 inverse distance weighted interpolation

        Default: ``"medmask"``.

    fs : {'median', 'gauss', 'gaussx', 'gaussy', 'moffat'}, ndarray, optional.
        Method to generate the fine structure. Combination of `fsmode`,
        `psfmodel`, `psfk` of `astroscrappy`.

        * ``'median'``: Use the median filter in the standard LA Cosmic
          algorithm. None of `psffwhm`, `psfsize`, and `psfbeta` are used.
        * other str: Use a Gaussian/Moffat model to generate the psf kernel.
          ``'gauss'|'moffat'`` produce circular PSF kernels.
          ``'gaussx'|'gaussy'`` produce Gaussian kernels in the x and y
          directions respectively. `psffwhm`, `psfsize` (plus `psfbeta` if
          "moffat") are used.
        * ndarray: PSF kernel array to use for the fine structure image. None
          of `psffwhm`, `psfsize`, and `psfbeta` are used.

        Summary of `astroscrappy` VS `ysfitsutilpy`:

        * ``fsmode="median"`` == ``fs="median"``
        * ``fsmode="convolve", psfmodel=*`` == ``fs=*``, where ``*`` can be
          any of ``{'gauss', 'gaussx', 'gaussy', 'moffat'}``.
        * ``fsmode="convolve", psfk=<ndarray>`` == ``fs=<ndarray>``

        Default: ``'median'``.

    psffwhm : float, optional
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.

    psfsize : int, optional
        Size of the kernel to calculate. Returned kernel will have size `psfsize`
        x `psfsize`. It should be an odd integer.
        Default: 7.

    psfbeta : float, optional
        Moffat beta parameter. Only used if ``fs='moffat'``.
        Default: 4.765.

    verbose : boolean, optional
        Print to the screen or not. Default: `False`.

    Returns
    -------
    _ccd : CCDData
        The cosmic-ray cleaned CCDData in ADU. `~astroscrappy` automatically
        does a gain correction, so I divided the `~astroscrappy` result by
        gain to restore to ADU (not to surprise the users).

    crmask : ndarray (mask)
        The cosmic-ray mask from `~astroscrappy`, propagated by the original
        mask of the ccd (if ``ccd.mask`` is not `None`) and `mask` given by
        the user.

    update_header : bool, optional.
        Whether to update the header if there is any.

    add_process : bool, optional.
        Whether to add ``PROCESS`` key to the header.

    Notes
    -----
    Detection related (important ones): `sigclip`, `sigfrac`, `objlim`
    Kernel (fine structure) related: `fsmode`
      * If "median": median filter of `psfsize` x `psfsize`
      * If "convolve":
        * If `psfk` is given, use it as the kernel.
        * If `psfk` is `None`, use `psfmodel` to generate the kernel.
          * `psffwhm`: The FWHM for `psfmodel` of "gauss" or "moffat".

      * `psfk` is  `psfsize`, `psfk`, `psfbeta`
    Detector specific parameters: `gain`, `rdnoise`
    Rarely tuned parameters: `pssl`, `satlevel`, `niter`

    (Note from `astroscrappy`)
    For best results on spectra, we recommend that you include an estimate of
    the background. One can generally obtain this by fitting columns with a
    smooth function. To efficiently identify cosmic rays, LA Cosmic and
    therefore astroscrappy estimates the cosmic ray free noise by smoothing the
    variance using a median filter. To minimize false positives on bright sky
    lines, if `inbkg` is provided, we do not smooth the variance contribution
    from the provided background. We only smooth the variance that is in
    addition to the Poisson contribution from the background so that we do not
    underestimate the noise (and therefore run the risk of flagging false
    positives) near narrow, bright sky lines.

    All defaults are based on IRAF version of L.A. Cosmic (Note the default
    parameters of L.A. Cosmic differ from version to version, so I took the
    IRAF version written by van Dokkum.)
    See the docstring of astroscrappy by
    >>> import astroscrappy
    >>> astroscrappy.detect_cosmics?

    Example
    -------
    >>> yfu.ccdutil.set_ccd_gain_rdnoise(ccd)
    >>> nccd, mask = crrej(ccd)
    """
    str_cr = ("Cosmic-Ray rejection (CRNFIX={:d} pixels fixed) by astroscrappy (v {}). "
              + "Parameters: {}")

    _t = Time.now()

    if gain is None:
        try:
            gain = ccd.gain
        except AttributeError:
            raise ValueError(
                "Gain must be given or accessible as ``ccd.gain``. "
                + "Use, e.g., yfu.set_ccd_gain_rdnoise(ccd)."
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
    if mask is None:
        inmask = None
    else:
        inmask = _parse_image(mask)[0]
        inmask = propagate_ccdmask(_ccd, additional_mask=inmask)

    # The L.A. Cosmic accepts only the gain in e/adu and rdnoise in e.
    gain = change_to_quantity(gain, u.electron/u.adu, to_value=True)
    rdnoise = change_to_quantity(rdnoise, u.electron, to_value=True)

    inbkg = None if inbkg is None else _parse_image(inbkg)[0]
    invar = None if invar is None else _parse_image(invar)[0]

    # remove the fucxing cosmic rays
    crrej_kwargs = dict(
        gain=1.0,
        rdnoise=rdnoise,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        satlevel=satlevel,
        niter=niter,
        sepmed=sepmed,
        cleantype=cleantype,
        **parse_crrej_psf(
            fs=fs,
            psffwhm=psffwhm,
            psfsize=psfsize,
            psfbeta=psfbeta
        )
    )
    try:
        crmask, cleanarr = detect_cosmics(
            data,
            inmask=inmask,
            inbkg=inbkg,
            invar=invar,
            verbose=verbose,
            **crrej_kwargs
        )
    except TypeError:  # astroscrappy < 1.1.1 (Commit on 2021-11-20) Jeez...
        try:
            crmask, cleanarr = detect_cosmics(
                data,
                inmask=inmask,
                bkg=inbkg,
                var=invar,
                verbose=verbose,
                **crrej_kwargs
            )
        except TypeError:  # astroscrappy < 1.1.0 (Commit on 2020-11-21) Jeez...
            # Error if inbkg is ndarray
            crmask, cleanarr = detect_cosmics(
                data,
                inmask=inmask,
                pssl=inbkg,
                verbose=verbose,
                **crrej_kwargs
            )

    # create the new ccd data object
    #   astroscrappy used to return the data in e- unit, but suddenly changed
    #   in around version 1.1.0.... Jeez...
    #   https://github.com/astropy/astroscrappy/issues/73
    divfactor = detect_cosmics(np.ones((3, 3)), gain=1., niter=0)[1][0, 0]
    # %timeit detect_cosmics(np.ones((3, 3)), gain=1., niter=0)[1][0, 0] == 1.
    # ^ takes ~ 40 us VS ~ 1200 us on MBP 15" [2018, macOS 11.6 i7-8850H (2.6
    # GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)] VS MBP 16"
    # [2021, macOS 12.0.1, M1Pro, 8P+2E core, GPU 16-core, RAM 16GB].
    # I dunno why they differ so much... Does not change much w.r.t. gain,
    # shape, etc. Maybe cuz I'm using Rosetta2 (Anaconda) on the latter?
    # 2021-12-13 13:27:35 (KST: GMT+09:00) ysBach
    if divfactor != 1.:
        _ccd.data = cleanarr / divfactor

    if propagate_crmask:
        _ccd.mask = propagate_ccdmask(_ccd, additional_mask=crmask)

    if add_process and hdr is not None:
        try:
            hdr["PROCESS"] += "C"
        except KeyError:
            hdr["PROCESS"] = "C"

    if update_header and hdr is not None:
        nrej_cr = np.sum(crmask)
        hdr["CRNFIX"] = (nrej_cr, "Number of cosmic-ray pixels fixed.")
        cmt2hdr(
            hdr, 'h', verbose=verbose, t_ref=_t,
            s=str_cr.format(nrej_cr, astroscrappy.__version__, crrej_kwargs)
        )
    else:
        if verbose:
            nrej_cr = np.sum(crmask)
            print(str_cr.format(nrej_cr, astroscrappy.__version__, crrej_kwargs))

    update_tlm(hdr)
    _ccd.header = hdr

    return _ccd, crmask


# TODO: put niter
# TODO: put medfilt_min
#   to get std at each pixel by medfilt[<medfilt_min] = 0, and std =
#   sqrt((1+snoise)*medfilt/gain + rdn**2)
def medfilt_bpm(
        ccd,
        cadd=1.e-10,
        std_model="std",
        gain=1.,
        rdnoise=0.,
        snoise=0.,
        size=5,
        sigclip_kw=dict(sigma=3., maxiters=5, std_ddof=1),
        std_section=None,
        footprint=None,
        mode='reflect',
        cval=0.0,
        origin=0,
        med_sub_clip=None,
        med_rat_clip=[0.5, 2],
        std_rat_clip=[-5, 5],
        dtype='float32',
        update_header=True,
        verbose=False,
        logical='and',
        full=False
):
    """ Find bad pixels from median filtering technique (non standard..?)
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The CCD to find the bad pixels.

    cadd : float, optional.
        A very small const to be added to the input array to avoid resulting
        value of 0.0 in the median filtered image which raises zero-division in
        median ratio (image/|median_filtered|).

    std_model : {"std", "ccd"} optional.
        The model used to calculate the std (standard deviation) map.

        - ``"std"``: Simple standard deviation is calculated.
        - ``"ccd"``: Using CCD noise model (``sqrt{(1 + snoise)*med_filt/gain
          + (rdnoise/gain)**2}``)

        For ``'std'``, the arguments `std_section` and `sigclip_kw` are used,
        while if ``'ccd'``, arguments `gain`, `rdnoise`, `snoise` will be used.

    size, footprint, mode, cval, origin : optional.
        The parameters to obtain the median-filtered map. See
        `~scipy.ndimage.median_filter`.

    sigclip_kw : dict, optional.
        The paramters used for `~astropy.stats.sigma_clipped_stats` when
        estimating the sky standard deviation at `std_section`. This is
        **ignored** if ``std_model='ccd'``.

    std_section : str, optinal.
        The region in FITS standard (1-indexing, end-inclusive, xyz order) to
        estimate the sky standard deviation to obtain the `std_ratio`. If
        `None` (default), the full region of the given array is used, which is
        many times not desirable due to the celestial objects in the FOV and
        computational cost. This is **ignored** if ``std_model='ccd'``.

    gain, rdnoise, snoise : float, optional.
        The gain (electrons/ADU), readout noise (electrons), and sensitivity
        noise (fractional error from flat fielding) of the frame. These are
        **ignored** if ``std_model="std"``.

    med_sub_clip : list of two float or `None`, optional.
        The thresholds to find bad pixel by ``med_sub = ccd.data -
        median_filter(ccd.data)``. The clipping will be turned off if it is
        `None` (default). If a list, must be in the order of ``[lower, upper]``
        and at most two of these can be `None`.

    med_rat_clip : list of two float or `None`, optional.
        The thresholds to find bad pixel by ``med_ratio =
        ccd.data/np.abs(median_filter(ccd.data))``. The clipping will be turned
        off if it is `None` (default). If a list, must be in the order of
        ``[lower, upper]`` and at most two of these can be `None`.

    std_rat_clip : list of two float or `None`, optional.
        The thresholds to find bad pixel by ``std_ratio = (ccd -
        median_filter(ccd))/std``. The clipping will be turned off if it is
        `None` (default). If a list, must be in the order of ``[lower, upper]``
        and at most two of these can be `None`.

    logical : {'and', '&', 'or', '|'} or list of these, optional.
        The logic to propagate masks determined by the ``_clip``'s. The mask is
        propagated such as ``posmask = med_sub > med_sub_clip[1] &/| med_ratio
        > med_rat_clip[1] &/| std_ratio > std_rat_clip[1]``. If a list, it must
        contain two str of these, in the order of ``[logical_negmask,
        logical_posmask]``.

    Returns
    -------
    ccd : CCDData
        The badpixel removed result.

    The followings are returned as dict only if ``full=True``.

    posmask, negmask : ndarry of bool
        The masked pixels by positive/negative criteria.

    sky_std : float
        The (sigma-clipped) sky standard deviation. Returned only if
        ``full=True``.

    Notes
    -----
    `med_sub_clips` is usually not necessary but useful to detect hot pixels in
    dark frames (no light) for some special circumstances. ::

    1. Median additive difference (data-medfilt) generated,
    2. Median ratio (data/|medfilt|) generated,
    3. Stddev ratio ((data-medfilt)/std) generated,
    4. posmask and negmask calculated by clips MB_[ADD/RAT/STD]_[U/L] and
      logic MB_[N/P]LOG (see keywords),
    5. Pixels of (posmask | negmask) are repleced with median filtered frame.

    """
    from scipy.ndimage import median_filter

    def _sanitize_clips(clips):
        clips = np.atleast_1d(clips)
        if clips.size == 1:
            clips = np.repeat(clips, 2)
        return clips

    if ((med_sub_clip is None) and (med_rat_clip is None) and (std_rat_clip is None)):
        warn("No BPM is found because all clips are None.", end=' ')
        if full:
            return ccd, dict(posmask=None, negmask=None, med_filt=None,
                             med_sub=None, med_rat=None, std_rat=None, std=None)
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

    medfilt_kw = dict(size=size, footprint=footprint, mode=mode, cval=cval, origin=origin)

    _t = Time.now()
    med_filt = median_filter(arr, **medfilt_kw)

    if update_header:
        cmt2hdr(
            hdr, 'h', verbose=verbose, t_ref=_t,
            s=f"Median filtered (convolved) frame calculated with {medfilt_kw}"
        )

    if std_model == 'ccd':
        _t = Time.now()
        gain = change_to_quantity(gain, u.electron/u.adu, to_value=True)
        rdnoise = change_to_quantity(rdnoise, u.electron, to_value=True)

        std = np.sqrt((1 + snoise)*med_filt/gain + (rdnoise/gain)**2)
        if update_header:
            cmt2hdr(
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
            hdr['MB_MODEL'] = (std_model,
                               "Method used for getting stdev map")
            hdr["MB_SSKY"] = (std,
                              "Sky stdev for median filter BPM (MBPM)")
            hdr["MB_SSECT"] = (f"{std_section}",
                               "Sky stdev calculation section in MBPM")
            cmt2hdr(
                hdr, 'h', verbose=verbose, t_ref=_t,
                s=("Sky standard deviation (MB_SSKY) calculated by sigma clipping at "
                   + f"MB_SSECT with {sigclip_kw}; used for std_ratio map calculation.")
            )

    elif isinstance(std_model, np.ndarray):
        if std_model.shape != ccd.data.shape:
            raise ValueError(f"std_model.shape (= {std_model.shape}) differs from "
                             + f"ccd.shape ({ccd.data.shape}")
        std = std_model
        if update_header:
            hdr['MB_MODEL'] = ("User input array", "Method used for getting stdev map")

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
        hdr["MB_NLOGI"] = (_LOGICAL_STR[0],
                           "The logic used for negative MBPM masks (and/or)")
        hdr["MB_PLOGI"] = (_LOGICAL_STR[1],
                           "The logic used for positive MBPM masks (and/or)")
        hdr["MB_RAT_U"] = (med_rat_clip[1],
                           "Upper clip of (data/|medfilt|) map (MBPM)")
        hdr["MB_RAT_L"] = (med_rat_clip[0],
                           "Lower clip of (data/|medfilt|) map (MBPM)")
        hdr["MB_SUB_U"] = (med_sub_clip[1],
                           "Upper clip of (data-medfilt) map (MBPM)")
        hdr["MB_SUB_L"] = (med_sub_clip[0],
                           "Lower clip of (data-medfilt) map (MBPM)")
        hdr["MB_STD_U"] = (std_rat_clip[1],
                           "Upper clip of (data-medfilt)/std map (MBPM)")
        hdr["MB_STD_L"] = (std_rat_clip[0],
                           "Lower clip of (data-medfilt)/std map (MBPM)")

        cmt2hdr(
            hdr, 'h', verbose=verbose, t_ref=_t,
            s="Median-filter based Bad-Pixel Masking (MBPM) applied."
        )
        # cmt2hdr(
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
def bdf_process(
        ccd,
        output=None,
        extension=None,
        mbiaspath=None,
        mdarkpath=None,
        mflatpath=None,
        mfringepath=None,
        mbias=None,
        mdark=None,
        mflat=None,
        mfringe=None,
        fringe_flat_fielded=True,
        fringe_scale_fun=np.mean,
        fringe_scale_section=None,
        trim_fits_section=None,
        calc_err=False,
        unit=None,
        gain=None,
        gain_key="GAIN",
        gain_unit=u.electron/u.adu,
        rdnoise=None,
        rdnoise_key="RDNOISE",
        rdnoise_unit=u.electron,
        exposure_key="EXPTIME",
        exposure_unit=u.s,
        fringe_exposure=None,
        dark_exposure=None,
        data_exposure=None,
        dark_scale=False,
        normalize_exposure=False,
        normalize_average=False,
        normalize_median=False,
        flat_min_value=None,
        flat_norm_value=1.,
        do_crrej=False,
        crrej_kwargs=None,
        propagate_crmask=False,
        verbose_crrej=False,
        verbose_bdf=True,
        output_verify='fix',
        overwrite=True,
        dtype="float32",
        uncertainty_dtype="float32"
):
    """ Do bias, dark and flat process.
    Parameters
    ----------
    ccd : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
        The ccd to be processed.

    output : path-like or None, optional.
        The path if you want to save the resulting `ccd` object.
        Default: `None`.

    mbiaspath, mdarkpath, mflatpath, mfringepath : path-like, optional.
        The path to master bias, dark, flat, and fringe FITS files. If `None`,
        the corresponding process is not done. These can be provided in
        addition to `mbias`, `mdark`, `mflat`, and/or `mfringe`.

    mbias, mdark, mflat, mfringe : CCDData, optional.
        The master bias, dark, and flat in `~astropy.nddata.CCDData`. If this
        is given, the files provided by `mbiaspath`, `mdarkpath`, `mflatpath`
        and/or `mfringe` are **not** loaded, but these paths will be used for
        header (``BIASFRM``, ``DARKFRM``, ``FLATFRM`` and/or ``FRINFRM``). If
        the paths are not given, the header values will be ``<User>``.

    fringe_scale_fun : function object, {"exp", "exposure", "exptime"}, optional.
        The function to be used to scale the fringe before subtraction,
        specified by the region `fringe_scale_section`. If one of ``{"exp",
        "exposure", "exptime"}``, the exposure time from `exposure_key` is used
        for scaling. Scaling is turned of if `fringe_scale_section` is `None`.
        Default: `np.mean`.

    fringe_scale_section : str, optional.
        The FITS-convention section of the fringe and object (science) frames
        to match the fringe pattern before the subtraction. If `None`, this
        scaling is turned off. To use all region, use such as ``'[:, :]`` for
        2-D.
        default: `None`.

    fringe_flat_fielded : bool, optional.
        Whether the fringe frame is flat-fielded. If `True`, fringe is
        subtracted AFTER flat-fielding the input frame. Otherwise (default),
        fringe is subtracted BEFORE flat-fielding the input frame.

    trim_fits_section: str, optional.
        Region of `ccd` to be trimmed; see `~ccdproc.subtract_overscan` for
        details. Default is `None`.

    calc_err : bool, optional.
        Whether to calculate the error map based on Poisson and readnoise error
        propagation.

        ..note::
            Currently it's encouraged to make error-map manually, as the API is
            not stable.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is `None`.

    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. These are all ignored if
        ``calc_err=False`` and ``do_crrej=False``. If ``calc_err=True``, it
        automatically seeks for suitable gain and readnoise value. If `gain` or
        `readnoise` is specified, they are interpreted with `gain_unit` and
        `rdnoise_unit`, respectively. If they are not specified, this function
        will seek for the header with keywords of `gain_key` and `rdnoise_key`,
        and interprete the header value in the unit of `gain_unit` and
        `rdnoise_unit`, respectively.

    gain_key, rdnoise_key : str, optional.
        See `gain`, `rdnoise` explanation above.
        These are all ignored if ``calc_err=False``.

    gain_unit, rdnoise_unit : str, astropy.Unit, optional.
        See `gain`, `rdnoise` explanation above.
        These are all ignored if ``calc_err=False``.

    dark_exposure, data_exposure : None, float, astropy Quantity, optional.
        The exposure times of dark and data frame, respectively. They should
        both be specified or both `None`. These are all ignored if
        ``mdarkpath=None``. If both are not specified while `mdarkpath` is
        given, then the code automatically seeks for header's `exposure_key`.
        Then interprete the value as the quantity with unit `exposure_unit`. If
        `mdkarpath` is not `None`, then these are passed to
        `~ccdproc.subtract_dark`.

    exposure_key : str, optional.
        The header keyword for exposure time.
        Ignored if ``mdarkpath=None``.

    exposure_unit : astropy Unit, optional.
        The unit of the exposure time.
        Ignored if ``mdarkpath=None``.

    normalize_exposure : bool, optional.
        Whether to normalize the values by the exposure time of each frame.
        Maybe useful for long exposure darks to make 1-sec darks.
        Default is `False`.

    normalize_average, normalize_median : bool, optional.
        Whether to normalize the values by the average or median value of each
        frame before combining. Only up to one of these must be True. Maybe
        useful for flat.
        Default is `False`.

    flat_min_value : float or None, optional.
        min_value of `ccdproc.flat_correct`. Minimum value for flat field. The
        value can either be None and no minimum value is applied to the flat or
        specified by a float which will replace all values in the flat by the
        min_value.
        Default is `None`.

    flat_norm_value : float or None, optional.
        The norm_value of `ccdproc.flat_correct`. If `None`, the flat is
        internally normalized by its mean before the flat correction, i.e., the
        flat correction will be like ``image/flat*mean(flat)``.
        If not `None`, the flat correction will be like
        ``image/flat*flat_norm_value``. Default is 1 (**different** from
        `ccdproc` which uses `None` as default).

    crrej_kwargs : dict or None, optional.
        If `None` (default), uses some default values (see `crrej`). It is
        always discouraged to use default except for quick validity-checking,
        because even the official L.A. Cosmic codes in different versions
        (IRAF, IDL, Python, etc) have different default parameters, i.e., there
        is nothing which can be regarded as _the default_. To see all possible
        keywords, do ``print(astroscrappy.detect_cosmics.__doc__)`` Also refer
        to
        https://nbviewer.jupyter.org/github/ysbach/AO2019/blob/master/Notebooks/07-Cosmic_Ray_Rejection.ipynb

    propagate_crmask : bool, optional.
        Whether to save (propagate) the mask from CR rejection (`astroscrappy`)
        to the CCD's mask. Default is `False`.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``. May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    dtype : str or `numpy.dtype` or None, optional.
        Allows user to set dtype. See `numpy.array` `dtype` parameter
        description. If `None` it uses ``np.float64``.
        Default is `None`.
    """
    # Set strings for header history & print (if verbose)
    str_bias = "Bias subtracted (see BIASFRM)"
    str_dark = "Dark subtracted (see DARKFRM)"
    str_dscale = "Dark scaling using {}"
    str_flat = "Flat corrected by image/flat*flat_norm_value (see FLATFRM; FLATNORM)"
    str_fringe_noscale = "Fringe subtracted (see FRINFRM)"
    str_fringe_scale = ("Finge subtracted with scaling (image - scale*fringe)"
                        + "(see FRINFRM, FRINSECT, FRINFUNC and FRINSCAL)")
    str_trim = "Trim by FITS section {} (see LTV, LTM, TRIMIM)"
    str_e0 = "Readnoise propagated with Poisson noise (using gain above) of source."
    str_ed = "Poisson noise from subtracted dark was propagated."
    str_ef = "Flat uncertainty was propagated."
    str_nexp = "Normalized by the exposure time."
    str_navg = "Normalized by the average value of the frame."
    str_nmed = "Normalized by the median value of the frame."

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

    def _sub_frin(proc, mfringe, fringe_scale_fun=np.mean, fringe_scale_section=None):
        _t = Time.now()
        if fringe_scale_section is not None:
            if isinstance(fringe_scale_fun, str):
                if fringe_scale_fun.lower() in ["exp", "exposure", "exptime"]:
                    scale = (float(proc.header.get(exposure_key, data_exposure))
                             / float(mfringe.header.get(exposure_key, fringe_exposure)))
                else:
                    raise ValueError(
                        "If `fringe_scale_fun` is str, it must be one of "
                        + f"{'exp', 'exposure', 'exptime'}. Now it's {fringe_scale_fun}"
                    )
            else:
                sl = fitsxy2py(fringe_scale_section)
                scale = fringe_scale_fun(proc.data[sl]) / fringe_scale_fun(mfringe.data[sl])
            proc.header["FRINSCAL"] = (scale, "Scale factor multiplied to fringe")
            s = str_fringe_scale
        else:
            scale = 1  # Not 1.0 so that (ccd in int) - (mfringe in int) remains int.
            s = str_fringe_noscale
        mfringe.data *= scale
        proc.data = proc.subtract(mfringe).data  # should I calc. error..?
        cmt2hdr(proc.header, 'h', s, verbose=verbose_bdf, t_ref=_t)
        return proc

    # ************************************************************************************ #
    # *                                  INITIAL SETTING                                 * #
    # ************************************************************************************ #
    # if not isinstance(ccd, CCDData):
    #     raise TypeError(f"ccd must be CCDData (now it is {type(ccd)})")
    ccd, _, _ = _parse_image(ccd, extension=extension, force_ccddata=True)
    PROCESS = []
    proc = ccd.copy()

    # == Log the CCDPROC version ========================================================== #
    if "CCDPROCV" in proc.header:
        if str(proc.header["CCDPROCV"]) != str(ccdproc.__version__):
            cmt2hdr(proc.header, "h",
                    ("The ccdproc version prior to this modification was "
                     + f"{proc.header['CCDPROCV']}."))
            proc.header["CCDPROCV"] = (ccdproc.__version__,
                                       "ccdproc version used for processing.")
        # else (no version change): do nothing.
    else:
        proc.header["CCDPROCV"] = (ccdproc.__version__,
                                   "ccdproc version used for processing.")

    # == Set for BIAS ==================================================================== #
    do_bias, mbias, mbiaspath = _load_master(mbiaspath, mbias)
    if do_bias:
        PROCESS.append("B")
        proc.header["BIASFRM"] = (str(mbiaspath), "Applied bias frame")

    # == Set for DARK ==================================================================== #
    do_dark, mdark, mdarkpath = _load_master(mdarkpath, mdark)

    if do_dark:
        PROCESS.append("D")
        proc.header["DARKFRM"] = (str(mdarkpath), "Applied dark frame")

        if dark_scale:
            # TODO: what if dark_exposure, data_exposure are given explicitly?
            cmt2hdr(proc.header, 'h', str_dscale.format(exposure_key), verbose=verbose_bdf)

    # == Set for FLAT ==================================================================== #
    do_flat, mflat, mflatpath = _load_master(mflatpath, mflat)
    if do_flat:
        PROCESS.append("F")
        proc.header["FLATFRM"] = (str(mflatpath),
                                  "Applied flat frame")
        proc.header["FLATNORM"] = (flat_norm_value,
                                   "flat_norm_value (none = mean of input flat)")

    # == Set for FRINGE ================================================================== #
    do_fringe, mfringe, mfringepath = _load_master(mfringepath, mfringe)
    if do_fringe:
        PROCESS.append("Fr")
        proc.header["FRINFRM"] = (str(mfringepath), "Applied fringe frame")
        if fringe_scale_section is not None:
            proc.header["FRINSECT"] = (fringe_scale_section,
                                       "FITS section used for scaling fringe")
            proc.header["FRINFUNC"] = (fringe_scale_fun.__name__,
                                       "Function used for fringe scaling")

    # == Set gain and rdnoise if at least one of calc_err and do_crrej is True. ========== #
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

    # ************************************************************************************ #
    # *                                 RUN PREPROCESSING                                * #
    # ************************************************************************************ #
    # == Do TRIM ========================================================================= #
    if trim_fits_section is not None:
        _t = Time.now()
        sect = dict(fits_section=trim_fits_section, update_header=False)
        proc = trim_ccd(proc, **sect)
        mbias = trim_ccd(mbias, **sect) if do_bias else None
        mdark = trim_ccd(mdark, **sect) if do_dark else None
        mflat = trim_ccd(mflat, **sect) if do_flat else None
        mfringe = trim_ccd(mfringe, **sect) if do_fringe else None
        PROCESS.append("T")

        cmt2hdr(proc.header, 'h', str_trim.format(trim_fits_section), verbose=verbose_bdf, t_ref=_t)

    # == Do BIAS ========================================================================= #
    if do_bias:
        _t = Time.now()
        proc = subtract_bias(proc, mbias)
        cmt2hdr(proc.header, 'h', str_bias.format(mbiaspath), verbose=verbose_bdf, t_ref=_t)

    # == Do DARK ========================================================================= #
    if do_dark:
        _t = Time.now()
        if dark_scale and data_exposure is None:
            try:
                data_exposure = proc.header[exposure_key]
            except (KeyError, AttributeError) as e:
                raise e("Dark must be scaled but data's exposure time "
                        + f"({exposure_key}) is not found from.")
        if dark_scale and dark_exposure is None:
            try:
                dark_exposure = mdark.header[exposure_key]
            except (KeyError, AttributeError) as e:
                raise e("Dark must be scaled but dark's exposure time "
                        + f"({exposure_key}) is not found from.")

        proc = subtract_dark(proc,
                             mdark,
                             dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
        cmt2hdr(proc.header, 'h', str_dark.format(mdarkpath), verbose=verbose_bdf, t_ref=_t)

    # == Set for uncertainty ============================================================= #
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
        cmt2hdr(proc.header, 'h', s, verbose=verbose_bdf, t_ref=_t)

    # == Do FRINGE **before** flat if not `fringe_flat_fielded` ========================== #
    if do_fringe and not fringe_flat_fielded:
        proc = _sub_frin(proc, mfringe, fringe_scale_fun=fringe_scale_fun,
                         fringe_scale_section=fringe_scale_section)

    # == Do FLAT ========================================================================= #
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

        cmt2hdr(proc.header, 'h', s, verbose=verbose_bdf, t_ref=_t)

    # == Do FRINGE **after** flat if `fringe_flat_fielded` =============================== #
    if do_fringe and fringe_flat_fielded:
        proc = _sub_frin(proc, mfringe, fringe_scale_fun=fringe_scale_fun,
                         fringe_scale_section=fringe_scale_section)

    # == Normalization =================================================================== #
    if normalize_exposure:
        _t = Time.now()
        if data_exposure is None:
            data_exposure = proc.header[exposure_key]
        proc.data = proc.data/data_exposure  # uncertainty will also be..
        cmt2hdr(proc.header, 'h', str_nexp, verbose=verbose_bdf, t_ref=_t)

    if normalize_average:
        _t = Time.now()
        avg = np.mean(proc.data)
        proc.data = proc.data/avg
        cmt2hdr(proc.header, 'h', str_navg, verbose=verbose_bdf, t_ref=_t)

    if normalize_median:
        _t = Time.now()
        med = np.median(proc.data)
        proc.data = proc.data/med
        cmt2hdr(proc.header, 'h', str_nmed, verbose=verbose_bdf, t_ref=_t)

    # == Do CRREJ ======================================================================== #
    if do_crrej:
        if crrej_kwargs is None:
            crrej_kwargs = {}
            warn("You are not specifying CR-rejection parameters! It can be"
                 + " dangerous to use defaults blindly.")

        _proc = proc.header["PROCESS"]
        if (("B" in _proc) + ("D" in _proc) + ("F" in _proc)) < 2:
            warn(
                "L.A. Cosmic should be run AFTER bias, dark, flat process. "
                + f"You have only done {proc.header['PROCESS']}. "
                + "See http://www.astro.yale.edu/dokkum/lacosmic/notes.html"
            )

        proc, _ = crrej(
            proc,
            propagate_crmask=propagate_crmask,
            update_header=True,
            gain=gain_Q,        # already parsed from local variables above
            rdnoise=rdnoise_Q,  # already parsed from local variables above
            verbose=verbose_crrej,
            **crrej_kwargs)

    # ************************************************************************************ #
    # *                                  PREPARE OUTPUT                                  * #
    # ************************************************************************************ #
    # To avoid ``pssl`` in cr rejection, subtract fringe AFTER the CRREJ.
    proc = CCDData_astype(proc, dtype=dtype, uncertainty_dtype=uncertainty_dtype)
    update_process(proc.header, PROCESS, key="PROCESS", delimiter='-')
    update_tlm(proc.header)

    if output is not None:
        if verbose_bdf:
            print(f"Writing FITS to {output}... ", end='')
        proc.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose_bdf:
            print("Saved.")
    return proc


def run_reduc_plan(
        plan,
        output=None,
        extension=None,
        col_file="file",
        col_bias="BIASFRM",
        col_dark="DARKFRM",
        col_flat="FLATFRM",
        col_mask="MASKFILE",
        col_fringe="FRINFRM",
        fixpix_kw=dict(priority=None, verbose=False),
        do_crrej=False,
        preload_cals=False,
        return_ccd=False,
        verbose=False,
):
    """Run reduction(preprocessing) based on the planner.

    Parameters
    ----------
    plan : `~pandas.DataFrame`

    col_bias, col_dark, col_flat, col_mask, col_fringe : str, optional
        The column names for bias, dark, flat, mask, and fringe frames in
        `plan`. Default values are set following IRAF convention.

    preload_cals : bool, optional
        Whether to pre-load all the calibration frames. This reduces file I/O
        time if same file have to be loaded for multiple times. Turn it off
        when too many calibration frames are there (so that memery cannot hold
        them).
        Default: `False`.

    verbose : bool, optional
        [description], by default False
    verbose_bdf : bool, optional
        [description], by default True
    """
    if not isinstance(plan, pd.DataFrame):
        raise TypeError("plan must be a pandas.DataFrame.")

    if output is None and not return_ccd:
        raise ValueError("No output file and return_ccd is False. "
                         + "Nothing will be saved and nothing will be returned.")

    calframes = {}
    output = [None]*len(plan) if output is None else listify(output)

    if len(output) != len(plan):
        raise ValueError("output must have the same length as the plan.")

    if preload_cals:
        for key, col in zip(["mbias", "mdark", "mflat", "mfringe", "mask"],
                            [col_bias, col_dark, col_flat, col_fringe, col_mask]):
            if col in plan.columns:
                calframes[key] = {fpath: load_ccd(fpath) for fpath in plan[col].unique()}
            else:
                calframes[key] = {}  # to be able to use `.get` later

    if return_ccd:
        ccds = []

    for (_, row), outpath in zip(plan.iterrows(), output):
        ccd = load_ccd(row[col_file])
        # ^ Better to load as CCDData here rather than pass filepath to
        # `bdf_process`, to avoid parsing overhead in `_parse_image`.

        mbiaspath = row.get(col_bias)
        mdarkpath = row.get(col_dark)
        mflatpath = row.get(col_flat)
        mfringepath = row.get(col_fringe)
        nccd = bdf_process(
            ccd,
            extension=extension,
            mbias=calframes["mbias"].get(mbiaspath),  # = None if not preload_cals
            mdark=calframes["mdark"].get(mdarkpath),  # = None if not preload_cals
            mflat=calframes["mflat"].get(mflatpath),  # = None if not preload_cals
            mfringe=calframes["mfringe"].get(mfringepath),  # = None if not preload_cals
            mbiaspath=mbiaspath,
            mdarkpath=mdarkpath,
            mflatpath=mflatpath,
            mfringepath=mfringepath,
        )
        maskpath = row.get(col_mask)
        mask = None if maskpath is None else load_ccd(maskpath)

        if do_crrej:
            nccd = crrej(
                nccd,
                mask=mask,

            )

        nccd = fixpix(nccd, mask, maskpath=maskpath, **fixpix_kw)

        if outpath is not None:
            nccd.write(outpath, overwrite=True, output_verify="fix")
        if return_ccd:
            ccds.append(nccd)
