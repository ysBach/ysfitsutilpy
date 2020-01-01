from warnings import warn

import astroscrappy
import ccdproc
import numpy as np
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astroscrappy import detect_cosmics
from ccdproc import flat_correct, subtract_bias, subtract_dark

from .ccdutil import (CCDData_astype, datahdr_parse, load_ccd, make_errmap,
                      propagate_ccdmask, set_ccd_gain_rdnoise, trim_ccd)
from .hdrutil import add_to_header
from .misc import LACOSMIC_KEYS, change_to_quantity

__all__ = [
    "crrej", "bdf_process"]

# Set strings for header history & print (if verbose)
str_bias = "Bias subtracted (see BIASPATH)"
str_dark = "Dark subtracted (see DARKPATH)"
str_dscale = "Dark scaling {} using {}"
str_flat = "Flat corrected (see FLATPATH)"
str_trim = "Trim by FITS section {} (see LTV, LTM, TRIMIM)"
str_e0 = ("Readnoise propagated with Poisson noise (using gain above)"
          + " of source.")
str_ed = "Poisson noise from subtracted dark was propagated."
str_ef = "Flat uncertainty was propagated."
str_nexp = "Normalized by the exposure time."
str_navg = "Normalized by the average value of the frame."
str_nmed = "Normalized by the median value of the frame."
str_cr = ("Cosmic-Ray rejected by astroscrappy (v {}), "
          + "with parameters: {}")


def _add_and_print(s, header, verbose, update_header=True):
    if update_header:
        # add as history
        add_to_header(header, 'h', s)
    if verbose:
        print(s)


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
          gain=None, rdnoise=None, verbose=True,
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
        Whether to save (propagate) the mask from CR rejection
        (``astroscrappy``) to the CCD's mask.
        Default is ``False``.

    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. If not ``Quantity``, they must be
        in electrons per adu and electron unit, respectively.

    sigclip : float, optional
        Laplacian-to-noise limit for cosmic ray detection. Lower values
        will flag more pixels as cosmic rays. Default: 4.5.

    sigfrac : float, optional
        Fractional detection limit for neighboring pixels. For cosmic
        ray neighbor pixels, a lapacian-to-noise detection limit of
        sigfrac * sigclip will be used. Default: 0.5.

    objlim : float, optional
        Minimum contrast between Laplacian image and the fine structure
        image. Increase this value if cores of bright stars are flagged
        as cosmic rays. Default: 1.0.

    pssl : float, optional
        Previously subtracted sky level in ADU. We always need to work
        in electrons for cosmic ray detection, so we need to know the
        sky level that has been subtracted so we can add it back in.
        Default: 0.0.

    satlevel : float, optional
        Saturation of level of the image (electrons). This value is used
        to detect saturated stars and pixels at or above this level are
        added to the mask. Default: ``np.inf``.

    niter : int, optional
        Number of iterations of the LA Cosmic algorithm to perform.
        Default: 4.

    sepmed : boolean, optional
        Use the separable median filter instead of the full median
        filter. The separable median is not identical to the full median
        filter, but they are approximately the same and the separable
        median filter is significantly faster and still detects cosmic
        rays well. Default: True

    cleantype : {'median', 'medmask', 'meanmask', 'idw'}, optional
        Set which clean algorithm is used:

        * ``'median'``: An umasked 5x5 median filter
        * ``'medmask'``: A masked 5x5 median filter
        * ``'meanmask'``: A masked 5x5 mean filter
        * ``'idw'``: A masked 5x5 inverse distance weighted interpolation

        Default: ``"meanmask"``.

    fsmode : {'median', 'convolve'}, optional
        Method to build the fine structure image:
        * ``'median'``: Use the median filter in the standard LA Cosmic
        algorithm
        * ``'convolve'``: Convolve the image with the psf kernel
        to calculate the fine structure image.
        Default: ``'median'``.

    psfmodel : {'gauss', 'gaussx', 'gaussy', 'moffat'}, optional
        Model to use to generate the psf kernel if ``fsmode='convolve'``
        and ``psfk`` is ``None``. The current choices are Gaussian and
        Moffat profiles. ``'gauss'`` and ``'moffat'`` produce circular
        PSF kernels. The ``'gaussx'`` and ``'gaussy'`` produce Gaussian
        kernels in the x and y directions respectively. Default:
        ``"gauss"``.

    psffwhm : float, optional
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.

    psfsize : int, optional
        Size of the kernel to calculate. Returned kernel will have size
        psfsize x psfsize. psfsize should be odd. Default: 7.

    psfk : float numpy array, optional
        PSF kernel array to use for the fine structure image if
        ``fsmode='convolve'``. If ``None`` and ``fsmode == 'convolve'``,
        we calculate the psf kernel using ``'psfmodel'``. Default:
        ``None``.

    psfbeta : float, optional
        Moffat beta parameter. Only used if ``fsmode='convolve'`` and
        ``psfmodel='moffat'``. Default: 4.765.

    verbose : boolean, optional
        Print to the screen or not. Default: ``False``.

    Notes
    -----
    All defaults are based on IRAF version of L.A. Cosmic (Note the
    default parameters of L.A. Cosmic differ from version to version, so
    I took the IRAF version written by van Dokkum.)
    See the docstring of astroscrappy by
    >>> import astroscrappy
    >>> astroscrappy.detect_cosmics?

    Example
    -------
    >>> yfu.ccdutil.set_ccd_gain_rdnoise(ccd)
    >>> nccd, mask = crrej(ccd)
    """
    if gain is None:
        try:
            gain = ccd.gain
        except AttributeError:
            raise ValueError(
                "Gain must be given or accessible as ``ccd.gain``. "
                "Use, e.g., yfu.set_ccd_gain_rdnoise(ccd).")

    if rdnoise is None:
        try:
            rdnoise = ccd.rdnoise
        except AttributeError:
            raise ValueError(
                "Readnoise must be given or accessible as ``ccd.rdnoise``. "
                "Use, e.g., yfu.set_ccd_gain_rdnoise(ccd).")

    _ccd = ccd.copy()
    data, hdr = datahdr_parse(_ccd)
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
        **crrej_kwargs)

    # create the new ccd data object
    #   astroscrappy automatically does the gain correction, so return
    #   back to avoid confusion.
    _ccd.data = cleanarr / gain
    if propagate_crmask:
        _ccd.mask = propagate_ccdmask(_ccd, additional_mask=crmask)

    try:
        hdr["PROCESS"] += "C"
    except KeyError:
        hdr["PROCESS"] = "C"

    _add_and_print(str_cr.format(astroscrappy.__version__, crrej_kwargs),
                   hdr, verbose, update_header=update_header)
    _ccd.header = hdr

    return _ccd, crmask


# NOTE: crrej should be done AFTER bias/dark and flat correction:
# http://www.astro.yale.edu/dokkum/lacosmic/notes.html
def bdf_process(ccd, output=None,
                mbiaspath=None, mdarkpath=None, mflatpath=None,
                trim_fits_section=None, calc_err=False, unit='adu',
                gain=None, gain_key="GAIN", gain_unit=u.electron/u.adu,
                rdnoise=None, rdnoise_key="RDNOISE", rdnoise_unit=u.electron,
                dark_exposure=None, data_exposure=None, exposure_key="EXPTIME",
                exposure_unit=u.s, dark_scale=False,
                normalize_exposure=False,
                normalize_average=False, normalize_median=False,
                flat_min_value=None, flat_norm_value=None,
                do_crrej=False, crrej_kwargs=None, propagate_crmask=False,
                verbose_crrej=False, verbose_bdf=True, output_verify='fix',
                overwrite=True, dtype="float32", uncertainty_dtype="float32"):
    ''' Do bias, dark and flat process.
    Parameters
    ----------
    ccd: CCDData
        The ccd to be processed.

    output : path-like or None, optional.
        The path if you want to save the resulting ``ccd`` object.
        Default is ``None``.

    mbiaspath, mdarkpath, mflatpath : path-like, optional.
        The path to master bias, dark, flat FITS files. If ``None``, the
        corresponding process is not done.

    trim_fits_section: str, optional.
        Region of ``ccd`` to be trimmed; see
        ``ccdproc.subtract_overscan`` for details. Default is ``None``.

    calc_err : bool, optional.
        Whether to calculate the error map based on Poisson and
        readnoise error propagation.
        * NOTE: Currently it's encouraged to make error-map manually, as
        the API is not stable.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is ``'adu'``.

    gain, rdnoise : None, float, astropy.Quantity, optional.
        The gain and readnoise value. These are all ignored if
        ``calc_err=False`` and ``do_crrej=False``. If ``calc_err=True``,
        it automatically seeks for suitable gain and readnoise value. If
        ``gain`` or ``readnoise`` is specified, they are interpreted
        with ``gain_unit`` and ``rdnoise_unit``, respectively. If they
        are not specified, this function will seek for the header with
        keywords of ``gain_key`` and ``rdnoise_key``, and interprete the
        header value in the unit of ``gain_unit`` and ``rdnoise_unit``,
        respectively.

    gain_key, rdnoise_key : str, optional.
        See ``gain``, ``rdnoise`` explanation above.
        These are all ignored if ``calc_err=False``.

    gain_unit, rdnoise_unit : str, astropy.Unit, optional.
        See ``gain``, ``rdnoise`` explanation above.
        These are all ignored if ``calc_err=False``.

    dark_exposure, data_exposure : None, float, astropy Quantity, optional.
        The exposure times of dark and data frame, respectively. They
        should both be specified or both ``None``. These are all ignored
        if ``mdarkpath=None``. If both are not specified while
        ``mdarkpath`` is given, then the code automatically seeks for
        header's ``exposure_key``. Then interprete the value as the
        quantity with unit ``exposure_unit``.

        If ``mdkarpath`` is not ``None``, then these are passed to
        ``ccdproc.subtract_dark``.

    exposure_key : str, optional.
        The header keyword for exposure time.
        Ignored if ``mdarkpath=None``.

    exposure_unit : astropy Unit, optional.
        The unit of the exposure time.
        Ignored if ``mdarkpath=None``.

    normalize_exposure : bool, optional.
        Whether to normalize the values by the exposure time of each
        frame. Maybe useful for long exposure darks to make 1-sec darks.
        Default is ``False``.

    normalize_average, normalize_median : bool, optional.
        Whether to normalize the values by the average or median value
        of each frame before combining. Only up to one of these must be
        True. Maybe useful for flat.
        Default is ``False``.

    flat_min_value : float or None, optional.
        min_value of `ccdproc.flat_correct`.
        Minimum value for flat field. The value can either be None and
        no minimum value is applied to the flat or specified by a float
        which will replace all values in the flat by the min_value.
        Default is ``None``.

    flat_norm_value : float or None, optional.
        norm_value of `ccdproc.flat_correct`.
        If not ``None``, normalize flat field by this argument rather
        than the mean of the image. This allows fixing several different
        flat fields to have the same scale. If this value is negative or
        0, a ``ValueError``
        is raised. Default is ``None``.

    crrej_kwargs : dict or None, optional.
        If ``None`` (default), uses some default values (see ``crrej``).
        It is always discouraged to use default except for quick
        validity-checking, because even the official L.A. Cosmic codes
        in different versions (IRAF, IDL, Python, etc) have different
        default parameters, i.e., there is nothing which can be regarded
        as _the default_. To see all possible keywords, do
        ``print(astroscrappy.detect_cosmics.__doc__)`` Also refer to
        https://nbviewer.jupyter.org/github/ysbach/AO2019/blob/master/Notebooks/07-Cosmic_Ray_Rejection.ipynb

    propagate_crmask : bool, optional.
        Whether to save (propagate) the mask from CR rejection
        (``astroscrappy``) to the CCD's mask.
        Default is ``False``.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``,
        ``"silentfix"``, ``"ignore"``, ``"warn"``, or ``"exception"``.
        May also be any combination of ``"fix"`` or ``"silentfix"`` with
        ``"+ignore"``, ``+warn``, or ``+exception" (e.g.
        ``"fix+warn"``).  See the astropy documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    dtype : str or `numpy.dtype` or None, optional.
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description. If ``None`` it uses ``np.float64``.
        Default is ``None``.
    '''

    # Initial setting
    if not isinstance(ccd, CCDData):
        raise TypeError(f"ccd must be CCDData (now it is {type(ccd)})")
    proc = ccd.copy()
    hdr_new = proc.header

    # Add PROCESS key
    try:
        _ = hdr_new["PROCESS"]
    except KeyError:
        hdr_new["PROCESS"] = ("", "The processed history: see comment.")
        hdr_new["CCDPROCV"] = (ccdproc.__version__,
                               "ccdproc version used for processing.")
        hdr_new.add_comment("PROCESS key can be B (bias), D (dark), F (flat), "
                            + "T (trim), W (WCS astrometry), C(CRrej).")

    # Set for BIAS
    if mbiaspath is None:
        do_bias = False
        mbias = CCDData(np.zeros_like(ccd), unit=unit)
    else:
        do_bias = True
        mbias = load_ccd(mbiaspath, unit=unit)
        if not calc_err:
            mbias.uncertainty = None
        hdr_new["PROCESS"] += "B"
        hdr_new["BIASNAME"] = (str(mbiaspath), "Path to the used bias file")
        _add_and_print(str_bias.format(mbiaspath), hdr_new, verbose_bdf)

    # Set for DARK
    if mdarkpath is None:
        do_dark = False
        mdark = CCDData(np.zeros_like(ccd), unit=unit)
    else:
        do_dark = True
        mdark = load_ccd(mdarkpath, unit=unit)
        if not calc_err:
            mdark.uncertainty = None
        hdr_new["PROCESS"] += "D"
        hdr_new["DARKPATH"] = (str(mdarkpath), "Path to the used dark file")
        _add_and_print(str_dark.format(mdarkpath), hdr_new, verbose_bdf)

        if dark_scale:
            _add_and_print(str_dscale.format(dark_scale, exposure_key),
                           hdr_new, verbose_bdf)

    # Set for FLAT
    if mflatpath is None:
        do_flat = False
        mflat = CCDData(np.ones_like(ccd), unit=unit)
    else:
        do_flat = True
        mflat = load_ccd(mflatpath, unit=unit)
        if not calc_err:
            mflat.uncertainty = None
        hdr_new["PROCESS"] += "F"
        hdr_new["FLATPATH"] = (str(mflatpath), "Path to the used flat file")
        _add_and_print(str_flat.format(mflatpath), hdr_new, verbose_bdf)

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
        proc = trim_ccd(proc, fits_section=trim_fits_section)
        mbias = trim_ccd(mbias, fits_section=trim_fits_section)
        mdark = trim_ccd(mdark, fits_section=trim_fits_section)
        mflat = trim_ccd(mflat, fits_section=trim_fits_section)
        hdr_new["PROCESS"] += "T"

        _add_and_print(str_trim.format(trim_fits_section),
                       hdr_new, verbose_bdf)

    # Do BIAS
    if do_bias:
        proc = subtract_bias(proc, mbias)

    # Do DARK
    if do_dark:
        proc = subtract_dark(proc,
                             mdark,
                             dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)

    # Make UNCERT extension before doing FLAT
    #   It is better to make_errmap a priori because of mathematical and
    #   computational convenience. See ``if do_flat:`` clause below.
    if calc_err:
        err = make_errmap(proc,
                          gain_epadu=gain,
                          subtracted_dark=mdark)

        proc.uncertainty = StdDevUncertainty(err)
        _add_and_print(str_e0, hdr_new, verbose_bdf)

        if do_dark:
            _add_and_print(str_ed, hdr_new, verbose_bdf)

    # Do FLAT
    if do_flat:
        # Flat error propagation is done automatically by
        # ``ccdproc.flat_correct``if it has the uncertainty attribute.
        proc = flat_correct(proc,
                            mflat,
                            min_value=flat_min_value,
                            norm_value=flat_norm_value)

        if calc_err and mflat.uncertainty is not None:
            _add_and_print(str_ef, hdr_new, verbose_bdf)

    # Normalize by the exposure time (e.g., ADU per sec)
    if normalize_exposure:
        if data_exposure is None:
            data_exposure = hdr_new[exposure_key]
        proc = proc.divide(data_exposure)  # uncertainty will also be..
        _add_and_print(str_nexp, hdr_new, verbose_bdf)

    # Normalize by the mean value
    if normalize_average:
        avg = np.mean(proc.data)
        proc = proc.divide(avg)
        _add_and_print(str_navg, hdr_new, verbose_bdf)

    # Normalize by the median value
    if normalize_median:
        med = np.median(proc.data)
        proc = proc.divide(med)
        _add_and_print(str_nmed, hdr_new, verbose_bdf)

    # Do CRREJ
    if do_crrej:
        if crrej_kwargs is None:
            crrej_kwargs = LACOSMIC_KEYS
            warn("You are not specifying CR-rejection parameters! It can be"
                 + " dangerous to use defaults blindly.")

        if (("B" in hdr_new["PROCESS"])
                + ("D" in hdr_new["PROCESS"])
                + ("F" in hdr_new["PROCESS"])) < 2:
            warn("L.A. Cosmic should be run AFTER bias, dark, flat process. "
                 + f"You have only done {hdr_new['PROCESS']}. "
                 + "See http://www.astro.yale.edu/dokkum/lacosmic/notes.html")

        proc, crmask = crrej(
            proc,
            propagate_crmask=propagate_crmask,
            update_header=True,
            gain=gain_Q,        # already parsed from local variables above
            rdnoise=rdnoise_Q,  # already parsed from local variables above
            verbose=verbose_crrej,
            **crrej_kwargs)
        hdr_new = proc.header

    proc = CCDData_astype(proc, dtype=dtype,
                          uncertainty_dtype=uncertainty_dtype)
    proc.header = hdr_new

    if output is not None:
        if verbose_bdf:
            print(f"Writing FITS to {output}... ", end='')
        proc.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose_bdf:
            print(f"Saved.")
    return proc
