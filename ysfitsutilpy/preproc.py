from warnings import warn
import ccdproc
import numpy as np
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from ccdproc import flat_correct, subtract_bias, subtract_dark, trim_image

from .ccdutil import CCDData_astype, make_errmap
from .hdrutil import get_from_header

__all__ = ["bdf_process"]


# NOTE: crrej should be done AFTER bias/dark and flat correction:
# http://www.astro.yale.edu/dokkum/lacosmic/notes.html
def bdf_process(ccd, output=None,
                mbiaspath=None, mdarkpath=None, mflatpath=None,
                trim_fits_section=None, calc_err=False, unit='adu',
                gain=None, gain_key="GAIN", gain_unit=u.electron/u.adu,
                rdnoise=None, rdnoise_key="RDNOISE", rdnoise_unit=u.electron,
                dark_exposure=None, data_exposure=None, exposure_key="EXPTIME",
                exposure_unit=u.s, dark_scale=False,
                normalize_exposure=False, normalize_average=False,
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
        Whether to calculate the error map based on Poisson and readnoise
        error propagation.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is ``'adu'``.

    gain, rdnoise : None, float, optional.
        The gain and readnoise value. These are all ignored if
        ``calc_err=False``. If ``calc_err=True``, it automatically seeks
        for suitable gain and readnoise value. If ``gain`` or
        ``readnoise`` is specified, they are interpreted with
        ``gain_unit`` and ``rdnoise_unit``, respectively. If they are
        not specified, this function will seek for the header with
        keywords of ``gain_key`` and ``rdnoise_key``, and interprete the
        header value in the unit of ``gain_unit`` and ``rdnoise_unit``,
        respectively.

    gain_key, rdnoise_key : str, optional.
        See ``gain``, ``rdnoise`` explanation above.
        These are all ignored if ``calc_err=False``.

    gain_unit, rdnoise_unit : astropy Unit, optional.
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
        If ``None`` (default), uses some default values defined in
        ``~.misc.LACOSMIC_KEYS``. It is always discouraged to use
        default except for quick validity-checking, because even the
        official L.A. Cosmic codes in different versions (IRAF, IDL,
        Python, etc) have different default parameters, i.e., there is
        nothing which can be regarded as default.
        To see all possible keywords, do
        ``print(astroscrappy.detect_cosmics.__doc__)``
        Also refer to
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
    def _add_and_print(s, header, verbose):
        header.add_history(s)
        if verbose:
            print(s)

    # Set strings for header history & print (if verbose_bdf)
    str_bias = "Bias subtracted using {}"
    str_dark = "Dark subtracted using {}"
    str_dscale = "Dark scaling {} using {}"
    str_flat = "Flat corrected using {}"
    str_trim = "Trim by FITS section {}"
    str_grd = "From {}, {} = {:.3f} [{}]"
    # str_grd.format(user/header_key, gain/rdnoise, val, unit)
    str_e0 = ("Readnoise propagated with Poisson noise (using gain above)"
              + " of source.")
    str_ed = "Poisson noise from subtracted dark was propagated."
    str_ef = "Flat uncertainty was propagated."
    str_nexp = "Normalized by the exposure time."
    str_navg = "Normalized by the average value of the frame."
    str_cr = ("Cosmic-Ray rejected by astroscrappy (v {}), "
              + "with parameters: {}")

    # Initial setting
    proc = CCDData(ccd)
    hdr_new = proc.header

    # Add PROCESS key
    try:
        _ = hdr_new["PROCESS"]
    except KeyError:
        hdr_new["PROCESS"] = ("", "The processed history: see comment.")
    hdr_new["PROCVER"] = (ccdproc.__version__,
                          "ccdproc version used for processing.")
    hdr_new.add_comment("PROCESS key can be B (bias), D (dark), F (flat), "
                        + "T (trim), W (WCS astrometry), C(CRrej).")

    # Set for BIAS
    if mbiaspath is None:
        do_bias = False
        mbias = CCDData(np.zeros_like(ccd), unit=proc.unit)
    else:
        do_bias = True
        mbias = CCDData.read(mbiaspath, unit=unit)
        hdr_new["PROCESS"] += "B"
        _add_and_print(str_bias.format(mbiaspath), hdr_new, verbose_bdf)

    # Set for DARK
    if mdarkpath is None:
        do_dark = False
        mdark = CCDData(np.zeros_like(ccd), unit=proc.unit)
    else:
        do_dark = True
        mdark = CCDData.read(mdarkpath, unit=unit)
        hdr_new["PROCESS"] += "D"
        _add_and_print(str_dark.format(mdarkpath), hdr_new, verbose_bdf)

        if dark_scale:
            _add_and_print(str_dscale.format(dark_scale, exposure_key),
                           hdr_new, verbose_bdf)

    # Set for FLAT
    if mflatpath is None:
        do_flat = False
        mflat = CCDData(np.ones_like(ccd), unit=proc.unit)
    else:
        do_flat = True
        mflat = CCDData.read(mflatpath)
        hdr_new["PROCESS"] += "F"
        _add_and_print(str_flat.format(mflatpath), hdr_new, verbose_bdf)

    # Set gain and rdnoise if at least one of calc_err and do_crrej is True.
    if calc_err or do_crrej:
        if gain is None:
            gain_Q = get_from_header(hdr_new,
                                     gain_key,
                                     unit=gain_unit,
                                     verbose=False,
                                     default=1.)
            gain_from = gain_key
        else:
            if not isinstance(gain, u.Quantity):
                gain_Q = gain * gain_unit
            else:
                gain_Q = gain
            gain_from = "User"

        _add_and_print(str_grd.format(gain_from,
                                      "gain",
                                      gain_Q.value,
                                      gain_Q.unit),
                       hdr_new, verbose_bdf)

        if rdnoise is None:
            rdnoise_Q = get_from_header(hdr_new,
                                        rdnoise_key,
                                        unit=rdnoise_unit,
                                        verbose=False,
                                        default=1.)
            rdnoise_from = rdnoise_key
        else:
            if not isinstance(rdnoise, u.Quantity):
                rdnoise_Q = rdnoise * rdnoise_unit
            else:
                rdnoise_Q = rdnoise
            rdnoise_from = "User"

        _add_and_print(str_grd.format(rdnoise_from,
                                      "rdnoise",
                                      rdnoise_Q.value,
                                      rdnoise_Q.unit),
                       hdr_new, verbose_bdf)

    # Do TRIM
    if trim_fits_section is not None:
        proc = trim_image(proc, trim_fits_section)
        mbias = trim_image(mbias, trim_fits_section)
        mdark = trim_image(mdark, trim_fits_section)
        mflat = trim_image(mflat, trim_fits_section)
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

    # Do CRREJ
    if do_crrej:
        import astroscrappy
        from astroscrappy import detect_cosmics
        from .misc import LACOSMIC_KEYS

        if crrej_kwargs is None:
            crrej_kwargs = LACOSMIC_KEYS
            warn("You are not specifying CR-rejection parameters and blindly"
                 + " using defaults. It can be dangerous.")

        if (("B" in hdr_new["PROCESS"])
                + ("D" in hdr_new["PROCESS"])
                + ("F" in hdr_new["PROCESS"])) < 2:
            warn("L.A. Cosmic should be run AFTER B/D/F process. "
                 + f"You are running it with {hdr_new['PROCESS']}. "
                 + "See http://www.astro.yale.edu/dokkum/lacosmic/notes.html")

        # remove the fucxing cosmic rays
        crmask, cleanarr = detect_cosmics(proc.data,
                                          inmask=proc.mask,
                                          gain=gain_Q.value,
                                          readnoise=rdnoise_Q.value,
                                          **crrej_kwargs,
                                          verbose=verbose_crrej)

        # create the new ccd data object
        #   astroscrappy automatically does the gain correction, so return
        #   back to avoid confusion.
        proc.data = cleanarr / gain_Q.value
        if propagate_crmask:
            if proc.mask is None:
                proc.mask = crmask
            else:
                proc.mask = proc.mask + crmask
        hdr_new["PROCESS"] += "C"
        _add_and_print(str_cr.format(astroscrappy.__version__, crrej_kwargs),
                       hdr_new, verbose_crrej)

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
