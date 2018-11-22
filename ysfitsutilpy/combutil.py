from warnings import warn
from pathlib import Path

import numpy as np

from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
from astropy import units as u
from ccdproc import sigma_func as ccdproc_mad2sigma_func
from ccdproc import (combine, trim_image, subtract_bias, subtract_dark,
                     flat_correct)

from .filemgmt import load_if_exists, make_summary
from .ccdutil import CCDData_astype, load_ccd, make_errmap
from .hdrutil import get_from_header

__all__ = ["MEDCOMB_KEYS_INT", "SUMCOMB_KEYS_INT", "MEDCOMB_KEYS_FLT32",
           "stack_FITS", "combine_ccd", "bdf_process"]

MEDCOMB_KEYS_INT = dict(dtype='int16',
                        combine_method='median',
                        reject_method=None,
                        unit='adu',
                        combine_uncertainty_function=None)

SUMCOMB_KEYS_INT = dict(dtype='int16',
                        combine_method='sum',
                        reject_method=None,
                        unit='adu',
                        combine_uncertainty_function=None)

MEDCOMB_KEYS_FLT32 = dict(dtype='float32',
                          combine_method='median',
                          reject_method=None,
                          unit='adu',
                          combine_uncertainty_function=None)


def stack_FITS(fitslist=None, extension=0, unit='adu', summary_table=None,
               table_filecol="file", trim_fits_section=None,
               type_key=None, type_val=None):
    ''' Stacks the FITS files specified in fitslist
    Parameters
    ----------
    fitslist: str, path-like, or list of such
        The list of FITS files to be stacked

    extension: int or str
        The extension of FITS to be stacked. For single extension, set it as 0.

    unit: Unit or str, optional

    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many
        FITS files and you want to use stacking many times, it is better to
        make a summary table by ``filemgmt.make_summary`` and use that instead
        of opening FITS files' headers every time you call this function.

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to
        the FITS files.

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
        if val.lstrip('+-').isdigit():  # if int
            result = int(val)
        else:
            try:
                result = float(val)
            except ValueError:
                result = str(val)
        return result

    if not ((fitslist is None) ^ (summary_table is None)):
        raise ValueError(
            "One and only one of filelist or summary_table must be not None.")

    if ((type_key is None) ^ (type_val is None)):
        raise KeyError(
            "type_key and type_val must be both specified or both None.")

    # Setting whether to group
    grouping = False
    if type_key is not None:
        if len(type_key) != len(type_val):
            raise ValueError(
                "type_key and type_val must be of the same length.")
        grouping = True
        # If str, change to list:
        if isinstance(type_key, str):
            type_key = [type_key]
        if isinstance(type_val, str):
            type_val = [type_val]

    all_ccd = []

    # Set fitslist and summary_table based on the given input and grouping.
    if fitslist is not None:
        fitslist = list(fitslist)
        if grouping:
            summary_table = make_summary(fitslist, extension=extension,
                                         verbose=True, fname_option='relative',
                                         keywords=type_key, sort_by=None)
            summary_table = summary_table.to_pandas()
    elif summary_table is not None:
        fitslist = summary_table[table_filecol].tolist()
        if isinstance(summary_table, Table):
            summary_table = summary_table.to_pandas()

    # Append appropriate CCDs to all_ccd
    if grouping:
        for i, row in summary_table.iterrows():
            mismatch = False
            for k, v in zip(type_key, type_val):
                hdr_val = _parse_val(row[k])
                parse_v = _parse_val(v)
                if (hdr_val != parse_v):
                    mismatch = True
                    break
            if mismatch:  # skip this row (file)
                continue

            # if not skipped:
            fpath = fitslist[i]
            ccd_i = load_ccd(fpath, extension=extension, unit=unit)
            if trim_fits_section is not None:
                ccd_i = trim_image(ccd_i, fits_section=trim_fits_section)
            all_ccd.append(ccd_i)

    else:
        for fpath in fitslist:
            ccd_i = load_ccd(fpath, extension=extension, unit=unit)
            if trim_fits_section is not None:
                ccd_i = trim_image(ccd_i, fits_section=trim_fits_section)
            all_ccd.append(ccd_i)

    # Generate warning OR information messages
    if len(all_ccd) == 0:
        if grouping:
            warn('No FITS file had "{:s} = {:s}"'.format(str(type_key),
                                                         str(type_val))
                 + "Maybe int/float/str confusing?")
        else:
            warn('No FITS file found')
    else:
        if grouping:
            print('{:d} FITS files with "{:s} = {:s}"'
                  ' are loaded.'.format(len(all_ccd),
                                        str(type_key),
                                        str(type_val)))
        else:
            print('{:d} FITS files are loaded.'.format(len(all_ccd)))

    return all_ccd


def combine_ccd(fitslist=None, summary_table=None, trim_fits_section=None,
                table_filecol="file", output=None, unit='adu',
                subtract_frame=None, combine_method='median',
                reject_method=None, normalize=False, exposure_key='EXPTIME',
                combine_uncertainty_function=ccdproc_mad2sigma_func,
                extension=0, type_key=None, type_val=None,
                dtype=np.float32, output_verify='fix', overwrite=False,
                **kwargs):
    ''' Combining images
    Slight variant from ccdproc.
    # TODO: accept the input like ``sigma_clip_func='median'``, etc.
    Parameters
    ----------
    fitslist: list of str, path-like
        list of FITS files.

    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many
        FITS files and you want to use stacking many times, it is better to
        make a summary table by ``filemgmt.make_summary`` and use that instead
        of opening FITS files' headers every time you call this function.

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to
        the FITS files.

    subtract_frame: CCDData, array-like
        The user-specified frame to be subtracted from combined CCD. It is
        better not to use bias frame here, because this is for a special
        subtraction rather than regular bias.

    combine: str
        The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median', 'sum'}

    reject: str
        Made for simple use of ``ccdproc.combine``,
        {None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema'}. Automatically turns
        on the option, e.g., ``clip_extrema = True`` or ``sigma_clip = True``.
        Leave it blank for no rejection.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.
        For an open HDU named ``hdu``, e.g., only the files which satisfies
        ``hdu[extension].header[type_key] == type_val`` among all the ``fitslist``
        will be used.

    **kwarg:
        kwargs for the ``ccdproc.combine``. See its documentation.
        This includes (RHS are the default values)
        ```
        weights=None,
        scale=None,
        mem_limit=16000000000.0,
        clip_extrema=False,
        nlow=1,
        nhigh=1,
        minmax_clip=False,
        minmax_clip_min=None,
        minmax_clip_max=None,
        sigma_clip=False,
        sigma_clip_low_thresh=3,
        sigma_clip_high_thresh=3,
        sigma_clip_func=<numpy.ma.core._frommethod instance>,
        sigma_clip_dev_func=<numpy.ma.core._frommethod instance>,
        dtype=None,
        combine_uncertainty_function=None, **ccdkwargs
        ```

    Returns
    -------
    master: astropy.nddata.CCDData
        Resulting combined ccd.

    '''

    def _set_reject_method(reject_method):
        ''' Convenience function for ccdproc.combine reject switches
        '''
        clip_extrema, minmax_clip, sigma_clip = False, False, False

        if reject_method == 'extrema':
            clip_extrema = True
        elif reject_method == 'minmax':
            minmax_clip = True
        elif ((reject_method == 'sigma_clip') or (reject_method == 'sigclip')):
            sigma_clip = True
        else:
            if reject_method is not None:
                raise KeyError("reject must be one of "
                               "{None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema'}")

        return clip_extrema, minmax_clip, sigma_clip

    def _print_info(combine_method, Nccd, reject_method, **kwargs):
        if reject_method is None:
            reject_method = 'no'

        info_str = ('"{:s}" combine {:d} images by "{:s}" rejection')

        print(info_str.format(combine_method, Nccd, reject_method))
        print(dict(**kwargs))
        return

    def _normalize_exptime(ccdlist, exposure_key):
        _ccdlist = ccdlist.copy()
        exptimes = []

        for i in range(len(_ccdlist)):
            exptime = _ccdlist[i].header[exposure_key]
            exptimes.append(exptime)
            _ccdlist[i] = _ccdlist[i].divide(exptime)

        if len(np.unique(exptimes)) != 1:
            print('There are more than one exposure times:\n\t', end=' ')
            print(np.unique(exptimes), end=' ')
            print('seconds')
        print(f'Normalized images by exposure time ("{exposure_key}").')

        return _ccdlist

    # def _ccdproc_combine(ccdlist, combine_method, min_value=0,
    #                     combine_uncertainty_function=ccdproc_mad2sigma_func,
    #                     **kwargs):
    #     ''' Combine after minimum value correction and then rejection/trimming.
    #     ccdlist:
    #         list of CCDData

    #     combine_method: str
    #         The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median',
    #         'sum'}

    #     **kwargs:
    #         kwargs for the ``ccdproc.combine``. See its documentation.
    #     '''
    #     if not isinstance(ccdlist, list):
    #         ccdlist = [ccdlist]

    #     # copy for safety
    #     use_ccds = ccdlist.copy()

    #     # minimum value correction and trim
    #     for ccd in use_ccds:
    #         ccd.data[ccd.data < min_value] = min_value

    #     #combine
    #     ccd_combined = combine(img_list=use_ccds,
    #                         method=combine_method,
    #                         combine_uncertainty_function=combine_uncertainty_function,
    #                         **kwargs)

    #     return ccd_combined

    if not ((fitslist is None) ^ (summary_table is None)):
        raise ValueError(
            "One and only one of filelist or summary_table must be not None.")

    if fitslist is not None:
        if not isinstance(fitslist, list):
            raise TypeError(
                f"fitslist must be a list. It's now {type(fitslist)}.")

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            print(f"{output} already exists:\n\tBut will be overridden.")
        else:
            print(f"{output} already exists:")
            return load_if_exists(output, loader=CCDData.read, if_not=None)

    ccdlist = stack_FITS(fitslist=fitslist,
                         summary_table=summary_table,
                         table_filecol=table_filecol,
                         extension=extension,
                         unit=unit,
                         trim_fits_section=trim_fits_section,
                         type_key=type_key,
                         type_val=type_val)
    header = ccdlist[0].header

    _print_info(combine_method=combine_method,
                Nccd=len(ccdlist),
                reject_method=reject_method,
                dtype=dtype,
                **kwargs)

    # Normalize by exposure
    if normalize:
        ccdlist = _normalize_exptime(ccdlist, exposure_key)

    # Set rejection switches
    clip_extrema, minmax_clip, sigma_clip = _set_reject_method(reject_method)

    if len(ccdlist) == 1:
        master = ccdlist[0]
    else:
        master = combine(img_list=ccdlist,
                         combine_method=combine_method,
                         clip_extrema=clip_extrema,
                         minmax_clip=minmax_clip,
                         sigma_clip=sigma_clip,
                         combine_uncertainty_function=combine_uncertainty_function,
                         **kwargs)

    str_history = '{:d} images with {:s} = {:s} are {:s} combined '
    ncombine = len(ccdlist)
    header["NCOMBINE"] = ncombine
    header.add_history(str_history.format(ncombine,
                                          str(type_key),
                                          str(type_val),
                                          str(combine_method)))

    if subtract_frame is not None:
        subtract = CCDData(subtract_frame.copy())
        master.data = master.subtract(subtract).data
        header.add_history("Subtracted a user-provided frame")

    master.header = header
    master = CCDData_astype(master, dtype=dtype)

    if output is not None:
        master.write(output, output_verify=output_verify, overwrite=overwrite)

    return master


# TODO: put an option such that the crrej can be done either before/after the preprocessing..?
def bdf_process(ccd, output=None, mbiaspath=None, mdarkpath=None, mflatpath=None,
                fits_section=None, calc_err=False, unit='adu', gain=None,
                rdnoise=None, gain_key="GAIN", rdnoise_key="RDNOISE",
                gain_unit=u.electron / u.adu, rdnoise_unit=u.electron,
                dark_exposure=None, data_exposure=None, exposure_key="EXPTIME",
                exposure_unit=u.s, dark_scale=False,
                min_value=None, norm_value=None,
                do_crrej=False, verbose_crrej=False,
                verbose_bdf=True, output_verify='fix', overwrite=True,
                dtype="float32"):
    ''' Do bias, dark and flat process.
    Parameters
    ----------
    ccd: array-like
        The ccd to be processed.
    output: str or path-like
        Saving path
    '''

    proc = CCDData(ccd)
    hdr_new = proc.header
    hdr_new["PROCESS"] = ("", "The processed history: see comment.")
    hdr_new.add_comment("PROCESS key can be B (bias), D (dark), F (flat), "
                        + "T (trim), W (WCS), C(CRrej).")

    if mbiaspath is None:
        do_bias = False
        mbias = CCDData(np.zeros_like(ccd), unit=proc.unit)

    else:
        do_bias = True
        mbias = CCDData.read(mbiaspath, unit=unit)
        hdr_new["PROCESS"] += "B"
        hdr_new.add_history(f"Bias subtracted using {mbiaspath}")

    if mdarkpath is None:
        do_dark = False
        mdark = CCDData(np.zeros_like(ccd), unit=proc.unit)

    else:
        do_dark = True
        mdark = CCDData.read(mdarkpath, unit=unit)
        hdr_new["PROCESS"] += "D"
        hdr_new.add_history(f"Dark subtracted using {mdarkpath}")
        if dark_scale:
            hdr_new.add_history(
                f"Dark scaling {dark_scale} using {exposure_key}")

    if mflatpath is None:
        do_flat = False
        mflat = CCDData(np.ones_like(ccd), unit=proc.unit)

    else:
        do_flat = True
        mflat = CCDData.read(mflatpath)
        hdr_new["PROCESS"] += "F"
        hdr_new.add_history(f"Flat corrected using {mflatpath}")

    if fits_section is not None:
        proc = trim_image(proc, fits_section)
        mbias = trim_image(mbias, fits_section)
        mdark = trim_image(mdark, fits_section)
        mflat = trim_image(mflat, fits_section)
        hdr_new["PROCESS"] += "T"
        hdr_new.add_history(f"Trim by FITS section {fits_section}")

    if do_bias:
        proc = subtract_bias(proc, mbias)

    if do_dark:
        proc = subtract_dark(proc,
                             mdark,
                             dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
        # if calc_err and verbose:
        #     if mdark.uncertainty is not None:
        #         print("Dark has uncertainty frame: Propagate in arithmetics.")
        #     else:
        #         print("Dark does NOT have uncertainty frame")

    if calc_err:
        if gain is None:
            gain = get_from_header(hdr_new, gain_key,
                                   unit=gain_unit,
                                   verbose=verbose_bdf,
                                   default=1.).value

        if rdnoise is None:
            rdnoise = get_from_header(hdr_new, rdnoise_key,
                                      unit=rdnoise_unit,
                                      verbose=verbose_bdf,
                                      default=0.).value

        err = make_errmap(proc,
                          gain_epadu=gain,
                          subtracted_dark=mdark)

        proc.uncertainty = StdDevUncertainty(err)
        errstr = (f"Error calculated using gain = {gain:.3f} [e/ADU] and "
                  + f"rdnoise = {rdnoise:.3f} [e].")
        hdr_new.add_history(errstr)

    if do_flat:
        if calc_err:
            if (mflat.uncertainty is not None) and verbose_bdf:
                print("Flat has uncertainty frame: Propagate in arithmetics.")
                hdr_new.add_history(
                    "Flat had uncertainty and is also propagated.")

        proc = flat_correct(proc,
                            mflat,
                            min_value=min_value,
                            norm_value=norm_value)

    # Do very simple L.A. Cosmic default crrejection
    if do_crrej:
        from astroscrappy import detect_cosmics

        # I skipped two params in LACOSMIC: gain=1.0, readnoise=6.5
        LACOSMIC = dict(sigclip=4.5, sigfrac=0.3, objlim=5.0,
                        satlevel=np.inf, pssl=0.0, niter=4, sepmed=False,
                        cleantype='medmask', fsmode='median', psfmodel='gauss',
                        psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765)
        if gain is None:
            gain = get_from_header(hdr_new, gain_key,
                                   unit=gain_unit,
                                   verbose=verbose_bdf,
                                   default=1.0).value

        if rdnoise is None:
            rdnoise = get_from_header(hdr_new, rdnoise_key,
                                      unit=rdnoise_unit,
                                      verbose=verbose_bdf,
                                      default=6.5).value

        crmask, cleanarr = detect_cosmics(proc.data, inmask=proc.mask,
                                          gain=gain, readnoise=rdnoise,
                                          **LACOSMIC, verbose=verbose_crrej)

        # create the new ccd data object
        proc.data = cleanarr
        if proc.mask is None:
            proc.mask = crmask
        else:
            proc.mask = proc.mask + crmask
        hdr_new["PROCESS"] += "C"
        hdr_new.add_history(
            f"Cosmic-Ray rejected by astroscrappy, LACOSMIC default setting.")

    proc = CCDData_astype(proc, dtype=dtype)
    proc.header = hdr_new

    if output is not None:
        proc.write(output, output_verify=output_verify, overwrite=overwrite)

    return proc