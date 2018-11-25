from warnings import warn
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
from astropy import units as u
from astropy.io import fits
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


def sstd(a, **kwargs):
    ''' Sample standard deviation function
    '''
    return np.std(a, ddof=1, **kwargs)


# # FIXME: Add this to Ccdproc
# def weighted_mean(ccds, unit='adu'):
#     datas = []
#     ws = []  # weights = 1 / sigma**2
#     for ccd in ccds:
#         datas.append(ccd.data)
#         ws.append(1 / ccd.uncertainty.array**2)
#     wmean = np.average(np.array(datas), axis=0, weights=ws)
#     wuncert = np.sqrt(1 / np.sum(np.array(ws), axis=0))
#     nccd = CCDData(data=wmean, header=ccds[0].header, unit=unit)
#     nccd.uncertainty = StdDevUncertainty(wuncert)
#     return nccd


def stack_FITS(fitslist=None, summary_table=None, extension=0,
               unit='adu', table_filecol="file", trim_fits_section=None,
               loadccd=True, type_key=None, type_val=None):
    ''' Stacks the FITS files specified in fitslist
    Parameters
    ----------
    fitslist: path-like, list of path-like, or list of CCDData
        The list of path to FITS files or the list of CCDData to be stacked.
        It is useful to give list of CCDData if you have already stacked/loaded
        FITS file into a list by your own criteria. If ``None`` (default),
        you must give ``fitslist`` or ``summary_table``. If it is not ``None``,
        this function will do very similar job to that of ``ccdproc.combine``.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many
        FITS files and you want to use stacking many times, it is better to
        make a summary table by ``filemgmt.make_summary`` and use that instead
        of opening FITS files' headers every time you call this function. If
        you want to use ``summary_table`` instead of ``fitslist`` and have set
        ``loadccd=True``, you must not have ``None`` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    extension: int or str
        The extension of FITS to be stacked. For single extension, set it as 0.

    unit: Unit or str, optional

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to
        the FITS files.

    trim_fits_section: str, optional
        Region of ``ccd`` to be trimmed; see ``ccdproc.subtract_overscan`` for
        details. Default is None.

    loadccd: bool, optional
        Whether to return file paths or loaded CCDData. If ``False``, it is
        a function to select FITS files using ``type_key`` and ``type_val``
        without using much memory.

    Return
    ------
    matched: list of Path or list of CCDData
        list containing Path to files if ``loadccd`` is ``False``. Otherwise
        it is a list containing loaded CCDData after loading the files. If
        ``ccdlist`` is given a priori, list of CCDData will be returned
        regardless of ``loadccd``.
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

    def _check_mismatch(row):
        mismatch = False
        for k, v in zip(type_key, type_val):
            hdr_val = _parse_val(row[k])
            parse_v = _parse_val(v)
            if (hdr_val != parse_v):
                mismatch = True
                break
        return mismatch

    if ((fitslist is not None) + (summary_table is not None) != 1):
        raise ValueError(
            "One and only one of filelist or summary_table must be not None.")

    # If fitslist
    if (fitslist is not None) and (not isinstance(fitslist, list)):
        fitslist = [fitslist]
        # raise TypeError(
        #     f"fitslist must be a list. It's now {type(fitslist)}.")

    # If summary_table
    if summary_table is not None:
        if ((not isinstance(summary_table, Table)) and
                (not isinstance(summary_table, pd.DataFrame))):
            raise TypeError(
                f"summary_table must be an astropy Table or Pandas DataFrame. It's now {type(summary_table)}.")

    # Check for type_key and type_val
    if ((type_key is None) ^ (type_val is None)):
        raise ValueError(
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

    matched = []

    print("Analyzing FITS... ", end='')
    # Set fitslist and summary_table based on the given input and grouping.
    if fitslist is not None:
        if grouping:
            summary_table = make_summary(fitslist, extension=extension,
                                         verbose=True, fname_option='relative',
                                         keywords=type_key, sort_by=None)
            summary_table = summary_table.to_pandas()
    elif summary_table is not None:
        fitslist = summary_table[table_filecol].tolist()
        if isinstance(summary_table, Table):
            summary_table = summary_table.to_pandas()

    print("Done", end='')
    if load_ccd:
        print(" and loading FITS... ")
    else:
        print(".")

    # Append appropriate CCDs or filepaths to matched
    if grouping:
        for i, row in summary_table.iterrows():
            mismatch = _check_mismatch(row)
            if mismatch:  # skip this row (file)
                continue

            # if not skipped:
            # TODO: Is is better to remove Path here?
            if isinstance(fitslist[i], CCDData):
                matched.append(fitslist[i])
            else:  # it must be a path to the file
                fpath = Path(fitslist[i])
                if loadccd:
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(
                            ccd_i, fits_section=trim_fits_section)
                    matched.append(ccd_i)
                else:
                    matched.append(fpath)
    else:
        for item in fitslist:
            if isinstance(item, CCDData):
                matched.append(item)
            else:
                if loadccd:
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(
                            ccd_i, fits_section=trim_fits_section)
                    matched.append(ccd_i)
                else:  # TODO: Is is better to remove Path here?
                    matched.append(Path(fpath))

    # Generate warning OR information messages
    if len(matched) == 0:
        if grouping:
            warn('No FITS file had "{:s} = {:s}"'.format(str(type_key),
                                                         str(type_val))
                 + "Maybe int/float/str confusing?")
        else:
            warn('No FITS file found')
    else:
        if grouping:
            N = len(matched)
            ks = str(type_key)
            vs = str(type_val)
            if load_ccd:
                print(f'{N} FITS files with "{ks} = {vs}" are loaded.')
            else:
                print(f'{N} FITS files with "{ks} = {vs}" are selected.')
        else:
            if load_ccd:
                print('{:d} FITS files are loaded.'.format(len(matched)))

    return matched


def combine_ccd(fitslist=None, summary_table=None, trim_fits_section=None,
                table_filecol="file", output=None, unit='adu',
                subtract_frame=None, combine_method='median',
                reject_method=None, normalize_exposure=False,
                exposure_key='EXPTIME', weights=None, mem_limit=16e9,
                combine_uncertainty_function=sstd, scale=None,
                extension=0, type_key=None, type_val=None,
                dtype="float32", uncertainty_dtype="float32",
                output_verify='fix', overwrite=False,
                verbose=True, **kwargs):
    ''' Combining images
    Slight variant from ccdproc.
    # TODO: accept the input like ``sigma_clip_func='median'``, etc.
    Parameters
    ----------
    fitslist: path-like, list of path-like, or list of CCDData
        The list of path to FITS files or the list of CCDData to be stacked.
        It is useful to give list of CCDData if you have already stacked/loaded
        FITS file into a list by your own criteria. If ``None`` (default),
        you must give ``fitslist`` or ``summary_table``. If it is not ``None``,
        this function will do very similar job to that of ``ccdproc.combine``.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    summary_table: pandas.DataFrame or astropy.table.Table
        The table which contains the metadata of files. If there are many
        FITS files and you want to use stacking many times, it is better to
        make a summary table by ``filemgmt.make_summary`` and use that instead
        of opening FITS files' headers every time you call this function. If
        you want to use ``summary_table`` instead of ``fitslist`` and have set
        ``loadccd=True``, you must not have ``None`` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    table_filecol: str
        The column name of the ``summary_table`` which contains the path to
        the FITS files.

    subtract_frame: CCDData, array-like
        The user-specified frame to be subtracted from combined CCD. It is
        better not to use bias frame here, because this is for a special
        subtraction rather than regular bias.

    combine: str
        The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median', 'sum'},
        or 'wmean', 'weighted mean', or 'weighted_mean' for weighted mean.

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

     scale : function or ``numpy.ndarray``-like or None, optional
        Scaling factor to be used when combining images.
        Images are multiplied by scaling prior to combining them. Scaling
        may be either a function, which will be applied to each image
        to determine the scaling factor, or a list or array whose length
        is the number of images in the `Combiner`. Default is ``None``.

    weights : ``numpy.ndarray`` or None, optional
        Weights to be used when combining images.
        An array with the weight values. The dimensions should match the
        the dimensions of the data arrays being combined.
        Default is ``None``.

    mem_limit : float, optional
        Maximum memory which should be used while combining (in bytes).
        Default is ``16e9``.

    **kwarg:
        kwargs for the ``ccdproc.combine``. See its documentation.
        This includes (RHS are the default values)
        ```
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

        info_str = ('"{:s}" combine {:d} images by "{:s}" rejection\n')

        print(info_str.format(combine_method, Nccd, reject_method))
        print(dict(**kwargs))
        return

    # def _normalize_exptime(ccdlist, exposure_key):
    #     _ccdlist = ccdlist.copy()
    #     exptimes = []

    #     for i in range(len(_ccdlist)):
    #         exptime = _ccdlist[i].header[exposure_key]
    #         exptimes.append(exptime)
    #         _ccdlist[i] = _ccdlist[i].divide(exptime)

    #     if verbose:
    #         if len(np.unique(exptimes)) != 1:
    #             print('There are more than one exposure times:\n\t', end=' ')
    #             print(np.unique(exptimes), end=' ')
    #             print('seconds')
    #         print(f'Normalized images by exposure time ("{exposure_key}").')

    #     return _ccdlist

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

    # Give only one
    if ((fitslist is not None) + (summary_table is not None) != 1):
        raise ValueError(
            "One and only one of [filelist, summary_table, ccdlist] must be not None.")

    # If fitslist
    if fitslist is not None:
        if not isinstance(fitslist, list):
            raise TypeError(
                f"fitslist must be a list. It's now {type(fitslist)}.")

    # If summary_table
    if summary_table is not None:
        if ((not isinstance(summary_table, Table)) and
                (not isinstance(summary_table, pd.DataFrame))):
            raise TypeError(
                f"summary_table must be an astropy Table or Pandas DataFrame. It's now {type(summary_table)}.")

    # Check for type_key and type_val
    if ((type_key is None) ^ (type_val is None)):
        raise ValueError(
            "type_key and type_val must be both specified or both None.")

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            print(f"{output} already exists:\n\tBut will be overridden.")
        else:
            print(f"{output} already exists:")
            return load_if_exists(output, loader=CCDData.read, if_not=None)

    # Loading CCD here may cause memory blast...
    ccdlist = stack_FITS(fitslist=fitslist,
                         summary_table=summary_table,
                         table_filecol=table_filecol,
                         extension=extension,
                         unit=unit,
                         trim_fits_section=trim_fits_section,
                         type_key=type_key,
                         type_val=type_val,
                         loadccd=False)

    try:
        header = ccdlist[0].header
    except AttributeError:
        header = fits.getheader(ccdlist[0])

    if verbose:
        _print_info(combine_method=combine_method,
                    Nccd=len(ccdlist),
                    reject_method=reject_method,
                    dtype=dtype,
                    **kwargs)

    # Normalize by exposure
    if normalize_exposure or scale is not None:
        header.add_history("Normalized by exposure time.")
        if scale is None:
            exptimes = make_summary(fitslist=fitslist,
                                    keywords=[exposure_key],
                                    verbose=False,
                                    sort_by=None)
            scale = 1 / np.array(exptimes.tolist())

    # Set rejection switches
    clip_extrema, minmax_clip, sigma_clip = _set_reject_method(reject_method)

    if len(ccdlist) == 1:
        if isinstance(ccdlist[0], CCDData):
            master = ccdlist[0]
        else:
            master = load_ccd(ccdlist[0], extension=extension, unit=unit)
    else:
        master = combine(img_list=ccdlist,
                         combine_method=combine_method,
                         clip_extrema=clip_extrema,
                         minmax_clip=minmax_clip,
                         sigma_clip=sigma_clip,
                         weights=weights,
                         scale=scale,
                         mem_limit=mem_limit,
                         combine_uncertainty_function=combine_uncertainty_function,
                         unit=unit,
                         hdu=extension,
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
    master = CCDData_astype(master, dtype=dtype,
                            uncertainty_dtype=uncertainty_dtype)

    if output is not None:
        if verbose:
            print(f"Writing FITS to {output}... ", end='')
        master.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose:
            print("Saved.")

    return master


# TODO: put an option such that the crrej can be done either before/after the preprocessing..?
def bdf_process(ccd, output=None, mbiaspath=None, mdarkpath=None, mflatpath=None,
                fits_section=None, calc_err=False, unit='adu', gain=None,
                rdnoise=None, gain_key="GAIN", rdnoise_key="RDNOISE",
                gain_unit=u.electron / u.adu, rdnoise_unit=u.electron,
                dark_exposure=None, data_exposure=None, exposure_key="EXPTIME",
                exposure_unit=u.s, dark_scale=False, normalize_exposure=False,
                min_value=None, norm_value=None,
                do_crrej=False, verbose_crrej=False,
                verbose_bdf=True, output_verify='fix', overwrite=True,
                dtype="float32", uncertainty_dtype="float32"):
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

    # Before going into flat or crrej, calculate Poisson error map first
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
        # TODO: let the unit extracted from input param
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

    # If normalize by the exposure time (e.g., ADU per sec)
    if normalize_exposure:
        if data_exposure is None:
            data_exposure = hdr_new[exposure_key]
        proc = proc.divide(data_exposure)

    # Do very simple L.A. Cosmic default crrejection
    # FIXME: This is very vulnerable...
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

    proc = CCDData_astype(
        proc, dtype=dtype, uncertainty_dtype=uncertainty_dtype)
    proc.header = hdr_new

    if output is not None:
        if verbose_bdf:
            print(f"Writing FITS to {output}... ", end='')
        proc.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose_bdf:
            print(f"Saved.")
    return proc
