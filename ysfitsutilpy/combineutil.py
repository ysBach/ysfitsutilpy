from warnings import warn
from pathlib import Path

import numpy as np

import ccdproc
from ccdproc import sigma_func as ccdproc_mad2sigma_func
from ccdproc import combine
from astropy.nddata import CCDData

from .filemgmt import load_if_exists
from .ccdutil import CCDData_astype

__all__ = ["stack_FITS"]


def stack_FITS(filelist, extension, unit='adu', trim_fits_section=None,
               type_key=None, type_val=None):
    ''' Stacks the FITS files specified in filelist
    Parameters
    ----------
    filelist: str, path-like, or list of such
        The list of FITS files to be stacked

    extension: int or str
        The extension of FITS to be stacked. For single extension, set it as 0.

    unit: Unit or str, optional

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
        if val.lstrip('+-').isdigit():
            result = int(val)
        else:
            try:
                result = float(val)
            except ValueError:
                result = str(val)
        return result

    iskey = False
    filelist = list(filelist)

    if ((type_key is None) ^ (type_val is None)):
        raise KeyError(
            "type_key and type_val must be both specified or both None.")

    if type_key is not None:
        iskey = True
        if isinstance(type_key, str):
            type_key = [type_key]
        if isinstance(type_val, str):
            type_val = [type_val]

        if len(type_key) != len(type_val):
            raise ValueError(
                "type_key and type_val must be of the same length.")

    all_ccd = []

    for i, fname in enumerate(filelist):
        if unit is not None:
            ccd_i = CCDData.read(fname, hdu=extension, unit=unit)
        else:
            ccd_i = CCDData.read(fname, hdu=extension)

        if iskey:
            mismatch = False
            for k, v in zip(type_key, type_val):
                hdr_val = _parse_val(ccd_i.header[k])
                if (hdr_val != v):
                    mismatch = True
                    break
            if mismatch:
                continue

        if trim_fits_section is not None:
            ccd_i = ccdproc.trim_image(ccd_i, fits_section=trim_fits_section)

        all_ccd.append(ccd_i)
#        im_i = hdu_i[extension].data
#        if (i == 0):
#            all_data = im_i
#        elif (i > 0):
#            all_data = np.dstack( (all_data, im_i) )

    if len(all_ccd) == 0:
        if iskey:
            warn('No FITS file had "{:s} = {:s}"'.format(str(type_key),
                                                         str(type_val))
                 + "Maybe int/float confusing?")

        else:
            warn('No FITS file found')
    else:
        if iskey:
            print('{:d} FITS files with "{:s} = {:s}"'
                  ' are loaded.'.format(len(all_ccd),
                                        str(type_key),
                                        str(type_val)))
        else:
            print('{:d} FITS files are loaded.'.format(len(all_ccd)))

    return all_ccd


def combine_ccd(fitslist, trim_fits_section=None, output=None, unit='adu',
                subtract_frame=None, combine_method='median', reject_method=None,
                normalize=False, exposure_key='EXPTIME',
                combine_uncertainty_function=ccdproc_mad2sigma_func,
                extension=0, type_key=None, type_val=None,
                dtype=np.float32, output_verify='fix', overwrite=False,
                **kwargs):
    ''' Combining images
    Slight variant from ccdproc.
    # TODO: accept the input like ``sigma_clip_func='median'``, etc.
    # TODO: normalize maybe useless..?
    Parameters
    ----------
    fitslist: list of str, path-like
        list of FITS files.

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

    if not isinstance(fitslist, list):
        raise TypeError(f"fitslist must be a list. It's now {type(fitslist)}.")

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            print(f"{output} already exists:\n\tBut will be overridden.",
                  end='')
        else:
            print(f"{output} already exists:\n\t", end='')
            return load_if_exists(output, loader=CCDData.read, if_not=None)

    ccdlist = stack_FITS(filelist=fitslist,
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
