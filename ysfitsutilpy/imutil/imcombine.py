from pathlib import Path

import bottleneck as bn
import numpy as np
from astropy.io.fits.verify import VerifyError
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

from ..combutil import group_fits
from ..filemgmt import make_summary
from ..hduutil import (_parse_data_header, _parse_extension,
                       calc_offset_physical, calc_offset_wcs, inputs2list,
                       load_ccd, slicefy, write2fits)
from ..misc import (_image_shape, cmt2hdr, get_size, is_list_like, str_now,
                    update_tlm)
from . import docstrings
from .util_comb import (_set_cenfunc, _set_combfunc, _set_gain_rdns,
                        _set_int_dtype, _set_keeprej, _set_mask,
                        _set_reject_name, _set_sigma, _set_thresh_mask, do_zs,
                        get_zsw)
from .util_reject import ccdclip_mask, minmax_mask, sigclip_mask

__all__ = ["group_combine", "group_save", "imcombine", "ndcombine"]

'''
removed : headers, project, masktype, maskvalue, sigscale, grow
partial removal:
    * combine in ["quadrature", "nmodel"]
replaced
    * reject in ["crreject", "avsigclip"] --> ccdclip with certain params
    * offsets in ["grid", <filename>]  --> offsets in ndarray

bpmasks                : ?
rejmask                : output_mask
nrejmasks              : output_nrej
expmasks               : Should I implement???
sigma                  : output_err
outtype                : dtype
outlimits              : trimsec
expname                : exposure_key

# ALGORITHM PARAMETERS ====================================================== #
lthreshold, hthreshold : thresholds (tuple)
nlow      , nhigh      : n_minmax (tuple)
nkeep                  : nkeep & maxrej
                        (IRAF nkeep > 0 && < 0 case, resp.)
mclip                  : cenfunc
lsigma    , hsigma     : sigma uple
'''


def group_combine(
        inputs,
        type_key=None,
        type_val=None,
        group_key=None,
        fmt=None,
        outdir=None,
        verbose=1,
        **kwargs
):
    ''' Combine sub-groups of FITS files from the given input.
    Parameters
    ----------
    inputs : glob pattern, list-like of path-like
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or list of
        files (each element must be path-like or CCDData). Although it is not a
        good idea, a mixed list of CCDData and paths to the files is also
        acceptable. For the purpose of imcombine function, the best use is to
        use the `~glob` pattern or list of paths.

    type_key, type_val : str, list of str
        The header keyword for the ccd type, and the value you want to match.

    group_key : None, str, list of str, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.

    verbose : int
        Larger number means it becomes more verbose::

          * 0: print nothing
          * 1: Only very essential things from this function
          * 2: + verbose from each imcombine

    fmt : str, optinal.
        The f-string for the output file names.

        ..example:: If `group_key="EXPTIME"` and we had two groups of
          ``EXPTIME`` is 1.0 and 2.0,

          * ``"dark_{:.1f}s"`` --> ``dark_1.0s.fits`` and ``dark_2.0s.fits``
          * For float, non-specification such as ``"d{}"`` is not recommended
            (filename can be ``0.3000...``).

        ..example:: If two `group_key`'s are used, resulting in ``("B", 2.0)``,
          ``("V", 12.0)``, ...:

          * ``"flat_{2:04.1f}_{1:s}"`` --> ``"flat_02.0_B.fits"`` and
            ``"flat_12.0_V.fits"``

    outdir : path-like, optinal.
        The directory where the output fits files will be saved.

    **kwargs :
        The keyword arguments for `imcombine`.

    Returns
    -------
    combined : dict of CCDData
        The dict object where keys are the header value of the `group_key` and
        the values are the combined images in CCDData object. If multiple keys
        for `group_key` is given, the key of this dict is a tuple.
    '''
    def _group_save(ccd, groupname, fmt=None, verbose=1, outdir=None):
        ''' Saves the results.
        '''
        outdir = Path('.') if outdir is None else Path(outdir)
        if verbose >= 1 and not outdir.exists():
            print(f"\tOutput directory: '{outdir}' <- does not exist! It will be newly made.")

        outdir.mkdir(exist_ok=True, parents=True)

        if fmt is None:
            nk = len(group_key) if is_list_like(group_key) else 1  # 1 if str
            fmt = "_".join(['{}']*nk)
            if verbose >= 1:
                print("\tWarning: fmt is not specified! Output file names might be ugly.")

        if isinstance(groupname, tuple):
            fname = fmt.format(*groupname) + ".fits"
        else:
            fname = fmt.format(groupname) + ".fits"

        fname = fname.replace(".fits.fits", ".fits")

        fpath = outdir/fname
        if verbose >= 1:
            if fpath.exists():
                print(f"\t{fpath} will be overridden.")
            else:
                print(f"\t{fpath}")
        ccd.write(fpath, overwrite=True)

    _t = Time.now()

    summary = make_summary(inputs, verbose=verbose >= 2)
    gs, gt_key = group_fits(
        summary, type_key=type_key, type_val=type_val, group_key=group_key
    )
    if verbose >= 1:
        print(f"Group and combine by {group_key} (total {len(gs)} groups)")

    combined = {}
    for g_val, group in gs:
        if is_list_like(g_val):
            if len(g_val) == 1:
                g_val = g_val[0]
        files = group["file"].to_list()
        if verbose >= 1:
            print(f"* {g_val}... ({len(files)} files)")
        if len(files) == 0:
            if verbose >= 1:
                print("No FITS to combine.")
            combined[g_val] = None
        elif len(files) == 1:
            if verbose >= 1:
                print("Only 1 FITS to combine -- returning it without any modification.")
            combined[g_val] = load_ccd(files[0])
            if outdir is not None or fmt is not None:
                _group_save(combined[g_val], g_val, fmt=fmt, outdir=outdir)
        else:
            combined[g_val] = imcombine(files, verbose=verbose >= 2, full=False, **kwargs)
            if outdir is not None or fmt is not None:
                _group_save(combined[g_val], g_val, fmt=fmt, outdir=outdir)

    if verbose >= 1:
        print(str_now(t_ref=_t))

    return combined


def group_save(combined, fmt='', verbose=1, outdir=None):
    ''' Saves the group_combine results.
    Paramters
    ---------
    combined : dict
        The result from `group_combine` function.
    '''
    outdir = Path('.') if outdir is None else Path(outdir)
    if verbose and not outdir.exists():
        print(f"\tOutput directory: '{outdir}' <- does not exist! It will be newly made.")

    outdir.mkdir(exist_ok=True, parents=True)

    if not fmt:
        fmt = "_".join(['{}']*len(list(combined.keys())[0]))
        if verbose:
            print("\tWarning: fmt is not specified! Output file names might be ugly.")

    for k, ccd in combined.items():
        if isinstance(k, tuple):
            fname = fmt.format(*k) + ".fits"
        else:
            fname = fmt.format(k) + ".fits"
        fpath = outdir/fname
        if verbose >= 1 and fpath.exists():
            print(f"The pre-existing file {fpath} will be overridden.")
        ccd.write(fpath, overwrite=True)


def _update_hdr(header, ncombine, imcmb_key, imcmb_val, offset_mode=None, offsets=None,
                zeros=None, scales=None, weights=None):
    """ **Inplace** update of the given header
    """
    def __rm_and_add(hdr, keybase, values):
        for i in range(999):
            if f"{keybase}{i+1:03d}" in hdr:
                del hdr[f"{keybase}{i+1:03d}"]
            else:
                break

        for i in range(min(999, len(values))):
            hdr[f"{keybase}{i+1:03d}"] = values[i]

        return

    header["NCOMBINE"] = (ncombine, "Number of combined images")
    if imcmb_key != '':
        header["IMCMBKEY"] = (imcmb_key, "Key used in IMCMBiii ('$I': filepath)")
        __rm_and_add(header, "IMCMB", imcmb_val)
        # remove header keyword IMCMBiii if it exists:
        for i in range(999):
            if f"IMCMB{i+1:03d}" in header:
                del header[f"IMCMB{i+1:03d}"]
            else:
                break

        for i in range(min(999, len(imcmb_val))):
            header[f"IMCMB{i+1:03d}"] = imcmb_val[i]

    if offset_mode is not None:
        header['OFFSTMOD'] = (offset_mode, "Offset method used for combine.")
        for i in range(min(999, len(imcmb_val))):
            header[f"OFFST{i:03d}"] = str(offsets[i, ][::-1].tolist())

    if not np.all(zeros == 0):
        __rm_and_add(header, "ZERO", zeros)

    if not np.all(scales == 1):
        __rm_and_add(header, "SCALE", scales)

    if not np.all(weights == 1):
        __rm_and_add(header, "WEIGH", weights)

    # Add "IRAF-TLM" like header key for continuity with IRAF.
    update_tlm(header)


def imcombine(
        inputs,
        mask=None,
        extension=None,
        extension_uncertainty=None,
        extension_mask=None,
        uncertainty_type='stddev',
        trimsec=None,
        blank=np.nan,
        offsets=None,
        thresholds=[-np.inf, np.inf],
        zero=None,
        zero_to_0th=True,
        zero_section=None,
        scale=None,
        scale_to_0th=True,
        scale_section=None,
        zero_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        weight=None,
        reject=None,
        sigma=[3., 3.],
        cenfunc='median',
        maxiters=50,
        ddof=1,
        nkeep=1,
        maxrej=None,
        n_minmax=[1, 1],
        rdnoise=0.,
        gain=1.,
        snoise=0.,
        pclip=-0.5,
        logfile=None,
        combine='average',
        dtype='float32',
        dtype_err='float32',
        dtype_low=None,
        dtype_upp=None,
        irafmode=True,
        memlimit=2.5e+9,
        verbose=False,
        full=False,
        return_variance=False,
        imcmb_key='$I',
        exposure_key="EXPTIME",
        output=None,
        output_mask=None,
        output_nrej=None,
        output_err=None,
        output_low=None,
        output_upp=None,
        output_rejcode=None,
        return_dict=False,
        output_verify='exception',
        overwrite=False,
        checksum=False
):
    if verbose:
        _t1 = Time.now()
        print(_t1.iso)
        print("- Organizing", end='... ')

    full = (full or output_mask is not None or output_nrej is not None or
            output_err is not None or output_low is not None or output_upp is not None
            or output_rejcode is not None)

    items = inputs2list(inputs, sort=True, accept_ccdlike=True, check_coherency=True)
    ncombine = len(items)
    reject_fullname = _set_reject_name(reject)
    int_dtype = _set_int_dtype(ncombine)
    extension = _parse_extension(extension)
    # If extensions are given as `None`, don't parse them and leave it as `None`.
    e_u = None if extension_uncertainty is None else _parse_extension(extension_uncertainty)
    e_m = None if extension_mask is None else _parse_extension(extension_mask)
    extract_hdr = imcmb_key not in [None, '', '$I']
    slice_load = None if trimsec is None else slicefy(trimsec)

    # These must be initialized at the early stage.
    zeros = np.zeros(shape=ncombine)
    scales = np.ones(shape=ncombine)
    weights = np.ones(shape=ncombine)

    if logfile is not None:
        logfile = Path(logfile)
        table_dict = dict(file=[], filesize=[])

    # == check if we should care about memory ============================================ #
    # It usually takes < 1ms for hundreds of files
    # What we get here is the lower bound of the total memory used.
    # Even if chop_load is False, we later may have to use chopping when combine. See below.
    # chop_load = False
    item_size_tot = 0
    for item in items:
        try:
            fpath = Path(item)
            _item_size = fpath.stat().st_size
        except (TypeError, ValueError, FileNotFoundError):
            fpath = f"User-provided {item.__class__.__name__}"
            _item_size = get_size(item)
        item_size_tot += _item_size
        if logfile is not None:
            table_dict['file'].append(fpath)
            table_dict['filesize'].append(_item_size)
    # if fsize_tot > memlimit:
    #     chop_load = True
    # ---------------------------------------------------------------------------------------- #

    _, hdr0 = _parse_data_header(items[0], extension=extension, parse_data=False)
    ndim = hdr0['NAXIS']
    # N x ndim. sizes[i, :] = images[i].shape
    shapes = np.ones((ncombine, ndim), dtype=int)

    extract_exptime = False
    if isinstance(scale, str):
        if scale.lower() in ["exp", "expos", "exposure", "exptime"]:
            extract_exptime = True

    if reject_fullname == 'ccdclip':
        extract_gain, gns = _set_gain_rdns(gain, ncombine, dtype=dtype)
        extract_rdnoise, rds = _set_gain_rdns(rdnoise, ncombine, dtype=dtype)
        extract_snoise, sns = _set_gain_rdns(snoise, ncombine, dtype=dtype)
    else:
        extract_gain, gns = False, 1
        extract_rdnoise, rds = False, 0
        extract_snoise, sns = False, 0

    # == Extract header info ============================================================= #
    # TODO: if offsets is None and `fsize_tot` << memlimit, why not
    # just load all data here?
    # initialize
    use_wcs, use_phy = False, False

    if isinstance(offsets, str):
        if offsets.lower() in ['world', 'wcs']:
            w_ref = WCS(hdr0)
            use_wcs = True
            offset_mode = "WCS"
            offsets = np.zeros((ncombine, ndim))
        elif offsets.lower() in ['physical', 'phys', 'phy']:
            use_phy = True
            offset_mode = "Physical"
            offsets = np.zeros((ncombine, ndim))
        else:
            raise ValueError("offsets not understood.")
    elif offsets is None:
        offset_mode = None
        offsets = np.zeros((ncombine, ndim))
    else:
        if offsets.shape[0] != ncombine:
            raise ValueError("offset.shape[0] must be num(images)")
        offset_mode = "User"
        offsets = np.array(offsets)

    imcmb_val = []
    # iterate over files
    extract_hdr = (extract_hdr or extract_exptime or extract_gain or extract_rdnoise or extract_snoise
                   or use_wcs or use_phy)
    for i, item in enumerate(items):
        if extract_hdr:
            _, hdr = _parse_data_header(item, extension=extension, copy=False)
            if imcmb_key not in [None, '']:
                if imcmb_key == "$I":
                    try:
                        imcmb_val.append(Path(item).name)
                    except TypeError:
                        imcmb_val.append(f"User-provided {type(item)}")
                else:
                    try:
                        imcmb_val.append(hdr[imcmb_key])
                    except KeyError:
                        imcmb_val.append('')

            if extract_exptime:
                scales[i] = float(hdr[exposure_key])

            if extract_gain:
                gns[i] = float(hdr[gain])  # gain is given as header key

            if extract_rdnoise:
                rds[i] = float(hdr[rdnoise])  # rdnoise is given as header key

            if extract_snoise:
                sns[i] = float(hdr[snoise])  # snoise is given as header key

            if hdr['NAXIS'] != ndim:
                raise ValueError(
                    "All FITS files must have the identical ndim, though they can have different sizes."
                )

            # Update offsets if WCS or Physical should be used
            if use_wcs:
                # Code if using WCS, which may be much slower (but accurate?)
                # Find the center's pixel position in w_ref, in nearest integer value.
                offsets[i, ] = calc_offset_wcs(WCS(hdr), w_ref, intify_offset=True,
                                               loc_target='center', loc_reference='center', order_xyz=False)
                # For IRAF-like calculation, use
                #   offsets[i, ] = [hdr[f'CRPIX{i}'] for i in range(ndim, 0, -1)]
            elif use_phy:
                offsets[i, ] = calc_offset_physical(hdr, None, intify_offset=True,
                                                    order_xyz=False, ignore_ltm=True)

            # NOTE: the indexing in python is [z, y, x] order!!
            shapes[i, ] = [int(hdr[f'NAXIS{i}']) for i in range(ndim, 0, -1)]

            del hdr

        else:
            if imcmb_key == "$I":
                try:
                    imcmb_val.append(Path(item).name)
                except TypeError:
                    imcmb_val.append(f"User-provided {type(item)}")
            data = _parse_data_header(item, extension=extension, parse_header=False)[0]
            if trimsec is not None:
                shapes[i, ] = data[slice_load].shape
            else:
                shapes[i, ] = data.shape
    # ------------------------------------------------------------------------------------ #

    # == Check the size of the temporary array for combination =========================== #
    offsets, sh_comb = _image_shape(shapes, offsets, method='outer', offset_order_xyz=False, intify_offsets=True)

    # Size of (N+1)-D array before combining along axis=0
    stacksize = np.prod((ncombine, *sh_comb))*(np.dtype(dtype).itemsize)
    # size estimated by full-stacked array (1st term) plus combined image (1/ncombine), low and upp
    # bounds (each 1/ncombine), mask (bool8), niteration (int8), and code(int8).
    # temp_arr_size = stacksize*(1 + 1/ncombine*4)

    # Copied from ccdproc v 2.0.1
    # https://github.com/astropy/ccdproc/blob/b9ec64dfb59aac1d9ca500ad172c4eb31ec305f8/ccdproc/combiner.py#L710
    # Set a memory use factor based on profiling
    combmeth = _set_combfunc(combine)
    memory_factor = 3 if combmeth == "median" else 2
    memory_factor *= 1.5
    mem_req = memory_factor * stacksize
    num_chunk = int(mem_req / memlimit) + 1

    # TODO: make chunking
    if num_chunk > 1:
        raise ValueError(
            "Currently chunked combine is not supporte T__T. "
            + f"Please try increasing memlimit to > {mem_req:.1e}, "
            + "or use combine='avg' than 'median'."
        )
    if verbose:
        print("Done.")
        if num_chunk > 1:
            print(f"memlimit reached: Split combine by {num_chunk} chunks.")

    # == Setup offset-ed array =========================================================== #
    # NOTE: Using NaN does not set array with dtype of int... Any solution?
    if verbose:
        print("- Loading, calculating offsets with zero/scale", end="... ")

    if extension_uncertainty is not None:
        var_full = np.nan*np.zeros(shape=(ncombine, *sh_comb), dtype=dtype)

    arr_full = np.nan*np.zeros(shape=(ncombine, *sh_comb), dtype=dtype)
    mask_full = np.zeros(shape=(ncombine, *sh_comb), dtype=bool)

    _t = Time.now()
    for i, (_item, _offset, _shape) in enumerate(zip(items, offsets, shapes)):
        # import os
        # import psutil
        # import sys
        # import gc

        # process = psutil.Process(os.getpid())
        # print("0: ", process.memory_info().rss/1.e+9)  # in bytes

        # -- Set slice ------------------------------------------------------------------- #
        # yfu.misc._offsets2slice is introduced much later than the code below was written, so not used here..
        slices = [i]
        # offset & size at each j-th dimension axis
        for offset_j, shape_j in zip(_offset, _shape):
            slices.append(slice(offset_j, offset_j + shape_j, None))

        # -- Load data ------------------------------------------------------------------- #
        # process = psutil.Process(os.getpid())
        # print("1: ", process.memory_info().rss/1.e+9)  # in bytes
        try:
            _data, _var, _mask, _ = load_ccd(
                _item,
                trimsec=trimsec,
                ccddata=False,
                extension=extension,
                extension_mask=e_m,
                extension_uncertainty=e_u,
                full=True
            )
        except TypeError:
            if isinstance(_item, CCDData):
                _data = _item.data.copy()
                if _item.mask is None:
                    _mask = np.zeros(_data.shape, dtype=bool)
                else:
                    _mask = _item.mask.copy()
                _var = None if _item.uncertainty is None else _item.uncertainty.copy()
            else:
                raise ValueError("Each item is not path-like or CCDData.")

        if mask is not None:
            _mask |= mask[i, ]

        # process = psutil.Process(os.getpid())
        # print("2: ", process.memory_info().rss/1.e+9)  # in bytes
        # local_vars = list(locals().items())
        # for var, obj in local_vars:
        #     print(var, sys.getsizeof(obj))

        # -- zero and scale -------------------------------------------------------------- #
        # better to calculate here than from full array, as the
        # latter may contain too many NaNs due to offest shifting.
        # TODO: let get_zsw to get functionals for zsw, so _set_calc_zsw
        # will not be repeateded for every iteration.
        if extract_exptime:
            _scale = scales[i]
        else:  # e.g., "med"
            _scale = scale
        _z, _s, _w = get_zsw(
            arr=np.array(_data[None, :]),  # make a fake (N+1)-D array
            zero=zero,
            scale=_scale,
            weight=weight,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_to_0th=False,   # to retain original zero
            scale_to_0th=False,  # to retain original scale
            zero_section=zero_section,
            scale_section=scale_section
        )
        zeros[i] = _z[0]
        scales[i] = _s[0]
        weights[i] = _w[0]

        # -- Insertion ------------------------------------------------------------------- #
        arr_full[tuple(slices)] = _data
        mask_full[tuple(slices)] = _mask
        if _var is not None:
            var_full[slices] = _var

    if verbose:
        print("Done.")
        if isinstance(items[0], str):
            print()
            print("-"*80)
            print("{:^45s}|{:^9s}|{:^9s}|{:^9s}".format("input", "zero", "scale", "weight"))
            print("-"*80)
            for item, z, s, w in zip(items, zeros, scales, weights):
                print("{:>45s}|{:3e}|{:3e}|{:3e}".format(item[-45:], z, s, w))
            print("-"*80)
            print()
        else:
            pass
    # ------------------------------------------------------------------------------------ #

    cmt2hdr(hdr0, 'h', t_ref=_t, verbose=verbose,
                  s=f"Loaded {ncombine} FITS, calculated zero, scale, weights")

    # == Combine with rejection! ========================================================= #
    _t = Time.now()

    comb = ndcombine(
        arr=arr_full,
        mask=mask_full,
        copy=False,  # No need to retain arr_full.
        combine=combine,
        reject=reject_fullname,
        scale=scales,    # it is scales , NOT scale , as it was updated above.
        zero=zeros,      # it is zeros  , NOT zero  , as it was updated above.
        weight=weights,  # it is weights, NOT weight, as it was updated above.
        zero_to_0th=zero_to_0th,
        scale_to_0th=scale_to_0th,
        scale_kw=scale_kw,
        zero_kw=zero_kw,
        thresholds=thresholds,
        n_minmax=n_minmax,
        nkeep=nkeep,
        maxrej=maxrej,
        cenfunc=cenfunc,
        sigma=sigma,
        maxiters=maxiters,
        ddof=ddof,
        rdnoise=rds,  # it is rds, not rdnoise, as it was updated above.
        gain=gns,     # it is gns, not gain   , as it was updated above.
        snoise=sns,   # it is sns, not snoise , as it was updated above.
        pclip=pclip,
        irafmode=irafmode,
        full=full,
        return_variance=return_variance,
        verbose=verbose
    )

    if full:  # unpack the output
        comb, err, mask_rej, mask_thresh, low, upp, nit, rejcode = comb
        mask_total = mask_full | mask_thresh | mask_rej


    # == Update header properly ========================================================== #
    # Update WCS or PHYSICAL keywords so that "lock frame wcs", etc, on SAO
    # ds9, for example, to give proper visualization:
    if use_wcs:  # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            hdr0[f"CRPIX{i}"] += offsets[0][ndim - i]

    if use_phy:  # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            hdr0[f"LTV{i}"] += offsets[0][ndim - i]

    _update_hdr(hdr0, ncombine, imcmb_key=imcmb_key, imcmb_val=imcmb_val,
                offset_mode=offset_mode, offsets=offsets, zeros=zeros, scales=scales,
                weights=weights,)

    try:
        unit = hdr0["BUNIT"].lower()
    except (KeyError, IndexError):
        unit = 'adu'

    cmt2hdr(hdr0, 'h', t_ref=_t, verbose=verbose, s="Rejection and combination done")
    comb = comb.astype(dtype)
    comb = CCDData(data=comb, header=hdr0, unit=unit)

    if verbose:
        print("\n- Writing output FITS", end="... ")

    # == Save FITS files ================================================================= #
    write_kw = dict(output_verify=output_verify, overwrite=overwrite, checksum=checksum)
    if output is not None:
        try:
            comb.write(output, **write_kw)
        except VerifyError:
            raise VerifyError("Use output_verify='fix'")

    if output_err is not None:
        err = err.astype(dtype_err)
        write2fits(err, hdr0, output_err, return_ccd=False, **write_kw)

    if output_low is not None:
        low = low.astype(dtype) if dtype_low is None else low.astype(dtype_low)
        write2fits(low, hdr0, output_low, return_ccd=False, **write_kw)

    if output_upp is not None:
        upp = low.astype(dtype) if dtype_upp is None else upp.astype(dtype_upp)
        write2fits(upp, hdr0, output_upp, return_ccd=False, **write_kw)

    if output_nrej is not None:  # Do this BEFORE output_mask!!
        nrej = np.count_nonzero(mask_total, axis=0).astype(int_dtype)
        write2fits(nrej, hdr0, output_nrej, return_ccd=False, **write_kw)

    if output_mask is not None:  # Do this AFTER output_nrej!!
        # FITS does not accept boolean. We need uint8.
        write2fits(mask_total.astype(np.uint8),
                   hdr0, output_mask, return_ccd=False, **write_kw)

    if output_rejcode is not None:
        write2fits(rejcode, hdr0, output_rejcode, return_ccd=False, **write_kw)

    if verbose:
        print("Done.")

    # == Return memroy... ================================================================ #
    del hdr0, arr_full, mask_full

    # == Write logfile =================================================================== #
    if logfile is not None:
        if verbose:
            print("\n- Writing summary table", end='...')
        # Use astropy table rather than import pandas
        for name in ['scales', 'zeros', 'weights']:
            table_dict[name] = eval(f'list({name})')

        table = Table(table_dict)
        table['gains'] = gns
        table['readnoises'] = rds
        table['snoises'] = sns
        # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            table[f'offset{i}'] = offsets[:, ndim - i]
        table.write(logfile, format='csv')
        if verbose:
            print("Done.")

    if verbose:
        _t2 = Time.now()
        print()
        print(_t2.iso, f"(TOTAL dt = {(_t2 - _t1).sec:5.3} sec)")

    # == Return ========================================================================== #
    if full:
        if return_dict:
            return dict(comb=comb,
                        err=err,
                        mask_total=mask_total,
                        mask_rej=mask_rej,
                        mask_thresh=mask_thresh,
                        low=low,
                        upp=upp,
                        nit=nit,
                        rejcode=rejcode
                        )
        else:
            return (comb, err, mask_total, mask_rej, mask_thresh,
                    low, upp, nit, rejcode)
    else:
        return comb


imcombine.__doc__ = '''A helper function for ndcombine to cope with FITS files.

    {}

    Parameters
    ----------

    inputs : glob pattern, list-like of path-like, list-like of CCDData-like
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or list of
        files (each element must be path-like or CCDData). Although it is not a
        good idea, a mixed list of CCDData and paths to the files is also
        acceptable. For the purpose of imcombine function, the best use is to
        use the `~glob` pattern or list of paths.

    mask : ndarray, optional.
        The mask of bad pixels. If given, it must satisfy
        ``mask.shape[0]`` identical to the number of images.

        .. note::
            If the user ever want to use masking, it's more convenient to use
            ``'MASK'`` extension to the FITS files or replace bad pixel to very
            large or small numbers and use `thresholds`.

    extension, extension_uncertainty, extension_mask : int, str, (str, int)
        The extension of FITS, uncertainty, and mask to be used. It can be
        given as integer (0-indexing) of the extension, ``EXTNAME`` (single
        str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used. If
        `extension_uncertainty` or `extension_mask` is `None` (default),
        uncertainty and mask are all ignored (turned off). Currently
        error-propagation or weighted combine is not supported, so only
        `extension_mask` can give difference to the output.

    {}

    {}

    imcmb_key : str
        The thing to add as ``IMCMBnnn`` in the output FITS file header. If
        ``"$I"``, following the default of IRAF, the file's name will be added.
        Otherwise, it should be a header keyword. If the key does not exist in
        ``nnn``-th file, a null string will be added. If a null string
        (``imcmb_key=""``), it does not set the ``IMCMBnnn`` keywords nor
        deletes any existing keyword.

        .. warning::
            If more than 999 files are combined, only the first 999 files will
            be recorded in the header.

    exposure_key : str, optional.
        The header keyword which contains the information about the exposure
        time of each FITS file. This is used only if scaling is done for
        exposure time (see `scale`).

    irafmode : bool, optional.
        Whether to use IRAF-like pixel restoration scheme.

    output : path-like, optional
        The path to the final combined FITS file. It has dtype of `dtype` and
        dimension identical to each input image. Optional keyword arguments for
        ``fits.writeto()`` can be provided as ``**kwargs``.

    output_xxx : path-like, optional
        The output path to the mask, number of rejected pixels at each
        position, final ``nanstd(combined, ddof=ddof, axis=0)`` (if
        `return_variance` is `False`) or ``nanvar(combined, ddof=ddof,
        axis=0)`` (if `return_variance` is `True`) result, lower and upper
        bounds for rejection, and the integer codes for the rejection algorithm
        (see `mask_total`, `mask_rej`, `err`, `low`, `upp`, and `rejcode` in
        Returns.)

    return_dict : bool, optional.
        Whether to return the results as dict (works only if ``full=True``).

    Returns
    -------
    Returns the followings depending on `full` and `return_dict`.

    comb : `astropy.nddata.CCDData` (dtype `dtype`)
        The combined data.

    {}

    {}
    '''.format(docstrings.NDCOMB_NOT_IMPLEMENTED(indent=4),
               docstrings.OFFSETS_LONG(indent=4),
               docstrings.NDCOMB_PARAMETERS_COMMON(indent=4),
               docstrings.NDCOMB_RETURNS_COMMON(indent=4),
               docstrings.IMCOMBINE_LINK(indent=4))


# ---------------------------------------------------------------------------------------- #
def ndcombine(
        arr,
        mask=None,
        copy=True,
        blank=np.nan,
        offsets=None,
        thresholds=[-np.inf, np.inf],
        zero=None,
        scale=None,
        weight=None,
        zero_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        zero_to_0th=True,
        scale_to_0th=True,
        zero_section=None,
        scale_section=None,
        reject=None,
        cenfunc='median',
        sigma=[3., 3.],
        maxiters=3,
        ddof=1,
        nkeep=1,
        maxrej=None,
        n_minmax=[1, 1],
        rdnoise=0.,
        gain=1.,
        snoise=0.,
        pclip=-0.5,
        combine='average',
        dtype='float32',
        memlimit=2.5e+9,
        irafmode=True,
        verbose=False,
        full=False,
        return_variance=False
):
    if copy:
        arr = arr.copy()

    if np.array(arr).ndim == 1:
        raise ValueError("1-D array combination is not supported!")

    _mask = _set_mask(arr, mask)  # _mask = propagated through this function.
    sigma_lower, sigma_upper = _set_sigma(sigma)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc)
    reject_fullname = _set_reject_name(reject)
    maxiters = int(maxiters)
    ddof = int(ddof)

    combfunc = _set_combfunc(combine, nameonly=False, nan=True)

    if verbose and reject is not None:
        print("- Rejection")
        if thresholds != [-np.inf, np.inf]:
            print(f"-- thresholds (low, upp) = {thresholds}")
        print(f"-- {reject=} ({irafmode=})")
        print(f"--       params: {nkeep=}, {maxrej=}, {maxiters=}, {cenfunc=}")
        if reject_fullname == "sigclip":
            print(f"  (for sigclip): {sigma=}, {ddof=}")
        elif reject_fullname == "ccdclip":
            print(f"  (for ccdclip): {gain=}, {rdnoise=}, {snoise=}")
        # elif reject_fullnme == "pclip":
        #   print(f"    (for pclip)  : spclip={pclip}")
        # elif reject_fullname == "minmax":
        # print(f" (for minmaxclip): n_minmax={n_minmax}")

    # == 01 - Thresholding + Initial masking ============================================= #
    # Updating mask: _mask = _mask | mask_thresh
    mask_thresh = _set_thresh_mask(
        arr=arr,
        mask=_mask,
        thresholds=thresholds,
        update_mask=True
    )

    # if safemode:
    #     # Backup the pixels which are rejected by thresholding and # initial
    #     mask for future restoration (see below) for debugging # purpose.
    #     backup_thresh = arr[mask_thresh]
    #     backup_thresh_inmask = arr[_mask]

    # TODO: remove this np.nan and instead, let `get_zsw` to accept mask.
    arr[_mask] = np.nan
    # ------------------------------------------------------------------------------------ #

    # == 02 - Calculate zero, scale, weights ============================================= #
    # This should be done before rejection but after threshold masking..
    zeros, scales, weights = get_zsw(
        arr=arr,
        zero=zero,
        scale=scale,
        weight=weight,
        zero_kw=zero_kw,
        scale_kw=scale_kw,
        zero_to_0th=zero_to_0th,
        scale_to_0th=scale_to_0th,
        zero_section=zero_section,
        scale_section=scale_section
    )
    arr = do_zs(arr, zeros=zeros, scales=scales)
    # ------------------------------------------------------------------------------------ #

    # == 02 - Rejection ================================================================== #
    if isinstance(reject_fullname, str):
        if reject_fullname == 'sigclip':
            _mask_rej = sigclip_mask(
                arr,
                mask=_mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                maxiters=maxiters,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                axis=0,
                irafmode=irafmode,
                full=full
            )
        elif reject_fullname == 'minmax':
            _mask_rej = minmax_mask(
                arr,
                mask=_mask,
                n_minmax=n_minmax,
                full=full
            )
        elif reject_fullname == 'ccdclip':
            _mask_rej = ccdclip_mask(
                arr,
                mask=_mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                scale_ref=np.mean(scales),
                zero_ref=np.mean(zeros),
                maxiters=maxiters,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                axis=0,
                gain=gain,
                rdnoise=rdnoise,
                snoise=snoise,
                irafmode=irafmode,
                full=full
            )
        elif reject_fullname == 'pclip':
            pass
        else:
            raise ValueError("reject not understood.")
        if full:
            _mask_rej, low, upp, nit, rejcode = _mask_rej
        # _mask is a subset of _mask_rej, so to extract pixels which are
        # masked PURELY due to the rejection is:
        mask_rej = _mask_rej ^ _mask
    elif reject_fullname is None:
        mask_rej = _set_mask(arr, None)
        if full:
            low = bn.nanmin(arr, axis=0)
            upp = bn.nanmax(arr, axis=0)
            nit = None
            rejcode = None
    else:
        raise ValueError("reject not understood.")

    if reject is not None and verbose:
        print("Done.")

    _mask |= mask_rej

    # ------------------------------------------------------------------------------------ #

    # TODO: add "grow" rejection here?

    # == 03 - combine ==================================================================== #
    # Replace rejected / masked pixel to NaN and backup for debugging purpose. This is done to reduce
    # memory (instead of doing _arr = arr.copy())
    # backup_nan = arr[_mask]
    if verbose:
        print("- Combining")
        print(f"-- combine = {combine}")
    arr[_mask] = np.nan

    # Combine and calc sigma
    comb = combfunc(arr, axis=0)
    if verbose:
        print("Done.")

    # Restore NaN-replaced pixels of arr for debugging purpose.
    # arr[_mask] = backup_nan
    # arr[mask_thresh] = backup_thresh_inmask
    if full:
        if verbose:
            print( "- Error calculation")
            print("-- to skip this, use `full=False`")
            print(f"-- return_variance={return_variance}, ddof={ddof}")
        if return_variance:
            err = bn.nanvar(arr, ddof=ddof, axis=0)
        else:
            err = bn.nanstd(arr, ddof=ddof, axis=0)
        if verbose:
            print("Done.")
        return comb, err, mask_rej, mask_thresh, low, upp, nit, rejcode
    else:
        return comb


ndcombine.__doc__ = ''' Combines the given arr assuming no additional offsets.

    {}
    #. offsets is not implemented to ndcombine (only to fitscombine).

    Parameters
    ----------
    arr : ndarray
        The array to be combined along axis 0.

    mask : ndarray, optional.
        The mask of bad pixels. If given, it must satisfy ``mask.shape[0]``
        identical to the number of images.

    copy : bool, optional.
        Whether to copy the input array. Set to `True` if you want to keep the
        original array unchanged. Otherwise, the original array may be
        destroyed.

    {}

    {}

    Returns
    -------
    comb : ndarray
        The combined array.

    {}

    {}
    '''.format(docstrings.NDCOMB_NOT_IMPLEMENTED(indent=4),
               docstrings.OFFSETS_SHORT(indent=4),
               docstrings.NDCOMB_PARAMETERS_COMMON(indent=4),
               docstrings.NDCOMB_RETURNS_COMMON(indent=4),
               docstrings.IMCOMBINE_LINK(indent=4)
               )
