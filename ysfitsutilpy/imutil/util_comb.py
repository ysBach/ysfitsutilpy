import bottleneck as bn
import numpy as np
from astropy.stats import sigma_clipped_stats
from ccdproc.utils.slices import slice_from_string
from .util_lmedian import lmedian, nanlmedian

__all__ = [
    'do_zs', 'get_zsw', '_set_int_dtype', '_get_dtype_limits',
    '_setup_reject', '_set_reject_name',
    '_set_mask', '_set_sigma', '_set_keeprej', '_set_minmax', '_set_thresh_mask',
    '_set_gain_rdns', '_set_cenfunc', '_set_combfunc'
]


def do_zs(arr, zeros, scales, copy=False):
    if copy:
        arr = arr.copy()
    # below two took less than 10 us for 100 images
    all0 = np.all(zeros == 0) or zeros is None
    all1 = np.all(scales == 1) or scales is None
    if not all0 and not all1:  # both zero and scale
        for i in range(arr.shape[0]):
            arr[i, ] = (arr[i, ] - zeros[i])/scales[i]
    elif not all0 and all1:  # zero only
        for i in range(arr.shape[0]):
            arr[i, ] = arr[i, ] - zeros[i]
    elif all0 and not all1:  # scale only
        for i in range(arr.shape[0]):
            arr[i, ] = arr[i, ]/scales[i]
    # Times:
    #   (np.random.normal(size=(1000,1000)) - 0)/1   21.6 ms +- 730 us
    #   np.random.normal(size=(1000,1000))           20.1 ms +- 197 us
    # Also note that both of the below show nearly identical timing
    # (https://stackoverflow.com/a/45895371/7199629)
    #   (data3d_nan.T / scale).T
    #   np.array([data3d_nan[i, ] / _s for i, _s in enumerate(scale)])
    # 7.58 +/- 0.3 ms, 8.46 +/- 1.88 ms, respectively. Though the former
    # is concise, latter is more explicit, so I used the latter.
    return arr


def get_zsw(
        arr,
        zero,
        scale,
        weight,
        zero_kw,
        scale_kw,
        zero_to_0th,
        scale_to_0th,
        zero_section,
        scale_section
):
    '''
    Originally this was designed to get zero/scale values for (N+1)-D stacked
    array for N-D images. However, imcombine uses it only for one image at a
    time.
    '''
    # TODO: add sigma-clipped mean, med, std as scale, zero, or weight.
    def _nanfun2nonnan(fun):
        '''
        Returns
        -------
        nonnan_function: The function without NaN policy
        check_isfinite: Whether the `isfinite` must be tested.
        '''
        if fun == bn.nanmean:
            return np.mean
        elif fun == bn.nanmedian:
            # There IS bn.median, but that doesn't accept tuple axis...
            return np.median
        elif fun == bn.nansum:
            return np.sum
        else:
            return fun

    def _set_calc_zsw(arr, zero_scale_weight, zsw_kw={}):
        if isinstance(zero_scale_weight, str):
            zswstr = zero_scale_weight.lower()
            calc_zsw = True
            zsw = []
            if zswstr in ['avg', 'average', 'mean']:
                calcfun = bn.nanmean
            elif zswstr in ['med', 'medi', 'median']:
                calcfun = bn.nanmedian
            # sigclip calcfun's below have dummy `axis` parameter because if
            # not, the ``zsw = fun_simple(data, axis=None)`` code below raises
            # unwanted TypeError, so that `redo` is always `True`.
            elif zswstr in ['avg_sc', 'average_sc', 'mean_sc']:
                _ = zsw_kw.pop('axis', None)  # if exist `axis`, remove it.
                calcfun = lambda x, axis: sigma_clipped_stats(x, axis=None, **zsw_kw)[0]
            elif zswstr in ['med_sc', 'medi_sc', 'median_sc']:
                _ = zsw_kw.pop('axis', None)  # if exist `axis`, remove it.
                calcfun = lambda x, axis: sigma_clipped_stats(x, axis=None, **zsw_kw)[1]
            else:
                raise ValueError(f"zero/scale/weight ({zero_scale_weight}) not understood")

        else:
            zsw = np.array(zero_scale_weight)
            if zsw.size == arr.shape[0]:
                zsw = zsw.flatten()
            else:
                raise ValueError("If scale/zero/weight are array-like, they must be of "
                                 + "size = arr.shape[0] (number of images to combine).")
            calc_zsw = False
            zsw = np.array(zero_scale_weight).ravel()  # to make 0-D to 1-D
            calcfun = None
        return calc_zsw, zsw, calcfun

    def _calc_zsw(fun, arr, section=None):
        fun_simple = _nanfun2nonnan(fun)
        if section is not None:
            try:
                sl = slice_from_string(section, fits_convention=True)
            except (AttributeError, ValueError):  # i.e., int-convertible
                sl = int(section)
        else:
            sl = [slice(None)]*(arr.ndim - 1)

        # TODO: There must be a better way to do this "redo" strategy...
        try:
            # If converted to numpy, this must work without error:
            if isinstance(sl, int):
                zsw = [fun_simple(arr[i].ravel()[::sl], axis=None)
                       for i in range(arr.shape[0])]
            else:
                zsw = fun_simple(arr[(slice(None), *sl)],
                                 axis=tuple(np.arange(arr.ndim)[1:]))
            redo = ~np.all(np.isfinite(zsw))
        except TypeError:  # does not accept tuple axis
            redo = True

        if redo:
            zsw = []
            if isinstance(sl, int):
                try:
                    zsw = [(fun(arr[i].ravel()[::sl], axis=None))
                           for i in range(arr.shape[0])]
                except TypeError:  # original fun has no `axis` parameter
                    zsw = [(fun(arr[i].ravel()[::sl]))
                           for i in range(arr.shape[0])]
            else:
                try:
                    zsw = [(fun(arr[(i, *sl)], axis=None)) for i in range(arr.shape[0])]
                except TypeError:  # original fun has no `axis` parameter
                    zsw = [(fun(arr[(i, *sl)])) for i in range(arr.shape[0])]
        # raise ValueError()
        return np.atleast_1d(zsw)

    if zero is None:
        zeros = np.zeros(arr.shape[0])
        calc_z = False
        fun_z = None
    else:
        calc_z, zeros, fun_z = _set_calc_zsw(arr, zero, zero_kw)

    if scale is None:
        scales = np.ones(arr.shape[0])
        calc_s = False
        fun_s = None
    else:
        calc_s, scales, fun_s = _set_calc_zsw(arr, scale, scale_kw)

    if weight is None:
        weights = np.ones(arr.shape[0])
        calc_w = False
        fun_w = None
    else:
        calc_w, weights, fun_w = _set_calc_zsw(arr, weight)

    if calc_z:
        zeros = _calc_zsw(fun_z, arr, section=zero_section)

    if calc_s:
        if fun_s is not None and fun_z is not None and fun_s == fun_z:
            scales = zeros.copy()
        else:
            scales = _calc_zsw(fun_s, arr, section=scale_section)

    if calc_w:  # TODO: Needs update to match IRAF's...
        if fun_w is not None and fun_s is not None and fun_w == fun_s:
            weights = scales.copy()
        elif fun_w is not None and fun_z is not None and fun_w == fun_z:
            weights = zeros.copy()
        else:
            weights = _calc_zsw(fun_w, arr, section=scale_section)

    if zero_to_0th:
        zeros -= zeros[0]

    if scale_to_0th:
        scales /= scales[0]  # So that normalize 1.000 for the 0-th image.

    return zeros, scales, weights


def _setup_reject(arr, mask, nkeep, maxrej, cenfunc):
    ''' Does the common default setting for all rejection algorithms.
    '''
    _arr = arr.copy()
    # NOTE: mask_nan is propagation of input mask && nonfinite(arr)
    _arr[_set_mask(_arr, mask)] = np.nan
    mask_nan = ~np.isfinite(_arr)

    numnan = np.count_nonzero(mask_nan, axis=0)
    nkeep, maxrej = _set_keeprej(_arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc, nameonly=False, nan=True)

    ncombine = _arr.shape[0]
    int_dtype = _set_int_dtype(ncombine)
    # n_iteration : in uint8
    nit = np.ones(_arr.shape[1:], dtype=np.uint8)
    # n_rejected/n_finite_old : all in int_dtype determined by _set_int_dtype
    n_nan = np.count_nonzero(mask_nan, axis=0).astype(int_dtype)
    n_finite_old = ncombine - n_nan  # num of remaining pixels

    # Initiate with min/max, not something like std.
    low = bn.nanmin(_arr, axis=0)
    upp = bn.nanmax(_arr, axis=0)
    low_new = low.copy()
    upp_new = upp.copy()

    no_nkeep = nkeep is None
    no_maxrej = maxrej is None
    if no_nkeep and no_maxrej:
        mask_nkeep = np.zeros(_arr.shape[1:], dtype=bool)
        mask_maxrej = np.zeros(_arr.shape[1:], dtype=bool)
        mask_pix = np.zeros(_arr.shape[1:], dtype=bool)
    elif not no_nkeep and no_maxrej:
        mask_nkeep = ((ncombine - numnan) < nkeep)
        mask_maxrej = np.zeros(_arr.shape[1:], dtype=bool)
        mask_pix = mask_nkeep.copy()
    elif not no_maxrej and no_nkeep:
        mask_nkeep = np.zeros(_arr.shape[1:], dtype=bool)
        mask_maxrej = (n_nan > maxrej)
        mask_pix = mask_maxrej.copy()
    else:
        mask_nkeep = ((ncombine - numnan) < nkeep)
        mask_maxrej = (n_nan > maxrej)
        mask_pix = mask_nkeep | mask_maxrej

    return (_arr,
            [mask_nan, mask_nkeep, mask_maxrej, mask_pix],
            [nkeep, maxrej],
            cenfunc,
            [nit, ncombine, n_finite_old],
            [low, upp, low_new, upp_new]
            )


def _calculate_step_sizes(x_size, y_size, num_chunks):
    """ Calculate the strides in x and y.
    Notes
    -----
    Calculate the strides in x and y to achieve at least the
    `num_chunks` pieces.

    Direct copy from ccdproc:
    https://github.com/astropy/ccdproc/blob/b9ec64dfb59aac1d9ca500ad172c4eb31ec305f8/ccdproc/combiner.py#L500
    """
    # First we try to split only along fast x axis
    xstep = max(1, int(x_size / num_chunks))

    # More chunks are needed only if xstep gives us fewer chunks than
    # requested.
    x_chunks = int(x_size / xstep)

    if x_chunks >= num_chunks:
        ystep = y_size
    else:
        # The x and y loops are nested, so the number of chunks
        # is multiplicative, not additive. Calculate the number
        # of y chunks we need to get at num_chunks.
        y_chunks = int(num_chunks / x_chunks) + 1
        ystep = max(1, int(y_size / y_chunks))

    return xstep, ystep


def _set_int_dtype(ncombine):
    if ncombine < 255:
        int_dtype = np.uint8
    elif ncombine > 65535:
        int_dtype = np.uint32
    else:
        int_dtype = np.uint16
    return int_dtype


def _get_dtype_limits(dtype):
    try:
        info = np.iinfo(dtype)
    except ValueError:
        # I don't use np.inf, because of a fear that some functions,
        # e.g., bn.partition mayy not work for these, as well as np.nan.
        info = np.finfo(dtype)
    return (info.min, info.max)


def _set_mask(arr, mask):
    if mask is None:
        mask = np.zeros_like(arr, dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)

    if arr.shape != mask.shape:
        raise ValueError("The input array and mask must have the identical shape. "
                         + f"Now arr.shape = {arr.shape} and mask.shape = {mask.shape}.")
    return mask


def _set_sigma(sigma, sigma_lower=None, sigma_upper=None):
    sigma = np.atleast_1d(sigma)
    if sigma.shape[0] == 1:
        if sigma_lower is None:
            sigma_lower = float(sigma)
        if sigma_upper is None:
            sigma_upper = float(sigma)
    elif sigma.shape[0] == 2:
        if sigma_lower is None:
            sigma_lower = float(sigma[0])
        if sigma_upper is None:
            sigma_upper = float(sigma[1])
    else:
        raise ValueError("sigma must have shape[0] of 1 or 2.")
    return sigma_lower, sigma_upper


def _set_keeprej(arr, nkeep, maxrej, axis):
    if nkeep is None:
        nkeep = 0
    elif nkeep < 1:
        nkeep = np.around(nkeep*arr.shape[axis])

    if maxrej is None:
        maxrej = arr.shape[axis]
    elif maxrej < 1:
        maxrej = np.around(maxrej*arr.shape[axis])

    return int(nkeep), int(maxrej)


def _set_minmax(arr, n_minmax, axis):
    n_minmax = np.atleast_1d(n_minmax)
    if n_minmax.shape[0] == 1:
        q_low = float(n_minmax[0])
        q_upp = float(n_minmax[0])
    elif n_minmax.shape[0] == 2:
        q_low = float(n_minmax[0])
        q_upp = float(n_minmax[1])
    else:
        raise ValueError("n_minmax must have shape[0] of 1 or 2.")

    nimages = arr.shape[axis]

    if q_low >= 1:  # if given as number of pixels
        q_low = q_low/nimages
    # else: already given as fraction

    if q_upp >= 1:  # if given as number of pixels
        q_upp = q_upp/nimages
    # else: already given as fraction

    if q_low + q_upp > 1:
        raise ValueError("min/max rejection more than images!")

    return q_low, q_upp


def _set_thresh_mask(arr, mask, thresholds, update_mask=True):
    if (thresholds[0] != -np.inf) and (thresholds[1] != np.inf):
        mask_thresh = (arr < thresholds[0]) | (arr > thresholds[1])
        if update_mask:
            mask |= mask_thresh
    elif (thresholds[0] == -np.inf):
        mask_thresh = (arr > thresholds[1])
        if update_mask:
            mask |= mask_thresh
    elif (thresholds[1] == np.inf):
        mask_thresh = (arr < thresholds[0])
        if update_mask:
            mask |= mask_thresh
    else :
        mask_thresh = np.zeros(arr.shape).astype(bool)
        # no need to update _mask

    return mask_thresh


def _set_gain_rdns(gain_or_rdnoise, ncombine, dtype='float32'):
    extract = False
    if isinstance(gain_or_rdnoise, str):
        extract = True
        arr = np.ones(ncombine, dtype=dtype)
    else:
        arr = np.atleast_1d(gain_or_rdnoise).astype(dtype)
        if arr.size == ncombine:
            arr = arr.ravel()
            if not np.all(np.isfinite(arr)):
                raise ValueError("ccdclip parameters contains NaN")
        elif arr.size == 1:
            if not np.isfinite(arr):
                raise ValueError("ccdclip parameters contains NaN")
            arr = arr*np.ones(ncombine, dtype=dtype)
        else:
            raise ValueError("ccdclip parameters size not equal to ncombine.")
    return extract, arr


def _set_cenfunc(cenfunc, shorten=False, nameonly=True, nan=True):
    if cenfunc is None:
        return None
    elif cenfunc in ['med', 'medi', 'median']:
        if nameonly:
            return 'med' if shorten else 'median'
        else:
            return bn.nanmedian if nan else bn.median
    elif cenfunc in ['avg', 'average', 'mean']:
        if nameonly:
            return 'avg' if shorten else 'average'
        else:
            return bn.nanmean if nan else np.mean
    elif cenfunc in ['lmed', 'lmd', 'lmedian']:
        if nameonly:
            return 'lmd' if shorten else 'lmedian'
        else:
            return nanlmedian if nan else lmedian
    else:
        raise ValueError('cenfunc not understood')


def _set_combfunc(combfunc, shorten=False, nameonly=True, nan=True):
    ''' Identical to _set_cenfunc, except 'sum' is allowed.
    '''
    if combfunc is None:
        return None
    elif combfunc in ['med', 'medi', 'median']:
        if nameonly:
            return 'med' if shorten else 'median'
        else:
            return bn.nanmedian if nan else bn.median
    elif combfunc in ['avg', 'average', 'mean']:
        if nameonly:
            return 'avg' if shorten else 'average'
        else:
            return bn.nanmean if nan else np.mean
    elif combfunc in ['sum']:
        if nameonly:
            return 'sum'
        else:
            return bn.nansum if nan else np.sum
    elif combfunc in ['lmed', 'lmd', 'lmedian']:
        if nameonly:
            return 'lmd' if shorten else 'lmedian'
        else:
            return nanlmedian if nan else lmedian
    else:
        raise ValueError('combfunc not understood')


def _set_reject_name(reject):
    if reject is None:
        return reject

    rej = reject.lower()
    if rej in ['sig', 'sc', 'sigclip', 'sigma', 'sigma clip', 'sigmaclip']:
        return 'sigclip'
    elif rej in ['mm', 'minmax']:
        return 'minmax'
    elif rej in ['ccd', 'ccdclip', 'ccdc']:
        return 'ccdclip'
    elif rej in ['pclip', 'pc', 'percentile']:
        return 'pclip'
