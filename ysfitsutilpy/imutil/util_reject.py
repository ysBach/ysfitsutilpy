import bottleneck as bn
import numpy as np

from . import docstrings
from .util_comb import (_get_dtype_limits, _set_cenfunc, _set_gain_rdns,
                        _set_keeprej, _set_mask, _set_minmax, _set_sigma,
                        _setup_reject, do_zs)

__all__ = ["sigclip_mask", "ccdclip_mask", "minmax_mask"]


try:
    import numexpr as ne
    HAS_NE = True
    NEVAL = ne.evaluate  # "n"umerical "eval"uator
    NPSTR = ""
except ImportError:
    HAS_NE = False
    NEVAL = eval  # "n"umerical "eval"uator
    NPSTR = "np."


def _iter_rej(
        arr,
        mask=None,
        sigma_lower=3.,
        sigma_upper=3.,
        maxiters=5,
        ddof=0,
        nkeep=3,
        maxrej=None,
        cenfunc='median',
        ccdclip=False,
        irafmode=True,
        rdnoise_ref=0.,
        snoise_ref=0.,
        scale_ref=1,
        zero_ref=0
):
    """ The common function for iterative rejection algorithms.

    Parameters
    ----------
    arr : ndarray
        The array to find the mask. It must be gain-corrected if
        ``ccdclip=True``.

    rdnoise_ref, snoise_ref : float
        The representative readnoise and sensitivity noise to estimate the
        error-bar for ``ccdclip=True``.

    scale_ref, zero_ref : float
        The representative scaling and zeroing value to estimate the error-bar
        for ``ccdclip=True``.
    """
    def __calc_censtd(_arr):
        # most are defined in upper _iter_rej function
        cen = cenfunc(_arr, axis=0)
        if ccdclip:  # use abs(pix value) to avoid NaN from negative pixels.
            _evalstr = f"{NPSTR}abs(cen + zero_ref)*scale_ref"
            # restore zeroing & scaling ; then add rdnoise
            _evalstr = f"(1 + snoise_ref)*{_evalstr} + rdnoise_ref**2"
            std = NEVAL(f"{NPSTR}sqrt({_evalstr})")
        else:
            std = bn.nanstd(_arr, axis=0, ddof=ddof)

        return cen, std

    # General setup
    _arr, _masks, keeprej, cenfunc, _nvals, lowupp = _setup_reject(
        arr=arr, mask=mask, nkeep=nkeep, maxrej=maxrej, cenfunc=cenfunc
    )
    mask_nan, mask_nkeep, mask_maxrej, mask_pix = _masks
    nkeep, maxrej = keeprej
    nit, ncombine, n_finite_old = _nvals
    low, upp, low_new, upp_new = lowupp

    nrej = ncombine - n_finite_old
    k = 0
    # mask_pix is where **NO** rejection should occur.
    if (nkeep == 0) and (maxrej == ncombine):
        print("nkeep, maxrej turned off.")
        # no need to check mask_pix iteratively
        while k < maxiters:
            cen, std = __calc_censtd(_arr=_arr)
            low_new[~mask_pix] = (cen - sigma_lower*std)[~mask_pix]
            upp_new[~mask_pix] = (cen + sigma_upper*std)[~mask_pix]

            # In numpy, > or < automatically applies along axis=0!!
            mask_bound = (_arr < low_new) | (_arr > upp_new) | ~np.isfinite(_arr)
            _arr[mask_bound] = np.nan

            n_finite_new = ncombine - np.count_nonzero(mask_bound, axis=0)
            n_change = n_finite_old - n_finite_new
            total_change = np.sum(n_change)

            mask_nochange = (n_change == 0)  # identical to say "max-iter reached"

            # no need to backup
            if total_change == 0:
                break

            # I put the test below because I thought it will be quicker to halt
            # clipping if all pixels are masked. But now I feel testing this in
            # every iteration is an unnecessary overhead for "nearly
            # impossible" situation.
            # - ysBach (2020-10-14 21:15:44 (KST: GMT+09:00))
            # if np.all(mask_pix):
            #     break

            # update only non-masked pixels
            nrej[~mask_pix] = n_change[~mask_pix]
            # update only changed pixels
            nit[~mask_nochange] += 1
            k += 1
            n_finite_old = n_finite_new

    else:
        while k < maxiters:
            cen, std = __calc_censtd(_arr=_arr)
            low_new[~mask_pix] = (cen - sigma_lower*std)[~mask_pix]
            upp_new[~mask_pix] = (cen + sigma_upper*std)[~mask_pix]

            # In numpy, > or < automatically applies along axis=0!!
            mask_bound = (_arr < low_new) | (_arr > upp_new) | ~np.isfinite(_arr)
            _arr[mask_bound] = np.nan

            n_finite_new = ncombine - np.count_nonzero(mask_bound, axis=0)
            n_change = n_finite_old - n_finite_new
            total_change = np.sum(n_change)

            mask_nochange = (n_change == 0)  # identical to say "max-iter reached"
            mask_nkeep = ((ncombine - nrej) < nkeep)
            mask_maxrej = (nrej > maxrej)

            # mask pixel position if any of these happened. Including
            # mask_nochange here will not change results but only spend more
            # time.
            mask_pix = mask_nkeep | mask_maxrej

            # revert to the previous ones if masked.
            # By doing this, pixels which was mask_nkeep now, e.g., will again
            # be True in mask_nkeep in the next iter but unchanged. This should
            # be done at every iteration (unfortunately) because, e.g., if
            # nkeep is very large, excessive rejection may happen for many
            # times, and the restoration CANNOT be done after all the
            # iterations.
            low_new[mask_pix] = low[mask_pix].copy()
            upp_new[mask_pix] = upp[mask_pix].copy()
            low = low_new
            upp = upp_new

            if total_change == 0:
                break

            # I put the test below because I thought it will be quicker to halt
            # clipping if all pixels are masked. But now I feel testing this in
            # every iteration is an unnecessary overhead for
            # "nearly impossible" situation.
            # - ysBach (2020-10-14 21:15:44 (KST: GMT+09:00))
            # if np.all(mask_pix):
            #     break

            # update only non-masked pixels
            nrej[~mask_pix] = n_change[~mask_pix]
            # update only changed pixels
            nit[~mask_nochange] += 1
            k += 1
            n_finite_old = n_finite_new

    mask = mask_nan | (arr < low_new) | (arr > upp_new)

    code = np.zeros(_arr.shape[1:], dtype=np.uint8)
    if (maxiters == 0):
        code += 1
    else:
        code += (2*mask_nochange + 4*mask_nkeep + 8*mask_maxrej).astype(np.uint8)

    if irafmode:
        n_minimum = max(nkeep, ncombine - maxrej)
        if n_minimum > 0:
            try:
                resid = np.abs(_arr - cen)
            except UnboundLocalError:  # cen undefined when maxiters=0
                resid = np.abs(_arr - cenfunc(_arr, axis=0))
            # need this cuz bn.argpartition cannot handle NaN:
            resid[np.isnan(resid)] = _get_dtype_limits(resid.dtype)[1]
            # ^ replace with max of dtype
            # after this, resid is guaranteed to have **NO** NaN values.

            resid_cut = np.max(bn.partition(resid, n_minimum, axis=0)[:n_minimum, ], axis=0)
            mask[resid <= resid_cut] = False

    # Note the mask returned here is mask from rejection PROPAGATED with the
    # input mask. So to extract the pixels masked PURELY from rejection, you
    # need ``mask_output^mask_input`` because the input mask is a subset of the
    # output one.

    return (mask, low, upp, nit, code)


# TODO: let `cenfunc` be function object...?
# **************************************************************************************** #
# *                                    SIGMA-CLIPPING                                    * #
# **************************************************************************************** #
def sigclip_mask(
        arr,
        mask=None,
        sigma=3.,
        sigma_lower=None,
        sigma_upper=None,
        maxiters=5,
        ddof=0,
        nkeep=3,
        maxrej=None,
        cenfunc='median',
        irafmode=False,
        axis=0,
        full=True
):
    if axis != 0:
        raise ValueError("Currently only axis=0 is supported")

    mask = _set_mask(arr, mask)
    sigma_lower, sigma_upper = _set_sigma(sigma, sigma_lower, sigma_upper)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis)
    cenfunc = _set_cenfunc(cenfunc)
    maxiters = int(maxiters)
    ddof = int(ddof)

    o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
        arr=arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        ddof=ddof,
        nkeep=nkeep,
        maxrej=maxrej,
        cenfunc=cenfunc,
        ccdclip=False,
        irafmode=irafmode
    )
    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask


sigclip_mask.__doc__ = ''' Finds masks of `arr` by sigma-clipping.

    Parameters
    ----------
    arr : ndarray
        The array to be subjected for masking. `arr` and `mask` must
        have the identical shape.

    {}

    Returns
    -------
    {}

    '''.format(docstrings.REJECT_PARAMETERS_COMMON(indent=4),
               docstrings.REJECT_PARAMETERS_SIGMA(indent=4),
               docstrings.REJECT_RETURNS_SIGMA(indent=4))


# **************************************************************************************** #
# *                                    MINMAX CLIPPING                                   * #
# **************************************************************************************** #
def _minmax(arr, mask=None, q_low=0, q_upp=0,
            calc_low=True, calc_upp=True):
    # General setup (nkeep and maxrej as dummy)
    _arr, _masks, _, _, _nvals, _lowupp = _setup_reject(
        arr=arr, mask=mask, nkeep=1, maxrej=None, cenfunc=None
    )
    # mask == input_mask | ~isfinite(arr)
    mask, _, _, mask_skiprej = _masks  # nkeep and maxrej not used in MINMAX.
    _, ncombine, n_old = _nvals  # nit is not used in MINMAX.
    low, upp, _, _ = _lowupp  # low_new, upp_new are not used in MINMAX.

    # adding 0.001 following IRAF
    n_rej_low = (n_old * q_low + 0.001).astype(int)  # 2-D array
    n_rej_upp = (n_old * q_upp + 0.001).astype(int)  # 2-D array
    n_low = np.max(n_rej_low)  # only ~ 0.1 ms for 1k x 1k array of int
    n_upp = np.max(n_rej_upp)

    # NOTE: Below, the `partition` by bottleneck is only marginally faster than
    #   numpy at best. Sometimes I found it slower than numpy. Furthermore, it does not
    #   support negative `kth`. Thus, I use numpy for simplicity without losing much speed.
    # get lower value map
    if n_low != 0:
        _arr[mask] = np.inf  # replace with largest value
        lowidx = np.argpartition(_arr, kth=n_low, axis=0)[:n_low, ]
        np.put_along_axis(mask, lowidx, True, axis=0)
        if calc_low:
            low = np.min(_arr, axis=0)
    # if n_low == 0: do not update mask & use `low` obtained above.

    # remove upper values
    if n_upp != 0:
        _arr[mask] = -np.inf  # replace with lowest values
        uppidx = np.argpartition(_arr, kth=-n_upp, axis=0)[-n_upp:, ]
        np.put_along_axis(mask, uppidx, True, axis=0)
        if calc_upp:
            upp = np.max(_arr, axis=0)
    # if n_upp == 0: use `upp` obtained above.

    code = np.zeros(_arr.shape[1:], dtype=np.uint8)
    no_rej = (n_rej_low == 0) | (n_rej_upp == 0)
    code += (1*no_rej).astype(np.uint8)

    return (mask, low, upp, 1, code)


def minmax_mask(
        arr,
        mask=None,
        n_minmax=[1, 1],
        full=True
):
    mask = _set_mask(arr, mask)
    q_low, q_upp = _set_minmax(arr, n_minmax, axis=0)
    o_mask, o_low, o_upp, o_nit, o_code = _minmax(
        arr,
        mask=mask,
        q_low=q_low,
        q_upp=q_upp,
        calc_low=full,
        calc_upp=full
    )
    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask


minmax_mask.__doc__ = ''' Finds masks of `arr` after rejecting `n_minmax` pixels.

    Parameters
    ----------
    arr : ndarray
        The array to be subjected for masking. `arr` and `mask` must have the
        identical shape. It must be in DN, i.e., **not** gain corrected.

    {}


    Returns
    -------
    {}

    '''.format(docstrings.REJECT_PARAMETERS_COMMON(indent=4),
               docstrings.REJECT_PARAMETERS_SIGMA(indent=4),
               docstrings.REJECT_RETURNS_SIGMA(indent=4))

# **************************************************************************************** #
# *                              PERCENTILE CLIPPING (PCLIP)                             * #
# **************************************************************************************** #


# **************************************************************************************** #
# *                          CCD NOISE MODEL CLIPPING (CCDCLIP)                          * #
# **************************************************************************************** #
def ccdclip_mask(
        arr,
        mask=None,
        sigma=3.,
        sigma_lower=None,
        sigma_upper=None,
        maxiters=5,
        ddof=0,
        nkeep=3,
        maxrej=None,
        cenfunc='median',
        irafmode=False,
        axis=0,
        gain=1.,
        rdnoise=0.,
        snoise=0.,
        scale_ref=1,
        zero_ref=0,
        dtype='float32',
        full=True
):
    if axis != 0:
        raise ValueError("Currently only axis=0 is supported")

    mask = _set_mask(arr, mask)
    ncombine = arr.shape[0]
    sigma_lower, sigma_upper = _set_sigma(sigma, sigma_lower, sigma_upper)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis)
    cenfunc = _set_cenfunc(cenfunc)
    maxiters = int(maxiters)
    ddof = int(ddof)
    _, gns = _set_gain_rdns(gain, ncombine, dtype=dtype)
    _, rds = _set_gain_rdns(rdnoise, ncombine, dtype=dtype)
    _, sns = _set_gain_rdns(snoise, ncombine, dtype=dtype)

    # Convert to gain-corrected
    arr = do_zs(arr, zeros=None, scales=1/gns)

    o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
        arr=arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        scale_ref=scale_ref,
        zero_ref=zero_ref,
        ddof=ddof,
        nkeep=nkeep,
        maxrej=maxrej,
        cenfunc=cenfunc,
        ccdclip=True,
        rdnoise_ref=np.mean(rds),  # Use mean as the representative value
        snoise_ref=np.mean(sns),   # Use mean as the representative value
        irafmode=irafmode
    )

    # Revert to ADU (DN)
    arr = do_zs(arr, zeros=None, scales=gns)

    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask


ccdclip_mask.__doc__ = ''' Finds masks of `arr` by CCD noise model.

    Parameters
    ----------
    arr : ndarray
        The array to be subjected for masking. `arr` and `mask` must have the
        identical shape. It must be in DN, i.e., **not** gain corrected.

    {}

    Returns
    -------
    {}

    '''.format(docstrings.REJECT_PARAMETERS_COMMON(indent=4),
               docstrings.REJECT_PARAMETERS_SIGMA(indent=4),
               docstrings.REJECT_RETURNS_SIGMA(indent=4))