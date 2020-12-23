import functools
import warnings

import numpy as np

'''
a = np.array(
    [[[ 0.,  1.],
    [ 2., np.nan],
    [np.nan, np.nan]],

   [[ 6.,  7.],
    [ 8., np.nan],
    [np.nan, np.nan]],

   [[12., 13.],
    [14., 15.],
    [np.nan, np.nan]],

   [[18., np.nan],
    [20., 21.],
    [22., np.nan]],

   [[24., np.nan],
    [np.nan, 27.],
    [28., np.nan]]]
)

nanmedian(a, axis=0, choose='mean')
nanmedian(a, axis=0, choose='lower')
nanmedian(a, axis=0, choose='upper')

numpy(mean):   lower:         upper:
  [[12, 7 ],    [[12, 7 ],      [[12, 7 ],
   [11, 21],     [8 , 21],       [14, 21],
   [25, nan]]    [22, nan]]      [28, nan]]

'''

__all__ = [
    'lmedian', 'nanlmedian',
    'median', 'nanmedian', 'ma_median'
]

array_function_dispatch = functools.partial(
    np.core.overrides.array_function_dispatch, module='numpy')


def lmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    return median(a, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims, choose='lower')


def nanlmedian(a, axis=None, out=None, overwrite_input=False, keepdims=np._NoValue):
    return nanmedian(a, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims,
                     choose='lower')


def _median_dispatcher(a, axis=None, out=None, overwrite_input=None, keepdims=None, choose=None):
    return (a, out)


@array_function_dispatch(_median_dispatcher)
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False, choose='mean'):
    r, k = np.lib.function_base._ureduce(
        a, func=_median, axis=axis, out=out, overwrite_input=overwrite_input, choose=choose
    )
    if keepdims:
        return r.reshape(k)
    else:
        return r


def _median(a, axis=None, out=None, overwrite_input=False, choose='mean'):
    # can't be reasonably be implemented in terms of percentile as we have to
    # call mean to not break astropy
    a = np.asanyarray(a)

    # Set the partition indexes
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        if choose == 'mean':
            kth = [szh - 1, szh]
            kth = [sz//2 - 1]
        elif choose == 'lower':
            kth = [szh - 1]
        elif choose == 'upper':
            kth = [szh + 1]
        else:
            raise ValueError("choose not understood")
    else:
        kth = [(sz - 1) // 2]
    # Check if the array contains any nan's
    if np.issubdtype(a.dtype, np.inexact):
        kth.append(-1)

    if overwrite_input:
        if axis is None:
            part = a.ravel()
            part.partition(kth)
        else:
            a.partition(kth, axis=axis)
            part = a
    else:
        part = np.partition(a, kth, axis=axis)

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)
    indexer = tuple(indexer)
    print(part[indexer])
    # Check if the array contains any nan's
    if np.issubdtype(a.dtype, np.inexact) and sz > 0:
        # warn and return nans like mean would
        rout = np.mean(part[indexer], axis=axis, out=out)
        return np.lib.utils._median_nancheck(part, rout, axis, out)
    else:
        # if there are no nans
        # Use mean in odd and even case to coerce data type
        # and check, use out array.
        return np.mean(part[indexer], axis=axis, out=out)


def ma_median(a, axis=None, out=None, overwrite_input=False, keepdims=False, choose='mean'):
    if not hasattr(a, 'mask'):
        m = median(np.ma.getdata(a, subok=True), axis=axis, out=out, overwrite_input=overwrite_input,
                   keepdims=keepdims, choose=choose)
        if isinstance(m, np.ndarray) and 1 <= m.ndim:
            return np.ma.masked_array(m, copy=False)
        else:
            return m

    r, k = np.lib.function_base._ureduce(a, func=_ma_median, axis=axis, out=out,
                                         overwrite_input=overwrite_input, choose=choose)
    if keepdims:
        return r.reshape(k)
    else:
        return r


def _ma_median(a, axis=None, out=None, overwrite_input=False, choose='mean'):
    # when an unmasked NaN is present return it, so we need to sort the NaN
    # values behind the mask
    if np.issubdtype(a.dtype, np.inexact):
        fill_value = np.inf
    else:
        fill_value = None
    if overwrite_input:
        if axis is None:
            asorted = a.ravel()
            asorted.sort(fill_value=fill_value)
        else:
            a.sort(axis=axis, fill_value=fill_value)
            asorted = a
    else:
        asorted = np.ma.sort(a, axis=axis, fill_value=fill_value)

    if axis is None:
        axis = 0
    else:
        axis = np.core.numeric.normalize_axis_index(axis, asorted.ndim)

    if asorted.shape[axis] == 0:
        # for empty axis integer indices fail so use slicing to get same result
        # as median (which is mean of empty slice = nan)
        indexer = [slice(None)] * asorted.ndim
        indexer[axis] = slice(0, 0)
        indexer = tuple(indexer)
        return np.ma.mean(asorted[indexer], axis=axis, out=out)

    if asorted.ndim == 1:
        counts = np.ma.count(asorted)
        idx, odd = divmod(np.ma.count(asorted), 2)
        mid = asorted[idx + odd - 1:idx + 1]
        if choose == 'lower':
            mid = mid[:1]
        elif choose == 'upper':
            mid = mid[-1:]
        if np.issubdtype(asorted.dtype, np.inexact) and asorted.size > 0:
            # avoid inf / x = masked
            s = mid.sum(out=out)
            if not odd:
                s = np.true_divide(s, 2., casting='safe', out=out)
            s = np.lib.utils._median_nancheck(asorted, s, axis, out)
        else:
            s = mid.mean(out=out)

        # if result is masked either the input contained enough
        # minimum_fill_value so that it would be the median or all values
        # masked
        if np.ma.is_masked(s) and not np.all(asorted.mask):
            return np.ma.minimum_fill_value(asorted)
        return s

    counts = np.ma.count(asorted, axis=axis, keepdims=True)
    h = counts // 2

    # duplicate high if odd number of elements so mean does nothing
    odd = counts % 2 == 1

    if choose == 'mean':
        l = np.where(odd, h, h-1)
    if choose == 'lower':
        l = np.where(odd, h, h-1)
        h = l
    if choose == 'upper':
        l = h

    lh = np.concatenate([l, h], axis=axis)

    # get low and high median
    low_high = np.take_along_axis(asorted, lh, axis=axis)

    def replace_masked(s):
        # Replace masked entries with minimum_full_value unless it all values
        # are masked. This is required as the sort order of values equal or
        # larger than the fill value is undefined and a valid value placed
        # elsewhere, e.g. [4, --, inf].
        if np.ma.is_masked(s):
            rep = (~np.all(asorted.mask, axis=axis, keepdims=True)) & s.mask
            s.data[rep] = np.ma.minimum_fill_value(asorted)
            s.mask[rep] = False

    replace_masked(low_high)

    if np.issubdtype(asorted.dtype, np.inexact):
        # avoid inf / x = masked
        s = np.ma.sum(low_high, axis=axis, out=out)
        np.true_divide(s.data, 2., casting='unsafe', out=s.data)

        s = np.lib.utils._median_nancheck(asorted, s, axis, out)
    else:
        s = np.ma.mean(low_high, axis=axis, out=out)

    return s


def _nanmedian1d(arr1d, overwrite_input=False, choose='mean'):
    arr1d, overwrite_input = np.lib._remove_nan_1d(arr1d, overwrite_input=overwrite_input)
    if arr1d.size == 0:
        return np.nan

    return median(arr1d, overwrite_input=overwrite_input, choose=choose)


def _nanmedian(a, axis=None, out=None, overwrite_input=False, choose='mean'):
    if axis is None or a.ndim == 1:
        part = a.ravel()
        if out is None:
            return _nanmedian1d(part, overwrite_input, choose=choose)
        else:
            out[...] = _nanmedian1d(part, overwrite_input, choose=choose)
            return out
    else:
        # for small medians use sort + indexing which is still faster than
        # apply_along_axis
        # benchmarked with shuffled (50, 50, x) containing a few NaN
        if a.shape[axis] < 600:
            return _nanmedian_small(a, axis, out, overwrite_input, choose)
        result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input, choose=choose)
        if out is not None:
            out[...] = result
        return result


def _nanmedian_small(a, axis=None, out=None, overwrite_input=False, choose='mean'):
    a = np.ma.masked_array(a, np.isnan(a))
    m = ma_median(a, axis=axis, overwrite_input=overwrite_input, choose=choose)
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=4)
    if out is not None:
        out[...] = m.filled(np.nan)
        return out
    return m.filled(np.nan)


def _nanmedian_dispatcher(a, axis=None, out=None, overwrite_input=None, keepdims=None, choose=None):
    return (a, out)


@array_function_dispatch(_nanmedian_dispatcher)
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=np._NoValue, choose='mean'):
    a = np.asanyarray(a)
    # apply_along_axis in _nanmedian doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)

    r, k = np.lib.function_base._ureduce(a, func=_nanmedian, axis=axis, out=out,
                                         overwrite_input=overwrite_input, choose=choose)
    if keepdims and keepdims is not np._NoValue:
        return r.reshape(k)
    else:
        return r
