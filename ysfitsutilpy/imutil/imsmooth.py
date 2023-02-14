from scipy.ndimage import median_filter


def smooth_med(
        ccd,
        cadd=1.0e-10,
        size=5,
        footprint=None,
        mode="reflect",
        cval=0.0,
        origin=0
):
    """ Smooth image by `~scipy.ndimage.median_filter`.
    ccd : `~astropy.nddata.CCDData`
        The CCD to find the bad pixels.

    cadd : float, ndarray, optional.
        A very small const to be added to the input array to avoid 0-valued
        pixel after median filtering. This is to avoid the problem when doing
        ``image/|median_filtered|``.

    size, footprint, mode, cval, origin : optional.
        The parameters to obtain the median-filtered map. See
        `~scipy.ndimage.median_filter`.
    """
    return median_filter(ccd.data.copy() + cadd, size=size, footprint=footprint,
                         mode=mode, cval=cval, origin=origin)



