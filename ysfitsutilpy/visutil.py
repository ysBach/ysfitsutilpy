import numpy as np
from astropy.visualization import (ImageNormalize, LinearStretch,
                                   ZScaleInterval, simple_norm)
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

__all__ = ["znorm", "zimshow", "norm_imshow", "colorbaring", "linearticker"]


def znorm(image, stretch=LinearStretch(), **kwargs):
    return ImageNormalize(image,
                          interval=ZScaleInterval(**kwargs),
                          stretch=stretch)


def zimshow(ax, image, stretch=LinearStretch(), cmap=None, **kwargs):
    im = ax.imshow(image,
                   norm=znorm(image, stretch=stretch, **kwargs),
                   origin='lower',
                   cmap=cmap)
    return im


def norm_imshow(ax, data, stretch='linear', power=1.0, asinh_a=0.1,
                min_cut=None, max_cut=None, min_percent=None, max_percent=None,
                percent=None, clip=True, log_a=1000, **kwargs):
    """ Do normalization and do imshow
    """
    norm = simple_norm(data, stretch=stretch, power=power, asinh_a=asinh_a,
                       min_cut=min_cut, max_cut=max_cut,
                       min_percent=min_percent, max_percent=max_percent,
                       percent=percent, clip=clip, log_a=log_a)
    im = ax.imshow(data, norm=norm, origin='lower', **kwargs)
    return im


def colorbaring(fig, ax, im, fmt="%.0f", orientation='horizontal',
                formatter=FormatStrFormatter, **kwargs):
    cb = fig.colorbar(im, ax=ax, orientation=orientation,
                      format=FormatStrFormatter(fmt), **kwargs)

    return cb


def linearticker(ax_list, xmajlocs, xminlocs, ymajlocs, yminlocs,
                 xmajfmts, ymajfmts,
                 xmajlocators=MultipleLocator,
                 xminlocators=MultipleLocator,
                 ymajlocators=MultipleLocator,
                 yminlocators=MultipleLocator,
                 xmajformatters=FormatStrFormatter,
                 ymajformatters=FormatStrFormatter,
                 majgridkw=dict(ls='-', alpha=0.5),
                 mingridkw=dict(ls=':', alpha=0.5)):
    _ax_list = np.atleast_1d(ax_list)
    n_axis = len(_ax_list)
    xmajloc = np.atleast_1d(xmajlocs)
    xminloc = np.atleast_1d(xminlocs)
    ymajloc = np.atleast_1d(ymajlocs)
    yminloc = np.atleast_1d(yminlocs)
    xmajfmt = np.atleast_1d(xmajfmts)
    ymajfmt = np.atleast_1d(ymajfmts)
    if xmajloc.shape[0] != n_axis:
        xmajloc = np.repeat(xmajloc, n_axis)
    if xminloc.shape[0] != n_axis:
        xminloc = np.repeat(xminloc, n_axis)
    if ymajloc.shape[0] != n_axis:
        ymajloc = np.repeat(ymajloc, n_axis)
    if yminloc.shape[0] != n_axis:
        yminloc = np.repeat(yminloc, n_axis)
    if xmajfmt.shape[0] != n_axis:
        xmajfmt = np.repeat(xmajfmt, n_axis)
    if ymajfmt.shape[0] != n_axis:
        ymajfmt = np.repeat(ymajfmt, n_axis)

    if not isinstance(xmajlocators, (tuple, list, np.ndarray)):
        xmajlocators = [xmajlocators] * n_axis
    if not isinstance(xminlocators, (tuple, list, np.ndarray)):
        xminlocators = [xminlocators] * n_axis
    if not isinstance(ymajlocators, (tuple, list, np.ndarray)):
        ymajlocators = [ymajlocators] * n_axis
    if not isinstance(yminlocators, (tuple, list, np.ndarray)):
        yminlocators = [yminlocators] * n_axis
    if not isinstance(xmajformatters, (tuple, list, np.ndarray)):
        xmajformatters = [xmajformatters] * n_axis
    if not isinstance(ymajformatters, (tuple, list, np.ndarray)):
        ymajformatters = [ymajformatters] * n_axis

    for i, aa in enumerate(_ax_list):
        aa.xaxis.set_major_locator(xmajlocators[i](xmajloc[i]))
        aa.yaxis.set_major_locator(ymajlocators[i](ymajloc[i]))
        aa.xaxis.set_minor_locator(xminlocators[i](xminloc[i]))
        aa.yaxis.set_minor_locator(yminlocators[i](yminloc[i]))
        aa.xaxis.set_major_formatter(xmajformatters[i](xmajfmt[i]))
        aa.yaxis.set_major_formatter(ymajformatters[i](ymajfmt[i]))
        aa.grid(which='major', **majgridkw)
        aa.grid(which='minor', **mingridkw)
