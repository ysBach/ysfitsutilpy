from warnings import warn

import numpy as np
from astropy.visualization import (ImageNormalize, LinearStretch,
                                   ZScaleInterval, simple_norm)
from matplotlib.ticker import (FormatStrFormatter, LogFormatterSciNotation,
                               LogLocator, MultipleLocator, NullFormatter)

__all__ = ["znorm", "zimshow", "norm_imshow", "colorbaring",
           "mplticker", "linticker", "logticker", "logxticker", "logyticker",
           "linearticker"]


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
                 majgridkw=dict(ls='-', alpha=0.8),
                 mingridkw=dict(ls=':', alpha=0.5)):
    warn("Use mplticker instead of linearticker.")
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


def mplticker(ax_list,
              xmajlocators=None, xminlocators=None,
              ymajlocators=None, yminlocators=None,
              xmajformatters=None, xminformatters=None,
              ymajformatters=None, yminformatters=None,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws=None,
              ymajfmtkws=None, yminfmtkws=None,
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    ''' Set tickers of Axes objects.
    Note
    ----
    Notation of arguments is <axis><which><name>. <axis> can be ``x`` or
    ``y``, and <which> can be ``major`` or ``minor``.
    For example, ``xmajlocators`` is the Locator object for x-axis
    major.  ``kw`` means the keyword arguments that will be passed to
    locator, formatter, or grid.
    If a single object is given for locators, formatters, grid, or kw
    arguments, it will basically be copied by the number of Axes objects
    and applied identically through all the Axes.

    Parameters
    ----------
    ax_list : Axes or 1-d array-like of such
        The Axes object(s).

    locators : Locator, None, list of such, optional
        The Locators used for the ticker. Must be a single Locator
        object or a list of such with the identical length of
        ``ax_list``.
        If ``None``, the default locator is not touched.

    formatters : Formatter, None, False, array-like of such, optional
        The Formatters used for the ticker. Must be a single Formatter
        object or an array-like of such with the identical length of
        ``ax_list``.
        If ``None``, the default formatter is not touched.
        If ``False``, the labels are not shown (by using the trick
        ``FormatStrFormatter(fmt="")``).

    grids : bool, array-like of such, optinoal.
        Whether to draw the grid lines. Must be a single bool object or
        an array-like of such with the identical length of ``ax_list``.

    lockws : dict, array-like of such, array-like, optional
        The keyword arguments that will be passed to the ``locators``.
        If it's an array-like but elements are not dict, it is
        interpreted as ``*args`` passed to locators.
        If it is (or contains) dict, it must be a single dict object or
        an array-like object of such with the identical length of
        ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    fmtkws : dict, str, list of such, optional
        The keyword arguments that will be passed to the ``formatters``.
        If it's an array-like but elements are not dict, it is
        interpreted as ``*args`` passed to formatters.
        If it is (or contains) dict, it must be a single dict object or
        an array-like object of such with the identical length of
        ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    gridkw : dict, list of such, optional
        The keyword arguments that will be passed to the grid. Must be a
        single dict object or a list of such with the identical length
        of ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    '''
    def _check(obj, name, n):
        arr = np.atleast_1d(obj)
        n_arr = arr.shape[0]
        if n_arr not in [1, n]:
            raise ValueError(f"{name} must be a single object or a 1-d array"
                             + f" with the same length as ax_list ({n}).")
        else:
            newarr = arr.tolist() * (n//n_arr)

        return newarr

    def _setter(setter, Setter, kw):
        # don't do anything if obj (Locator or Formatter) is None:
        if (Setter is not None) and (kw is not None):

            # matplotlib is so poor in log plotting....
            if (Setter == LogLocator) and ("numticks" not in kw):
                kw["numticks"] = 50

            if isinstance(kw, dict):
                setter(Setter(**kw))
            else:  # interpret as ``*args``
                setter(Setter(*(np.atleast_1d(kw).tolist())))
            # except:
            #     raise ValueError("Error occured for Setter={} with input {}"
            #                      .format(Setter, kw))

    _ax_list = np.atleast_1d(ax_list)
    if _ax_list.ndim > 1:
        raise ValueError("ax_list must be at most 1-d.")
    n_axis = _ax_list.shape[0]

    _xmajlocators = _check(xmajlocators, "xmajlocators", n_axis)
    _xminlocators = _check(xminlocators, "xminlocators", n_axis)
    _ymajlocators = _check(ymajlocators, "ymajlocators", n_axis)
    _yminlocators = _check(yminlocators, "yminlocators", n_axis)

    _xmajformatters = _check(xmajformatters, "xmajformatters ", n_axis)
    _xminformatters = _check(xminformatters, "xminformatters ", n_axis)
    _ymajformatters = _check(ymajformatters, "ymajformatters", n_axis)
    _yminformatters = _check(yminformatters, "yminformatters", n_axis)

    _xmajlockws = _check(xmajlockws, "xmajlockws", n_axis)
    _xminlockws = _check(xminlockws, "xminlockws", n_axis)
    _ymajlockws = _check(ymajlockws, "ymajlockws", n_axis)
    _yminlockws = _check(yminlockws, "yminlockws", n_axis)

    _xmajfmtkws = _check(xmajfmtkws, "xmajfmtkws", n_axis)
    _xminfmtkws = _check(xminfmtkws, "xminfmtkws", n_axis)
    _ymajfmtkws = _check(ymajfmtkws, "ymajfmtkws", n_axis)
    _yminfmtkws = _check(yminfmtkws, "yminfmtkws", n_axis)

    _xmajgrids = _check(xmajgrids, "xmajgrids", n_axis)
    _xmingrids = _check(xmingrids, "xmingrids", n_axis)
    _ymajgrids = _check(ymajgrids, "ymajgrids", n_axis)
    _ymingrids = _check(ymingrids, "ymingrids", n_axis)

    _xmajgridkws = _check(xmajgridkws, "xmajgridkws", n_axis)
    _xmingridkws = _check(xmingridkws, "xmingridkws", n_axis)
    _ymajgridkws = _check(ymajgridkws, "ymajgridkws", n_axis)
    _ymingridkws = _check(ymingridkws, "ymingridkws", n_axis)

    for i, aa in enumerate(_ax_list):
        _xmajlocator = _xmajlocators[i]
        _xminlocator = _xminlocators[i]
        _ymajlocator = _ymajlocators[i]
        _yminlocator = _yminlocators[i]

        _xmajformatter = _xmajformatters[i]
        _xminformatter = _xminformatters[i]
        _ymajformatter = _ymajformatters[i]
        _yminformatter = _yminformatters[i]

        _xmajgrid = _xmajgrids[i]
        _xmingrid = _xmingrids[i]
        _ymajgrid = _ymajgrids[i]
        _ymingrid = _ymingrids[i]

        _xmajlockw = _xmajlockws[i]
        _xminlockw = _xminlockws[i]
        _ymajlockw = _ymajlockws[i]
        _yminlockw = _yminlockws[i]

        _xmajfmtkw = _xmajfmtkws[i]
        _xminfmtkw = _xminfmtkws[i]
        _ymajfmtkw = _ymajfmtkws[i]
        _yminfmtkw = _yminfmtkws[i]

        _xmajgridkw = _xmajgridkws[i]
        _xmingridkw = _xmingridkws[i]
        _ymajgridkw = _ymajgridkws[i]
        _ymingridkw = _ymingridkws[i]

        _setter(aa.xaxis.set_major_locator, _xmajlocator, _xmajlockw)
        _setter(aa.xaxis.set_minor_locator, _xminlocator, _xminlockw)
        _setter(aa.yaxis.set_major_locator, _ymajlocator, _ymajlockw)
        _setter(aa.yaxis.set_minor_locator, _yminlocator, _yminlockw)
        _setter(aa.xaxis.set_major_formatter, _xmajformatter, _xmajfmtkw)
        _setter(aa.xaxis.set_minor_formatter, _xminformatter, _xminfmtkw)
        _setter(aa.yaxis.set_major_formatter, _ymajformatter, _ymajfmtkw)
        _setter(aa.yaxis.set_minor_formatter, _yminformatter, _yminfmtkw)

        # Strangely, using ``b=_xmingrid`` does not work if it is
        # False... I had to do it manually like this... OMG matplotlib..
        if _xmajgrid:
            aa.grid(axis='x', which='major', **_xmajgridkw)
        if _xmingrid:
            aa.grid(axis='x', which='minor', **_xmingridkw)
        if _ymajgrid:
            aa.grid(axis='y', which='major', **_ymajgridkw)
        if _ymingrid:
            aa.grid(axis='y', which='minor', **_ymingridkw)


def linticker(ax_list,
              xmajlocators=MultipleLocator, xminlocators=MultipleLocator,
              ymajlocators=MultipleLocator, yminlocators=MultipleLocator,
              xmajformatters=FormatStrFormatter,
              xminformatters=NullFormatter,
              ymajformatters=FormatStrFormatter,
              yminformatters=NullFormatter,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws={},
              ymajfmtkws=None, yminfmtkws={},
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logticker(ax_list,
              xmajlocators=LogLocator, xminlocators=LogLocator,
              ymajlocators=LogLocator, yminlocators=LogLocator,
              xmajformatters=LogFormatterSciNotation,
              xminformatters=NullFormatter,
              ymajformatters=LogFormatterSciNotation,
              yminformatters=NullFormatter,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws={},
              ymajfmtkws=None, yminfmtkws={},
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logxticker(ax_list,
               xmajlocators=LogLocator, xminlocators=LogLocator,
               ymajlocators=MultipleLocator, yminlocators=MultipleLocator,
               xmajformatters=LogFormatterSciNotation,
               xminformatters=NullFormatter,
               ymajformatters=FormatStrFormatter,
               yminformatters=NullFormatter,
               xmajgrids=True, xmingrids=True,
               ymajgrids=True, ymingrids=True,
               xmajlockws=None, xminlockws=None,
               ymajlockws=None, yminlockws=None,
               xmajfmtkws=None, xminfmtkws={},
               ymajfmtkws=None, yminfmtkws={},
               xmajgridkws=dict(ls='-', alpha=0.5),
               xmingridkws=dict(ls=':', alpha=0.5),
               ymajgridkws=dict(ls='-', alpha=0.5),
               ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logyticker(ax_list,
               xmajlocators=MultipleLocator, xminlocators=MultipleLocator,
               ymajlocators=LogLocator, yminlocators=LogLocator,
               xmajformatters=FormatStrFormatter,
               xminformatters=NullFormatter,
               ymajformatters=LogFormatterSciNotation,
               yminformatters=NullFormatter,
               xmajgrids=True, xmingrids=True,
               ymajgrids=True, ymingrids=True,
               xmajlockws=None, xminlockws=None,
               ymajlockws=None, yminlockws=None,
               xmajfmtkws=None, xminfmtkws={},
               ymajfmtkws=None, yminfmtkws={},
               xmajgridkws=dict(ls='-', alpha=0.5),
               xmingridkws=dict(ls=':', alpha=0.5),
               ymajgridkws=dict(ls='-', alpha=0.5),
               ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())
