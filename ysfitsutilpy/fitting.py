import numpy as np
from astropy.stats import sigma_clip
from astropy.modeling.fitting import (JointFitter, LevMarLSQFitter,
                                      LinearLSQFitter, SimplexLSQFitter,
                                      SLSQPLSQFitter, FittingWithOutlierRemoval)
# from astropy.modeling.functional_models import (AiryDisk2D, ArcCosine1D,
#                                                 ArcSine1D, ArcTangent1D, Box1D,
#                                                 Box2D, Const1D, Const2D,
#                                                 Cosine1D, Disk2D, Ellipse2D,
#                                                 Exponential1D, Gaussian1D,
#                                                 Gaussian2D,
#                                                 KingProjectedAnalytic1D,
#                                                 Linear1D, Logarithmic1D,
#                                                 Lorentz1D, Moffat1D, Moffat2D,
#                                                 Multiply, Planar2D,
#                                                 RedshiftScaleFactor,
#                                                 RickerWavelet1D,
#                                                 RickerWavelet2D, Ring2D, Scale,
#                                                 Sersic1D, Sersic2D, Shift,
#                                                 Sine1D, Tangent1D, Trapezoid1D,
#                                                 TrapezoidDisk2D, Voigt1D)
# from astropy.modeling.polynomial import (SIP, Chebyshev1D, Chebyshev2D,
#                                          Hermite1D, Hermite2D, InverseSIP,
#                                          Legendre1D, Legendre2D,
#                                          OrthoPolynomialBase, Polynomial1D,
#                                          Polynomial2D, PolynomialModel)
# from astropy.modeling.powerlaws import (BrokenPowerLaw1D,
#                                         ExponentialCutoffPowerLaw1D,
#                                         LogParabola1D, PowerLaw1D,
#                                         SmoothlyBrokenPowerLaw1D)
import bottleneck as bn
import astropy.modeling as am
# At this moment, astropy modeling is used.
# If possible, better to detache astropy-dependency in the nearest future.


__all__ = ["gridding", "get_fitter", "get_model", "fit_model", "fit_model_iter"]

ALL_MODELS = []
for module in (am.functional_models, am.physical_models, am.polynomial, am.powerlaws,):
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, am.FittableModel):
            ALL_MODELS.append(obj)

ALL_MODELS = tuple(ALL_MODELS)
ALL_MODELS = {mod.__name__.lower(): mod for mod in ALL_MODELS}


def gridding(data, mask=None, steps=None, copy=True, force_flat=False):
    """
    gridding(np.arange(3) + 10) == (array([[0, 1, 2]]), array([10, 11, 12]))
    gridding(np.eye(3)) == (
        array([[[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]],

       [[0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]]]),
        array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
    )

    """
    steps = (np.atleast_1d(steps)).ravel()
    if steps.size == 1:
        steps = np.repeat(steps, data.ndim)
    elif steps.size != data.ndim:
        raise ValueError(f"steps must have length equal to data.ndim ({data.ndim})")

    sls = [slice(None, data.shape[i], steps[i]) for i in range(data.ndim)]
    grids = np.mgrid[tuple(sls)]
    # ^ zyx order

    if mask is not None:
        _m = ~mask.astype(bool)
        data_1d = data[_m].copy() if copy else data[_m]
        grid_1d = [g[_m] for g in grids]  # No need to safeguard `grids`
        return np.array(grid_1d), data_1d
    else:
        if force_flat:
            data_1d = data.flatten() if copy else data.ravel()
            grid_1d = [g.ravel() for g in grids]  # No need to safeguard `grids`
            return np.array(grid_1d), data_1d
        else:
            return grids, data


def get_fitter(fitter_name="LM", /, **kwargs):
    """ Returns astropy fitter.

    Parameters
    ----------
    name : str, optional.
        The name of the fitter. Only the first 2-3 letters are used for
        determining the fitter (case-insensitive)::

          * starts with "lev" or "lm": Levenberg-Marquardt (`LevMarLSQFitter`)
          * starts with "lin": Linear (`LinearLSQFitter`)
          * starts with "sim": Simplex (`SimplexLSQFitter`)
          * starts with "sl": SLSQP (`SLSQPLSQFitter`)
          * starts with "jo": Joint (`JointFitter`)

    **kwargs :
        Keyword arguments for the fitter (name and astropy default values)::

          * `LinearLSQFitter`: `calc_uncertainties=False`
          * `LevMarLSQFitter`: `calc_uncertainties=False`
          * `SimplexLSQFitter`, `SLSQPLSQFitter`: N/A
          * `JointFitter`: `models`, jointparameters`, `initvals` (must be given)
    """
    if fitter_name.lower().startswith("lev") or fitter_name.lower().startswith("lm"):
        return LevMarLSQFitter(**kwargs)
    elif fitter_name.lower().startswith("lin"):
        return LinearLSQFitter(**kwargs)
    elif fitter_name.lower().startswith("sl"):
        return SLSQPLSQFitter(**kwargs)
    elif fitter_name.lower().startswith("sim"):
        return SimplexLSQFitter(**kwargs)
    elif fitter_name.lower().startswith("jo"):
        return JointFitter(**kwargs)
    else:
        return fitter_name(**kwargs)
        # ^ assume the `fitter_name` is already an astropy fitter


def get_model(model_name, *args, **kwargs):
    """ Finds and returns the model with the given name.
    For instance, it's customary to put degrees as args/kwargs for polynomial models:
    `get_model("chebyshev2d", 2, 2)` or `get_model("chebyshev2d", x_degree=2,
    y_degree=2)` return `Chebyshev2D(x_degree=2, y_degree=2)`.
    For other cases, it depends: `get_model("gaussian1d")` returns
    `Gaussian1D`, but sometimes you may want to initialize it with initial
    values: `get_model("gaussian1d", amplitude=1, mean=0, stddev=1)`.
    """
    # Try to find the model even if there are typos
    _name = model_name.lower()
    for ch in [" "] + list("!@#$%^&*()_+-=[]{}|;':,./<>?`~\""):
        _name = _name.replace(ch, "")
    try:
        return ALL_MODELS[_name](*args, **kwargs) if kwargs or args else ALL_MODELS[_name]
    except KeyError:
        raise KeyError(f"Unknown model name `{model_name}`. Available: {ALL_MODELS.keys()}")


def fit_model(
        model_name,
        data,
        mask=None,
        steps=None,
        fitter_name="LM",
        fitter_kw={},
        full=False,
        **model_kw
):
    """ Fit a model to a data.

    Parameters
    ----------
    model_name : str
        The model to fit. See `get_model`.
    ndim : int, optional.
        The dimension of the data. If not given, it is inferred from the model.

    **kwargs :
        Keyword arguments for the fitter (name and astropy default values)::

          * `LinearLSQFitter`: `calc_uncertainties=False`
          * `LevMarLSQFitter`: `calc_uncertainties=False`
          * `SimplexLSQFitter`, `SLSQPLSQFitter`: N/A
          * `JointFitter`: `models`, jointparameters`, `initvals` (must be given)

    **model_kw :
        The paramters to initialize model. It can be ::
          * degrees for polynomial (e.g., `degree` for `Chebyshev1D`)
          * initial parameters for others (e.g., `amplitude` for `Gaussian1D`)

    Returns
    -------
    model : `astropy.modeling.Model`
        The fitted model.
    """
    fitter = get_fitter(fitter_name, **fitter_kw)
    model_init = get_model(model_name)(**model_kw)
    grid_1d, data_1d = gridding(data, mask=mask, steps=steps, force_flat=True, copy=True)
    model_fit = fitter(model_init, *grid_1d[::-1], data_1d)
    if full:
        return model_fit, fitter, grid_1d, data_1d
    else:
        return model_fit


def fit_model_iter(
        model_name,
        data,
        outlier_func=sigma_clip,
        outlier_kw=dict(sigma=3, maxiters=1, cenfunc="median", stdfunc="std"),
        maxiters=3,
        weights=None,
        mask=None,
        steps=None,
        fitter_name="LM",
        fitter_kw={},
        full=False,
        **model_kw
):
    """ Fit a model to a data with `FittingWithOutlierRemoval`.

    Parameters
    ----------
    model_name : `astropy.modeling.Model`
        The model to fit. See `get_model`.

    outlier_func : callable
        A function for outlier removal.
        If this accepts an ``axis`` parameter like the `numpy` functions, the
        appropriate value will be supplied automatically when fitting model
        sets (unless overridden in ``outlier_kwargs``), to find outliers for
        each model separately; otherwise, the same filtering must be performed
        in a loop over models, which is almost an order of magnitude slower.

    maxiters : int, optional
        Maximum number of iterations.

    **kwargs :
        Keyword arguments for the fitter (name and astropy default values)::

          * `LinearLSQFitter`: `calc_uncertainties=False`
          * `LevMarLSQFitter`: `calc_uncertainties=False`
          * `SimplexLSQFitter`, `SLSQPLSQFitter`: N/A
          * `JointFitter`: `models`, jointparameters`, `initvals` (must be given)

    **model_kw :
        The paramters to initialize model. It can be ::
          * degrees for polynomial (e.g., `degree` for `Chebyshev1D`)
          * initial parameters for others (e.g., `amplitude` for `Gaussian1D`)

    Returns
    -------
    model : `astropy.modeling.Model`
        The fitted model.
    """
    fitter = FittingWithOutlierRemoval(
        get_fitter(fitter_name, **fitter_kw),
        outlier_func=outlier_func,
        niter=maxiters,
        **outlier_kw
    )
    model_init = get_model(model_name)(**model_kw)
    grid_1d, data_1d = gridding(data, mask=mask, steps=steps, force_flat=True, copy=True)
    model_fit, mask = fitter(model_init, *grid_1d[::-1], data_1d, weights=weights)
    if full:
        return model_fit, fitter, grid_1d, data_1d, mask
    else:
        return model_fit
