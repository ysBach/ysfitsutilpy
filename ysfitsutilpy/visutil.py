from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from matplotlib.ticker import FormatStrFormatter

__all__ = ["znorm", "zimshow", "colorbaring"]


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


def colorbaring(fig, ax, im, fmt=None, formatter=FormatStrFormatter):
    cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                      format=FormatStrFormatter(fmt))

    return cb
