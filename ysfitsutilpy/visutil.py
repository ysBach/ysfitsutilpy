from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval

__all__ = ["znorm", "zimshow"]


def znorm(image, stretch=LinearStretch(), **kwargs):
    return ImageNormalize(image,
                          interval=ZScaleInterval(**kwargs),
                          stretch=stretch)


def zimshow(ax, image, stretch=LinearStretch(), cmap=None, **kwargs):
    return ax.imshow(image,
                     norm=znorm(image, stretch=stretch, **kwargs),
                     origin='lower',
                     cmap=cmap)
