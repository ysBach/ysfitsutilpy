from astropy.visualization import ZScaleInterval, ImageNormalize

__all__ = ["znorm", "zimshow"]


def znorm(image):
    return ImageNormalize(image, interval=ZScaleInterval())


def zimshow(ax, image, **kwargs):
    return ax.imshow(image, norm=znorm(image), origin='lower', **kwargs)

