# from pathlib import Path

# from astropy import units as u
# from astropy.io import fits
# from astropy.nddata import CCDData

# from .hduutil import _parse_extension, load_ccd, set_ccd_gain_rdnoise

# try:
#     import fitsio
#     HAS_FITSIO = True
# except ImportError:
#     HAS_FITSIO = False

# # class AstroImageMixin:
# #     @classmethod
# #     def load_header():




# class AstroImage:
#     def __init__(self, data=None, header=None, path=None, extension=None,
#                  keys_attr={"gain": ("GAIN", 1), "rdnoise": ("RDNOISE", 0), "exptime": ("EXPTIME", 1)},
#                  verbose=True, update_header=True):
#         self.path = path
#         self.extension = extension
#         self.data = data
#         self.header = header
#         if (self.header is not None) and keys_attr:
#             for attr, (key, default) in keys_attr.items():
#                 if key in self.header:
#                     setattr(self, attr, self.header[key])
#                 else:
#                     setattr(self, attr, default)

#     @classmethod
#     def frompath(cls, path, load_header=True, *args, ext=None, extname=None, extver=None, **kwargs):
#         extension = _parse_extension(*args, ext=ext, extname=extname, extver=extver)
#         if load_header:
#             hdu = fits.open(path, **kwargs)[extension]
#             return cls(data=hdu.data, header=hdu.header, path=path, extension=extension)
#         else:
#             if HAS_FITSIO:
#                 data = fitsio.read(path)
#             else:
#                 data = fits.getdata(path)

#     # def __init__(self, fpath, load_header=True,
#     #              keys_attr={"gain": ("GAIN", 1), "rdnoise": ("RDNOISE", 0), "exptime": ("EXPTIME", 1)},
#     #              verbose=True, update_header=True):
#     #     self.fpath = Path(fpath)

#     #     if load_header:
#     #         self.hdu =

#     #     self.bias_cor = False
#     #     self.dark_cor = False
#     #     self.ovsc_cor = False
#     #     self.flat_cor = False
#     #     self.crrej_cor = False

#     # def info(self):
#     #     ''' Prints information (fits.fitsinfo())
#     #     '''
#     #     pass
