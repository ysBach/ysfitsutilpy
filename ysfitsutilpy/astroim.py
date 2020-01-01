from astropy.nddata import CCDData
from pathlib import Path
from .ccdutil import load_ccd, set_ccd_gain_rdnoise
from astropy import units as u


class AstroImage:
    def __init__(self, fpath, unit='adu',
                 gain=None, gain_key="GAIN", gain_unit=u.electron/u.adu,
                 rdnoise=None, rdnoise_key="RDNOISE", rdnoise_unit=u.electron,
                 verbose=True, update_header=True):
        self.fpath = Path(fpath)
        self.ccd = load_ccd(self.fpath, extension=0,
                            usewcs=True, hdu_uncertainty="UNCERT",
                            unit=unit, prefer_bunit=True)

        set_ccd_gain_rdnoise(self.ccd,
                             gain=gain,
                             gain_key=gain_key,
                             gain_unit=gain_unit,
                             rdnoise=rdnoise,
                             rdnoise_key=rdnoise_key,
                             rdnoise_unit=rdnoise_unit,
                             verbose=verbose,
                             update_header=True
                             )
        self.bias_cor = False
        self.dark_cor = False
        self.ovsc_cor = False
        self.flat_cor = False
        self.crrej_cor = False


    def
