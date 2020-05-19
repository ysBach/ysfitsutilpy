# ysfitsutilpy
Convenience utilities made by ysBach especially for dealing FITS files in astronomical sciences.

Install by

```
$ cd <where you want to download this package>
$ git clone https://github.com/ysBach/ysfitsutilpy
$ cd ysfitsutilpy
$ python setup.py install
```



This package is made to be used for
* Preprocessing (= bias, dark, and flat) of imaging data (not tested for spectroscopy yet)
* Simple analysis of FITS files by making summary csv file, getting statistics (``misc.give_stats``), zscale imshow (``visutil.zimshow``), etc.
* Educational purpose
* ...

It is **not** designed for very general use, e.g., MEF (multi-extension FITS) and radio data, for instance. MEF is somewhat treatable in current version, but not satisfactorily yet.

You may import using ``import ysfitsutilpy as yfu``.

Although the package is subdivided into ``ccdutil``, ``filemgmt``, etc, all the modules' functions are imported to the core package by ``__init__.py``. So you never have to care about the submodules of this package, but just use ``yfu``.

An example usage to make a summary file of FITS files:
```python
import ysfitsutilpy as yfu

from pathlib import Path

# The keywords you want to extract (from the headers of FITS files)
keys = ["OBS-TIME", "FILTER", "OBJECT"]  # actually it is case-insensitive

# The working path (directory) and save path
TOPPATH = Path("./observation_2018-01-01")
savepath = TOPPATH/"summary_20180101.csv"

# list of all the fits files in Path object
allfits = list((TOPPATH/"rawdata").glob("*.fits"))

summary = yfu.make_summary(
    allfits, 
    keywords=keys, 
    fname_option='name',                       
    sort_by="DATE-OBS", 
    output=savepath,
    pandas=True  # default: False, so that astropy table is returned.
)
summary
```