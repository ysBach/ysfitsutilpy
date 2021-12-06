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
* Simple analysis of FITS files by making summary csv file, getting statistics (``misc.give_stats``), etc.
* Educational purpose
* ...

It is **not** designed for very general use, e.g., MEF (multi-extension FITS) and radio data, for instance. MEF is somewhat treatable in current version, but not satisfactorily yet.

You may import using ``import ysfitsutilpy as yfu``.

Although the package is subdivided into ``ccdutil``, ``filemgmt``, etc, all the modules' functions are imported to the core package by ``__init__.py``. So you never have to care about the submodules of this package, but just use ``yfu``.

An example usage to make a summary file of FITS files:
```python
import ysfitsutilpy as yfu

# The keywords you want to extract (from the headers of FITS files)
keys = ["DATE-OBS", "FILTER", "OBJECT"]  # actually it is case-insensitive

summary = yfu.make_summary(
    "observation_2018-01-01/R*.fits",
    keywords=keys,
    fname_option='name',  # 'file' column will contain only the name of the file (not full path)
    sort_by="DATE-OBS",  # 'file's will be sorted based on "DATE-OBS" value in the header
    output="summary_2018-01-01.csv",
)

summary
# shows results of the summary CSV file.

```

A simple example to combine multiple images:
```python
import ysfitsutilpy as yfu

comb = yfu.imcombine(
    "observation_2018-01-01/R*M101*.fits",
    combine="med",  # med, median | avg, mean, average | sum
    reject="sc",  # sc, sigc, sigclip, ... | ccd, ccdc, ccdclip
    sigma=[2, 2],  # default is [3., 3.]
    offset="wcs",  # combine by integer shift based on WCS information in headers
    output="combined.fits",
    output_err="comb_err.fits",  # errormap of survived pixels
    output_mask="comb_mask.fits",  # N+1-dimensional mask of the rejected pixel positions
    output_nrej="comb_nrej.fits",  # number of pixels rejected in the output file.
    output_low="comb_low.fits",  # the lower limit used in pixel value rejection
    output_upp="comb_upp.fits",  # the upper limit used in pixel value rejection
    output_rejcode="comb_rejcode.fits",  # represents what rejection has happened (see docstring)
    full=True,
    verbose=True
)
```

A quick dark combine:
```python
import ysfitsutilpy as yfu

# Say dark frames have header OBJECT = "calib" && "IMAGE-TYP" = "DARK"
comb = yfu.group_combine(
    "observation_2018-01-01/*DARK*.fits",
    type_key=["OBJECT", "IMAGE-TYP"],
    type_val=["calib", "DARK"],
    group_key=["FILTER", "EXPTIME"],
    fmt="dark_{:s}_{:.1f}sec.fits",  # output file name format
    outdir="cal-dark"
)
```