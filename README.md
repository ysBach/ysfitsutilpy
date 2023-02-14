# ysfitsutilpy
Convenience utilities made by ysBach especially for dealing FITS files in astronomical sciences.

Please always refer to [GitHub repo](https://github.com/ysBach/ysfitsutilpy) for the most recent updates.

**Why "ys"fitsutilpy? The name "fitsutilpy" is too general, and I believe a better package should take that name, so I decided not to occupy the name. I see many useless packages that preoccupy meaningful names...**

Install by

```
$ pip install ysfitsutilpy
```

or

```
$ cd <where you want to download this package>
$ git clone https://github.com/ysBach/ysfitsutilpy
$ cd ysfitsutilpy
$ git pull && pip install -e .
```
From the second time, **just run the last line**.


This package is made to be used for
* Preprocessing (= bias, dark, and flat) of imaging data (not tested for spectroscopy yet)
* Simple analysis of FITS files by making summary csv file, getting statistics (``misc.give_stats``), etc.
* Educational purpose
* ...

Although I tried to make some functions as general as possible, this package as a whole is **not** designed for very general use, e.g., MEF (multi-extension FITS) and radio data, for instance. MEF is somewhat treatable in current version, but not satisfactorily yet.

You may import using ``import ysfitsutilpy as yfu``.

An example usage to make a summary file of FITS files:
```python
import ysfitsutilpy as yfu

summary = yfu.make_summary(
    "observation_2018-01-01/R*.fits",
    keywords=["DATE-OBS", "FILTER", "OBJECT"],  # header keywords; actually it is case-insensitive
    fname_option='name',  # 'file' column will contain only the name of the file (not full path)
    sort_by="DATE-OBS",  # 'file' column will be sorted based on "DATE-OBS" value in the header
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
    outdir="cal-dark"  # output directory (will automatically be made if not exist)
)
```