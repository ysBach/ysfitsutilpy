from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time

from ..hduutil import (CCDData_astype, _has_header, _parse_extension,
                       _parse_image, calc_offset_physical, calc_offset_wcs,
                       imslice)
from ..misc import _offsets2slice, cmt2hdr, update_tlm

__all__ = ["imarith"]

# TODO: add sections


def _update_hdr(
        header,
        params,
        name1,
        op,
        name2,
        output,
        error_calc=False,
        t_ref=None,
        verbose=False
):
    if isinstance(params, str):
        header["OBJECT"] = params
    elif isinstance(params, dict):
        for k, v in params.items():
            header[k] = v
    elif params is not None:
        raise TypeError("hdr_params must be None, str, or dict of header keyword-value pairs.")

    infostr = f"IMARITH: ({name1}) {op} ({name2})"
    if output is not None:
        infostr += f" -> {output}"
    if error_calc:
        infostr += " with error propagation"
    cmt2hdr(header, 'h', s=infostr, t_ref=t_ref, verbose=verbose)
    update_tlm(header)


def _ccddata_operator(op):
    if op in ['**', '%', '//']:
        raise ValueError(f"Operator {op} is not supported when error_calc is True.")
    elif op == '+':
        operator = 'add'
    elif op == '-':
        operator = 'subtract'
    elif op == '*':
        operator = 'multiply'
    elif op == '/':
        operator = 'divide'
    else:
        raise ValueError(f"Operator {op} is not supported.")
    return operator


def _replace_nan(res, header, replace=None):
    if replace is not None:
        res.data[~np.isfinite(res.data)] = replace
        cmt2hdr(header, 'h', f"Non-finite pixels replaced by {replace}", time_fmt=None)


def _load_im_name_hdr(
        im1,
        im2,
        name1,
        name2,
        extension1,
        extension2,
        offsets=None,
        force_ccddata=False,
        verbose=False
):
    ''' Prepare images as ndarray unless FORCING to become CCDData.
    It, however, *tries* to find at least one image with header for logging.
    '''
    def __check_if_has_header(im, name, extension):
        ''' Checks if has header && convert to HDUList if path-like.
        '''
        try:  # if path-like, assume it has hdr. It's ~ 100x faster than opening the file by _has_header.
            fpath = Path(im)
            im = fits.open(fpath)[_parse_extension(extension)]  # turn it into HDUList
            name = str(fpath) if name is None else name
            # ^ If not set again, the imname will become User-provided PrimaryHDU.
            im_has_hdr = True
        except TypeError:  # im is not a path-like
            im_has_hdr = _has_header(im, extension)
        except FileNotFoundError:
            raise FileNotFoundError(f"im is path-like but doesn't exist at {im}")
        return im, name, im_has_hdr

    im1, name1, im1_has_hdr = __check_if_has_header(im1, name1, extension1)

    if im1_has_hdr:
        hdr_ref = im1.header
    else:
        im2, name2, im2_has_hdr = __check_if_has_header(im2, name2, extension2)  # load to HDUList if path-like
        if im2_has_hdr:
            hdr_ref = im2.header
        else:
            hdr_ref = None

    if offsets is not None:  # We always have to open headers of both, numeric or ndarray not acceptible.
        im1, im1name, _ = _parse_image(im1, extension1, name1, prefer_ccddata=True)  # Returns CCDData only if
        im2, im2name, _ = _parse_image(im2, extension2, name2, prefer_ccddata=True)  # CCDLIKE or path-like
        try:
            shapes = [im1.data.shape, im2.data.shape]
        except AttributeError:
            raise ValueError("If offsets is used, im must be CCDData-like or path-like (header needed)")

        if offsets.lower() in ['phy', 'phys', 'physical']:
            offsets_name = "Physical"
            offsets = [np.array([0]*im1.data.ndim),
                       calc_offset_physical(im1, im2, ignore_ltm=True, intify_offset=True)]
        elif offsets.lower() in ['wcs', 'world']:
            offsets_name = "World (WCS)"
            offsets = [np.array([0]*im1.data.ndim),
                       calc_offset_wcs(im1, im2, loc_target='center', loc_reference='center',
                                       intify_offset=True)]
        else:
            raise ValueError("offsets not understood.")

        trimsecs = _offsets2slice(shapes, offsets, method='inner', fits_convention=True)
        if verbose:
            print(f"Using offsets {offsets_name}, trimming happened for im1 and im2:")

        if force_ccddata:
            im1 = imslice(im1, trimsecs[0], verbose=verbose)
            im2 = imslice(im2, trimsecs[1], verbose=verbose)
        else:
            # No need to update header
            im1 = imslice(im1, trimsecs[0], verbose=verbose, update_header=False).data
            im2 = imslice(im2, trimsecs[1], verbose=verbose, update_header=False).data

    else:  # Open only 1 header
        im1, im1name, _ = _parse_image(im1, extension1, name1, force_ccddata=force_ccddata)
        im2, im2name, _ = _parse_image(im2, extension2, name2, force_ccddata=force_ccddata)

    return im1, im2, im1name, im2name, hdr_ref


def imarith(
        im1,
        op,
        im2,
        output=None,
        extension1=None,
        extension2=None,
        name1=None,
        name2=None,
        offsets=None,
        replace=0,
        header_params=None,
        dtype='float32',
        error_calc=False,
        ignore_header=False,
        overwrite=False,
        verbose=True
):
    ''' Similar to IRAF IMARITH
    Parameters
    ----------
    im1, im2 : `~astropy.nddata.CCDData`, ndarray, number-like, path-like
        The images to be operated. A string that can be converted to float
        (``float(im)``) will be interpreted as numbers; if not, it will be
        interpreted as a path to the FITS file.

    op : str in ['**', '%', '//', '+', '-', '*', '/']
        The operation to be done. Unlike IRAF, 'min' and 'max' are not
        implemented (easy to implement in the future but I don't know the
        necessity). ``['**', '%', '//']`` are not supported when
        ``error_calc=True``.

    output : path-like, optional.
        The path for the resulting file to be saved.

    extension1, extension2 : int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    name1, name2: str, optional.
        The names of the images that will be logged into the header
        (``HISTORY``). If `None`, function automatically chooses appropriate
        name for it: the number (if `im` is number-like), the path (if `im` is
        path-like), or explanatory strings (if `im` is either ndarray or
        `~astropy.nddata.CCDData`)

    replace : np.nan, float-like, optional.
        The value to replace pixels where the value is NaN or Inf (i.e.,
        `~np.isfinite(ccd.data)`). Use `None` to keep the ``nan`` and ``inf``
        as is. Default is ``0``, following IRAF. Note that both nan and inf are
        **not** representable in integer data types.

    header_params : str or dict, optional.
        If a string, the output file's ``OBJECT`` keyword will be replaced by
        `headdr_params`. If a dict, it must be a dict of header keyword and
        value pairs, so that the output file's header will have those key-value
        pairs (care is needed since it overwrites the pre-existing keys). If
        dict, it can be ``{key:value}`` or ``{key:(value, comment)}``.

        ..note::
            The behavior is different from IRAF. In IRAF, ``hparams`` is used
            to propagate the header keyword. This is used mainly for
            ``"EXPTIME"`` to sum the exposure time if two images are combined.

    dtype : str, dtype, optional.
        The data type of the output CCDData and/or file.

    error_calc : bool, optional.
        If `True`, the uncertainties are propagated by `~astropy.nddata`
        arithmetics. If `False` (default), there is no need to load data as
        `~astropy.nddata.CCDData` (because header parsing time gets enromous if
        iterated through hundreds of files), and hence the cfitsio-like
        `fitsio` is used. Error calculation is done only if
        ``ignore_header=False`` at the moment.

    ignore_header : bool, optional.
        Whether to ignore all the header informations of `im1` and `im2`. This
        will boost the speed of the code because none of the effort will be put
        to find/parse header of any file. A cost is that virtually no
        information (including unit such as ``BUNIT``) is preserved in the
        output. The returned HDU will only have meaningful ``HISTORY``.
        `offsets` will be also be ignored, and it will raise critical error if
        FITS images have different shape. Error propagation is of course
        impossible at this moment, because there is no way to infer the
        extension for the uncertainty and its type (wheter variance or standard
        deviation, etc.)

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        If `ignore_header` is `False` (default).

    hdu : `~astropy.io.fits.PrimaryHDU`
        If `ignore_header` is `True`. This is not in `~astropy.nddata.CCDData`
        format since there is no way to infer the unit (e.g., ``BUNIT``).

    Notes
    -----
    Performance tip: if you iterate over many images but one is fixed (e.g.,
    ``imarith(images[i], '/', mflat_path) for i in range(100)``), **load the
    multiply used file** and give that HDU or `~astropy.nddata.CCDData` to
    `imarith`. This will reduce the time spent for file I/O.

    Converting an array to CCDData takes only ~ 10 us regardless on the array
    size on MBP 15"*; this is because most time is spent on metadata
    generation. Note, however, that *reading* a FITS file takes ~ 10 ms, i.e.,
    1000 times slower.

    Tested on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)] 2020-11-02 21:50:07 (KST:
    GMT+09:00) - ysBach

    The type checking ``isinstance(im, CCDData)`` takes only ~ 0.1 us.
    '''

    _t = Time.now()

    if ignore_header:  # Only work with ndarray
        if error_calc and verbose:
            print("Error propagation is not supported (yet...?) when ignore_header is True.")
        # Never use CCDData-like format for input data ndarray
        im1, name1, _ = _parse_image(im1, extension1, name1)  # ndarray
        im2, name2, _ = _parse_image(im2, extension2, name2)  # ndarray

        # ``eval`` takes nearly identical time to many if-blocks of ``if op == '+'``, etc.
        hdu = fits.PrimaryHDU(data=(eval(f'im1 {op} im2')).astype(dtype))
        _replace_nan(hdu, hdu.header, replace=replace)
        _update_hdr(hdu.header, header_params, name1, op, name2, output,
                    error_calc=error_calc, t_ref=_t, verbose=verbose)
        if output is not None:
            hdu.writeto(Path(output), overwrite=overwrite)
        return hdu

    else:
        if error_calc:  # Only work with CCDData
            # Errors will be automatically calculated if NDData's arithmetic methods, so force to CCDData:
            im1, im2, name1, name2, hdr_ref = _load_im_name_hdr(
                im1, im2, name1, name2, extension1, extension2,
                force_ccddata=True, offsets=offsets, verbose=verbose
            )
            operator = _ccddata_operator(op)
            res = CCDData_astype(eval(f"im1.{operator}(im2)"), dtype=dtype)  # in CCDData

        else:  # Only work with ndarray (and extract only one header)
            if op not in ['**', '%', '//', '+', '-', '*', '/']:
                raise ValueError(f"Operator {op} is not supported")

            # im1 and im2 are all converted to ndarray.
            im1, im2, name1, name2, hdr_ref = _load_im_name_hdr(
                im1, im2, name1, name2, extension1, extension2,
                force_ccddata=False, offsets=offsets, verbose=verbose
            )
            # ``eval`` takes nearly identical time to many if-blocks of ``if op == '+'`` etc:
            res = fits.PrimaryHDU(data=eval(f'(im1 {op} im2)').astype(dtype))  # HDU

        if hdr_ref is None:
            hdr_ref = res.header
        _replace_nan(res, hdr_ref)
        _update_hdr(res.header, header_params, name1, op, name2, output,
                    error_calc=error_calc, t_ref=_t, verbose=verbose)
        res.header = hdr_ref

        if output is not None:
            if isinstance(res, CCDData):
                res.write(output, overwrite=overwrite)
            else:
                res.writeto(output, overwrite=overwrite)

        return res
