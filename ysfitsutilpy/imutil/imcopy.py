import numpy as np

from ..ccdutil import CCDData_astype, trim_ccd
from ..hdrutil import update_tlm
from ..misc import inputs2list, _parse_image

__all__ = ['imcopy']


# TODO: use fitsio if (outputs is None) and not return_ccd
def imcopy(inputs, extension=None, fits_sections=None, outputs=None, return_ccd=True, dtype='float32',
           **kwargs):
    ''' Similar to IRAF IMCOPY
    Parameters
    ----------
    inputs : glob pattern, list-like of path-like, list-like of CCDData
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or list of files (each element must
        be path-like or CCDData). Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable. For the purpose of imcombine function, the best use is to use the
        `~glob` pattern or list of paths.

    extension : int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    fits_sections : str or array-like of such, optional.
        The section specified by FITS convention, i.e., bracket embraced, comma separated, XY order,
        1-indexing, and including the end index. If given as array-like format of length ``N``, all
        such sections in all FITS files will be extracted.

    outputs : path-like or array-like of such, optional.
        The output paths of each FITS file to be copied. If array-like, it must have the shape of ``(M,
        N)`` where ``M`` and ``N`` are the sizes of `fpaths` and `fits_sections`, respectively.

    return_ccd : bool, optional.
        Whether to load the FITS files as `~astropy.nddata.CCDData` and return it.

    dtype : dtype, optional.
        The dtype for the `outputs` or returning ccds.

    kwargs : optionals
        The keyword arguments for ``CCDData.write``.

    Return
    ------
    results: CCDData or list of CCDData
        Only if `return_ccd` is set `True`. A sinlge `~astropy.nddata.CCDData will be returned if
        only one was input. Otherwise, the same number of `~astropy.nddata.CCDData will be gathered as
        a list and returned.
    Note
    ----
    Due to the memory issue, it is generally better NOT to load all the FITS files and pass them to
    this function. Therefore, as it is in IRAF, I made this function to accept only the file paths, not
    the pre-loaded CCDData objects. I here will load the

    All the sections will be flattened if they are higher than 1-d. I think it will only increase the
    complexity of the code if I accept that...?

    Example
    -------
    >>> from ysfitsutilpy import imcopy
    >>> from pathlib import Path
    >>>
    >>> datapath = Path("./data")
    >>> files = datapath.glob("*.pcr.fits")
    >>> sections = ["[50:100, 50:100]", "[50:100, 50:150]"]
    >>> outputs = [datapath/"test1.fits", datapath/"test2.fits"]
    >>>
    >>> # single file, single section
    >>> trim = imcopy(pcrfits[0], sections[0])
    >>>
    >>> # single file, multi sections
    >>> trims = imcopy(pcrfits[0], sections)
    >>>
    >>> # Save with overwrite option
    >>> imcopy(pcrfits[0], sections, outputs=outputs, overwrite=True)
    >>>
    >>> # multi file multi section
    >>> trims2d = imcopy(pcrfits[:2], fits_sections=sections, outputs=None)
    '''
    to_trim = False
    to_save = False

    str_flat = "{} with dimension higher than 2-d is not supported yet. Currently it's {}-d. Flattening..."
    inputs = inputs2list(inputs, sort=True, accept_ccdlike=True, check_coherency=False)

    m = len(inputs)

    if fits_sections is not None:
        sects = np.atleast_1d(fits_sections)
        to_trim = True
        if sects.ndim > 1:
            print(str_flat.format("fits_sections", sects.ndim))
            sects = sects.flatten()
        n = sects.shape[0]
    else:
        sects = None
        n = 1

    if outputs is not None:
        outputs = np.atleast_2d(outputs)
        to_save = True
        if outputs.ndim > 2:
            raise ValueError("outputs should be lower than 3-d.")
        if outputs.shape != (m, n):
            raise ValueError(
                "If outputs is array-like, it's shape must have the shape of (fpaths.size, "
                + "fits_sections.size)= ({}, {}). Now it's ({}).".format(m, n, *outputs.shape)
            )

    if return_ccd:
        results = []

    # TODO: Use fits.open rather than CCDData for speed isseu.
    for i, item in enumerate(inputs):
        ccd = _parse_image(item, extension=extension, force_ccddata=True)[0]
        result = []
        if to_trim:  # n CCDData will be in `result`
            for sect in sects:
                # FIXME: use ccdproc.trim_image when trim_ccd is removed.
                nccd = trim_ccd(ccd, fits_section=sect)
                nccd = CCDData_astype(nccd, dtype=dtype)
                update_tlm(nccd.header)
                result.append(nccd)
        else:  # only one single CCDData will be in `result`
            nccd = CCDData_astype(ccd, dtype=dtype)
            update_tlm(nccd.header)
            result.append(nccd)

        if to_save:
            for j, res in enumerate(result):
                res.write(outputs[i, j], **kwargs)

        if return_ccd:
            if len(result) == 1:
                results.append(result[0])
            else:
                results.append(result)

    if return_ccd:
        if len(results) == 1:
            return results[0]
        else:
            return results
