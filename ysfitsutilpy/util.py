from astropy import units as u

__all__ = ["key_mapper", "change_to_quantity"]


def key_mapper(header, keymap, deprecation=False):
    ''' Update the header to meed the standard (keymap).
    Parameters
    ----------
    header: Header
        The header to be modified
    keymap: dict
        The dictionary contains ``{<standard_key>:<original_key>}`` information
    deprecation: bool, optional
        Whether to change the original keywords' comments to contain
        deprecation warning. If ``True``, the original keywords' comments will
        become ``Deprecated. See <standard_key>.``.
    '''
    newhdr = header.copy()
    for k_new, k_old in keymap.items():
        # if k_new already in the header, only deprecate k_old.
        # if not, copy k_old to k_new and deprecate k_old.
        if k_old is not None:
            if k_new in newhdr:
                if deprecation:
                    newhdr.comments[k_old] = f"Deprecated. See {k_new}"
            else:
                try:
                    comment_ori = newhdr.comments[k_old]
                    newhdr[k_new] = (newhdr[k_old], comment_ori)
                    if deprecation:
                        newhdr.comments[k_old] = f"Deprecated. See {k_new}"
                except KeyError:
                    pass

    return newhdr


def change_to_quantity(x, desired=None):
    ''' Change the non-Quantity object to astropy Quantity.
    Parameters
    ----------
    x: object changable to astropy Quantity
        The input to be changed to a Quantity. If a Quantity is given, ``x`` is
        changed to the ``desired``, i.e., ``x.to(desired)``.
    desired: astropy Unit, optional
        The desired unit for ``x``.
    Returns
    -------
    ux: Quantity
    Note
    ----
    If Quantity, transform to ``desired``. If ``desired = None``, return it as
    is. If not Quantity, multiply the ``desired``. ``desired = None``, return
    ``x`` with dimensionless unscaled unit.
    '''
    if not isinstance(x, u.quantity.Quantity):
        if desired is None:
            ux = x * u.dimensionless_unscaled
        else:
            ux = x * desired
    else:
        if desired is None:
            ux = x
        else:
            ux = x.to(desired)
    return ux