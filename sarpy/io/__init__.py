# -*- coding: utf-8 -*-

from .complex.converter import open_complex
from .phase_history.converter import open_phase_history
from .product.converter import open_product

__classification__ = "UNCLASSIFIED"

def open(file_name):
    """
    Given a file, try to find and return the appropriate reader object.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    BaseReader

    Raises
    ------
    IOError
    """

    try:
        return open_complex(file_name)
    except IOError:
        pass

    try:
        return open_phase_history(file_name)
    except IOError:
        pass

    try:
        return open_product(file_name)
    except IOError:
        pass

    raise IOError(
        'The format of file {} does not match any reader in the complex, '
        'phase_history, or product modules.')
