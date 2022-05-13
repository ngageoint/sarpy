
__classification__ = "UNCLASSIFIED"


def open(file_name: str):
    """
    Given a file, try to find and return the appropriate reader object.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    AbstractReader

    Raises
    ------
    SarpyIOError
    """

    from .complex.converter import open_complex
    from .product.converter import open_product
    from .phase_history.converter import open_phase_history
    from .received.converter import open_received
    from .general.converter import open_general
    from .general.base import SarpyIOError

    try:
        return open_complex(file_name)
    except SarpyIOError:
        pass

    try:
        return open_product(file_name)
    except SarpyIOError:
        pass

    try:
        return open_phase_history(file_name)
    except SarpyIOError:
        pass

    try:
        return open_received(file_name)
    except SarpyIOError:
        pass

    try:
        return open_general(file_name)
    except SarpyIOError:
        pass

    raise SarpyIOError(
        'The format of file {} does not match any reader in the complex, '
        'product, phase_history, received, or general modules.'.format(file_name))
