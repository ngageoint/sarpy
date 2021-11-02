
__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def bytes_to_string(bytes_in, encoding='utf-8'):
    """
    Ensure that the input bytes is mapped to a string.

    Parameters
    ----------
    bytes_in : bytes
    encoding : str
        The encoding to apply, if necessary.

    Returns
    -------
    str
    """

    if isinstance(bytes_in, str):
        return bytes_in

    if not isinstance(bytes_in, bytes):
        raise TypeError('Input is required to be bytes. Got type {}'.format(type(bytes_in)))

    return bytes_in.decode(encoding)
