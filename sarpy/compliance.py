"""
Some basic item definitions for python 2 & 3 dual use.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import sys
from xml.dom import minidom  # this is for python 2 import problems, leave this here

integer_types = (int, )
string_types = (str, )
int_func = int

if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    from cStringIO import StringIO
    BytesIO = StringIO
    # noinspection PyUnresolvedReferences
    int_func = long
    # noinspection PyUnresolvedReferences
    integer_types = (int, long)
    # noinspection PyUnresolvedReferences
    string_types = (str, unicode)
else:
    # noinspection PyUnresolvedReferences
    from io import StringIO
    from io import BytesIO


def bytes_to_string(bytes_in, encoding='utf-8'):
    """
    Ensure that the input bytes is mapped to a

    Parameters
    ----------
    bytes_in : bytes
    encoding : str
        The encoding to apply, if necessary.

    Returns
    -------
    str
    """

    if isinstance(bytes_in, string_types):
        return bytes_in

    if not isinstance(bytes_in, bytes):
        raise TypeError('Input is required to be bytes. Got type {}'.format(type(bytes_in)))

    return bytes_in.decode(encoding)
