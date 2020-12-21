# -*- coding: utf-8 -*-
"""
This package contains the elements for interpreting complex radar data in a variety of formats.
For non-SICD files, the radar metadata will be converted to something compatible with the SICD
standard, to the extent feasible.

It also permits converting complex data from any form which can be read to a file or files in
SICD or SIO format.
"""


def open(*args, **kwargs):
    from .converter import open_complex
    return open_complex(*args, **kwargs)

def convert(*args, **kwargs):
    from .converter import conversion_utility
    return conversion_utility(*args, **kwargs)
