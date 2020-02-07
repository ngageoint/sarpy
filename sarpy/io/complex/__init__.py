# -*- coding: utf-8 -*-
"""
This package contains the elements for interpreting complex radar data in a variety of formats.
For non-SICD files, the radar metadata will be converted to something compatible with the SICD
standard, to the extent feasible.

It also permits converting complex data from any form which can be read to a file or files in
SICD or SIO format.
"""

from .converter import open_complex as open, conversion_utility as convert

