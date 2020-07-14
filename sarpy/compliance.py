# -*- coding: utf-8 -*-
"""
Some basic item definitions for python 2 & 3 dual use.
"""

import sys

integer_types = (int, )
string_types = (str, )
int_func = int

if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    from cStringIO import StringIO
    # noinspection PyUnresolvedReferences
    int_func = long
    # noinspection PyUnresolvedReferences
    integer_types = (int, long)
    # noinspection PyUnresolvedReferences
    string_types = (str, unicode)
else:
    # noinspection PyUnresolvedReferences
    from io import StringIO
