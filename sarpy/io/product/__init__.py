# -*- coding: utf-8 -*-
"""
This package contains the elements for interpreting product data.
"""


def open(*args, **kwargs):
    from .converter import open_product
    return open_product(*args, **kwargs)
