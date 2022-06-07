"""
This package contains the elements for interpreting product data.
"""

__classification__ = 'UNCLASSIFIED'


def open(*args, **kwargs):
    from .converter import open_product
    return open_product(*args, **kwargs)
