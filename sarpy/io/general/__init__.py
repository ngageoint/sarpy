"""
This package mostly centered on base implementations for reader architecture.
"""


def open(*args, **kwargs):
    from .converter import open_general
    return open_general(*args, **kwargs)
