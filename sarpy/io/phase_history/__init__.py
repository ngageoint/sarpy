"""
This package contains the elements for interpreting phase history data.
"""

__classification__ = 'UNCLASSIFIED'


def open(*args, **kwargs):
    from .converter import open_phase_history
    return open_phase_history(*args, **kwargs)
