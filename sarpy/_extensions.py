__classification__ = "UNCLASSIFIED"

import importlib.metadata


def entry_points(*, group):
    """
    Simple wrapper around importlib.metadata.entry_points

    Parameters
    ----------
    group : str
        entry point group name

    Returns
    -------
    list of entry points belonging to group

    Notes
    -----
    This function is only needed to support Python < 3.10.
    importlib.metadata was introduced in Python 3.8 as a provisional module.
    The stable interface was introduced in Python 3.10 and the original interface was removed in 3.12.
    """

    eps = importlib.metadata.entry_points()
    if hasattr(eps, 'select'):
        # Python >= 3.10
        return eps.select(group=group)
    else:
        return eps.get(group, [])