# -*- coding: utf-8 -*-
"""
Common use sicd_elements methods.
"""


def _get_center_frequency(RadarCollection, ImageFormation):
    """
    Helper method.

    Parameters
    ----------
    RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
    ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

    Returns
    -------
    None|float
        The center processed frequency, in the event that RadarCollection.RefFreqIndex is `None` or `0`.
    """

    if RadarCollection is None or RadarCollection.RefFreqIndex is None or RadarCollection.RefFreqIndex == 0:
        return None
    if ImageFormation is None or ImageFormation.TxFrequencyProc is None or \
            ImageFormation.TxFrequencyProc.MinProc is None or ImageFormation.TxFrequencyProc.MaxProc is None:
        return None
    return 0.5 * (ImageFormation.TxFrequencyProc.MinProc + ImageFormation.TxFrequencyProc.MaxProc)
