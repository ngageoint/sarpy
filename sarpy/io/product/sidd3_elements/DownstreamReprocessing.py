"""
Types implementing the SIDD 3.0 DownstreamReprocessing Parameters
"""
__classification__ = 'UNCLASSIFIED'

# SIDD 3.0 reuses the SIDD 2.0 types.  Make those symbols available in this module.
from sarpy.io.product.sidd2_elements.DownstreamReprocessing import (
    GeometricChipType,
    ProcessingEventType,
    DownstreamReprocessingType,
)

__REUSED__ = (  # to avoid unused import lint errors
    GeometricChipType,
    ProcessingEventType,
    DownstreamReprocessingType,
)
