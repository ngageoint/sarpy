"""
Types implementing the SIDD 3.0 DigitalElevationData Parameters
"""
__classification__ = 'UNCLASSIFIED'

# SIDD 3.0 reuses the SIDD 2.0 types.  Make those symbols available in this module.
from sarpy.io.product.sidd2_elements.DigitalElevationData import (
    GeographicCoordinatesType,
    GeopositioningType,
    AccuracyType,
    PositionalAccuracyType,
    DigitalElevationDataType,
)

__REUSED__ = (  # to avoid unused import lint errors
    GeographicCoordinatesType,
    GeopositioningType,
    AccuracyType,
    PositionalAccuracyType,
    DigitalElevationDataType,
)
