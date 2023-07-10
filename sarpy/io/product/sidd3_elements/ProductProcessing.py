"""
Types implementing the SIDD 3.0 ProductProcessing Parameters
"""
__classification__ = 'UNCLASSIFIED'

# SIDD 3.0 reuses the SIDD 2.0 types.  Make those symbols available in this module.
from sarpy.io.product.sidd2_elements.ProductProcessing import (
    ProcessingModuleType,
    ProductProcessingType,
)

__REUSED__ = (  # to avoid unused import lint errors
    ProcessingModuleType,
    ProductProcessingType,
)
