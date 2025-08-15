"""
Common functionality for creating the SIDD structure from a SICD structure and
OrthorectificationHelper.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import numpy

from sarpy.io.complex.utils import two_dim_poly_fit, get_im_physical_coords
from sarpy.processing.ortho_rectify import OrthorectificationHelper, ProjectionHelper, \
    PGProjection
# agnostic to versions 1 & 2
from sarpy.io.product.sidd2_elements.Measurement import PlaneProjectionType, ProductPlaneType
from sarpy.io.product.sidd2_elements.blocks import ReferencePointType, Poly2DType, XYZPolyType, \
    FilterType, FilterBankType, PredefinedFilterType, NewLookupTableType, PredefinedLookupType
# version 3 elements
from sarpy.io.product.sidd3_elements.SIDD import SIDDType as SIDDType3
from sarpy.io.product.sidd3_elements.Display import ProductDisplayType as ProductDisplayType3, \
    NonInteractiveProcessingType as NonInteractiveProcessingType3, \
    ProductGenerationOptionsType as ProductGenerationOptionsType3, RRDSType as RRDSType3, \
    InteractiveProcessingType as InteractiveProcessingType3, GeometricTransformType as GeometricTransformType3, \
    SharpnessEnhancementType as SharpnessEnhancementType3, DynamicRangeAdjustmentType as DynamicRangeAdjustmentType3, \
    ScalingType as ScalingType3, OrientationType as OrientationType3
from sarpy.io.product.sidd3_elements.GeoData import GeoDataType as GeoDataType3
from sarpy.io.product.sidd3_elements.Measurement import MeasurementType as MeasurementType3, \
    PlaneProjectionType as PlaneProjectionType3, ProductPlaneType as ProductPlaneType3
from sarpy.io.product.sidd3_elements.ExploitationFeatures import ExploitationFeaturesType as ExploitationFeaturesType3
from sarpy.io.product.sidd3_elements.ProductCreation import ProductCreationType as ProductCreationType3
from sarpy.io.product.sidd3_elements.blocks import ReferencePointType as ReferencePointType3, \
    Poly2DType as Poly2DType3, XYZPolyType as XYZPolyType3, FilterType as FilterType3, \
    FilterBankType as FilterBankType3, PredefinedFilterType as PredefinedFilterType3, \
    NewLookupTableType as NewLookupTableType3, PredefinedLookupType as PredefinedLookupType3
# version 2 elements
from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.io.product.sidd2_elements.Display import ProductDisplayType as ProductDisplayType2, \
    NonInteractiveProcessingType as NonInteractiveProcessingType2, \
    ProductGenerationOptionsType as ProductGenerationOptionsType2, RRDSType as RRDSType2, \
    InteractiveProcessingType as InteractiveProcessingType2, GeometricTransformType as GeometricTransformType2, \
    SharpnessEnhancementType as SharpnessEnhancementType2, DynamicRangeAdjustmentType as DynamicRangeAdjustmentType2, \
    ScalingType as ScalingType2, OrientationType as OrientationType2
from sarpy.io.product.sidd2_elements.GeoData import GeoDataType as GeoDataType2
from sarpy.io.product.sidd2_elements.Measurement import MeasurementType as MeasurementType2
from sarpy.io.product.sidd2_elements.ExploitationFeatures import ExploitationFeaturesType as ExploitationFeaturesType2
from sarpy.io.product.sidd2_elements.ProductCreation import ProductCreationType as ProductCreationType2
# version 1 elements
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.product.sidd1_elements.Display import ProductDisplayType as ProductDisplayType1
from sarpy.io.product.sidd1_elements.GeographicAndTarget import GeographicAndTargetType as GeographicAndTargetType1, \
    GeographicCoverageType as GeographicCoverageType1, GeographicInformationType as GeographicInformationType1
from sarpy.io.product.sidd1_elements.Measurement import MeasurementType as MeasurementType1
from sarpy.io.product.sidd1_elements.ExploitationFeatures import ExploitationFeaturesType as ExploitationFeaturesType1
from sarpy.io.product.sidd1_elements.ProductCreation import ProductCreationType as ProductCreationType1
import sarpy.visualization.remap as remap

logger = logging.getLogger(__name__)

_proj_helper_text = 'Unhandled projection helper type `{}`'

# TODO: move this to processing for 1.3.0


def _fit_timecoa_poly(proj_helper, bounds):
    """
    Fit the TimeCOA in new pixel coordinates.

    Parameters
    ----------
    proj_helper : ProjectionHelper
    bounds : numpy.ndarray
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.

    Returns
    -------
    Poly2DType
    """

    # what is the order of the sicd timecoapoly?
    in_poly = proj_helper.sicd.Grid.TimeCOAPoly
    use_order = max(in_poly.order1, in_poly.order2)
    if use_order == 0:
        # this is a constant polynomial, must be a spotlight collect
        return Poly2DType(Coefs=in_poly.get_array())
    # create an ortho coordinate grid
    samples = use_order+10
    ortho_grid = numpy.zeros((samples, samples, 2), dtype=numpy.float64)
    ortho_grid[:, :, 1], ortho_grid[:, :, 0] = numpy.meshgrid(
        numpy.linspace(bounds[2], bounds[3], num=samples),
        numpy.linspace(bounds[0], bounds[1], num=samples))
    # map to pixel grid coordinates
    pixel_grid = proj_helper.ortho_to_pixel(ortho_grid)
    pixel_rows_m = get_im_physical_coords(
        pixel_grid[:, :, 0], proj_helper.sicd.Grid, proj_helper.sicd.ImageData, 'row')
    pixel_cols_m = get_im_physical_coords(
        pixel_grid[:, :, 1], proj_helper.sicd.Grid, proj_helper.sicd.ImageData, 'col')
    # evaluate the sicd timecoapoly
    timecoa_values = proj_helper.sicd.Grid.TimeCOAPoly(pixel_rows_m, pixel_cols_m)
    sidd_timecoa_coeffs, residuals, rank, sing_values = two_dim_poly_fit(
        pixel_rows_m, pixel_cols_m, timecoa_values,
        x_order=use_order, y_order=use_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
    logger.info(
        'The time_coa_fit details:\n\t'
        'root mean square residuals = {}\n\t'
        'rank = {}\n\t'
        'singular values = {}'.format(residuals, rank, sing_values))
    return Poly2DType(Coefs=sidd_timecoa_coeffs)


def _create_plane_projection(proj_helper, bounds):
    """
    Construct the PlaneProjection structure for both version 1 & 2.

    Parameters
    ----------
    proj_helper : PGProjection
    bounds : numpy.ndarray
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.

    Returns
    -------
    PlaneProjectionType
    """

    ref_pixels = proj_helper.reference_pixels
    return PlaneProjectionType(
        ReferencePoint=ReferencePointType(ECEF=proj_helper.reference_point,
                                          Point=(float(ref_pixels[0]-bounds[0]), float(ref_pixels[1]-bounds[2]))),
        SampleSpacing=(proj_helper.row_spacing, proj_helper.col_spacing),
        TimeCOAPoly=_fit_timecoa_poly(proj_helper, bounds),
        ProductPlane=ProductPlaneType(RowUnitVector=proj_helper.row_vector,
                                      ColUnitVector=proj_helper.col_vector))


def _fit_timecoa_poly_v3(proj_helper, bounds):
    """
    Fit the TimeCOA in new pixel coordinates.

    Parameters
    ----------
    proj_helper : ProjectionHelper
    bounds : numpy.ndarray
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.

    Returns
    -------
    Poly2DType3
    """

    # what is the order of the sicd timecoapoly?
    in_poly = proj_helper.sicd.Grid.TimeCOAPoly
    use_order = max(in_poly.order1, in_poly.order2)
    if use_order == 0:
        # this is a constant polynomial, must be a spotlight collect
        return Poly2DType3(Coefs=in_poly.get_array())
    # create an ortho coordinate grid
    samples = use_order+10
    ortho_grid = numpy.zeros((samples, samples, 2), dtype=numpy.float64)
    ortho_grid[:, :, 1], ortho_grid[:, :, 0] = numpy.meshgrid(
        numpy.linspace(bounds[2], bounds[3], num=samples),
        numpy.linspace(bounds[0], bounds[1], num=samples))
    # map to pixel grid coordinates
    pixel_grid = proj_helper.ortho_to_pixel(ortho_grid)
    pixel_rows_m = get_im_physical_coords(
        pixel_grid[:, :, 0], proj_helper.sicd.Grid, proj_helper.sicd.ImageData, 'row')
    pixel_cols_m = get_im_physical_coords(
        pixel_grid[:, :, 1], proj_helper.sicd.Grid, proj_helper.sicd.ImageData, 'col')
    # evaluate the sicd timecoapoly
    timecoa_values = proj_helper.sicd.Grid.TimeCOAPoly(pixel_rows_m, pixel_cols_m)
    # fit this at the ortho_grid coordinates
    sidd_timecoa_coeffs, residuals, rank, sing_values = two_dim_poly_fit(
        pixel_rows_m, pixel_cols_m, timecoa_values,
        x_order=use_order, y_order=use_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
    logger.info(
        'The time_coa_fit details:\n\t'
        'root mean square residuals = {}\n\t'
        'rank = {}\n\t'
        'singular values = {}'.format(residuals, rank, sing_values))
    return Poly2DType3(Coefs=sidd_timecoa_coeffs)


def _create_plane_projection_v3(proj_helper, bounds):
    """
    Construct the PlaneProjection structure for both version 1 & 2.

    Parameters
    ----------
    proj_helper : PGProjection
    bounds : numpy.ndarray
    The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.

    Returns
    -------
    PlaneProjectionType3
    """

    ref_pixels = proj_helper.reference_pixels
    return PlaneProjectionType3(
        ReferencePoint=ReferencePointType3(ECEF=proj_helper.reference_point,
                                           Point=(float(ref_pixels[0]-bounds[0]),
                                                  float(ref_pixels[1]-bounds[2]))),
        SampleSpacing=(proj_helper.row_spacing, proj_helper.col_spacing),
        TimeCOAPoly=_fit_timecoa_poly_v3(proj_helper, bounds),
        ProductPlane=ProductPlaneType3(RowUnitVector=proj_helper.row_vector,
                                       ColUnitVector=proj_helper.col_vector))


#########################
# Version 3 element creation

def create_sidd_structure_v3(ortho_helper, bounds, product_class, pixel_type, 
                             remap_function=remap.get_registered_remap('nrl')):
    """
    Create a SIDD version 3.0 structure based on the orthorectification helper
    and pixel bounds.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
    bounds : numpy.ndarray|list|tuple
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.
    product_class : str
        A descriptive name for the product class. Examples -
        :code:`Dynamic Image, Amplitude Change Detection, Coherent Change Detection`
    pixel_type : str
        Must be one of `MONO8I, MONO16I` or `RGB24I`.
    remap_function: RemapFunction
        Must be an instantiation of the RemapFunction class see sarpy/visualation/remap

    Returns
    -------
    SIDDType3
    """

    def _create_display_v3():
        if pixel_type in ('MONO8I', 'MONO16I'):
            bands = 1
        elif pixel_type == 'RGB24I':
            bands = 3
        else:
            raise ValueError('pixel_type must be one of MONO8I, MONO16I, RGB24I. Got {}'.format(pixel_type))

        return ProductDisplayType3(
            PixelType=pixel_type,
            NumBands=bands,
            NonInteractiveProcessing=[NonInteractiveProcessingType3(
                ProductGenerationOptions=ProductGenerationOptionsType3(
                    DataRemapping=NewLookupTableType3(
                        LUTName=remap_function.name.upper(),
                        Predefined=PredefinedLookupType3(
                            DatabaseName=remap_function.name.upper()))),
                RRDS=RRDSType3(DownsamplingMethod='DECIMATE'),
                band=i+1) for i in range(bands)],
            InteractiveProcessing=[InteractiveProcessingType3(
                GeometricTransform=GeometricTransformType3(
                    Scaling=ScalingType3(
                        AntiAlias=FilterType3(
                            FilterName='AntiAlias',
                            FilterBank=FilterBankType3(
                                Predefined=PredefinedFilterType3(DatabaseName='BILINEAR')),
                            Operation='CONVOLUTION'),
                        Interpolation=FilterType3(
                            FilterName='Interpolation',
                            FilterBank=FilterBankType3(
                                Predefined=PredefinedFilterType3(DatabaseName='BILINEAR')),
                            Operation='CORRELATION')),
                    Orientation=OrientationType3(ShadowDirection='ARBITRARY')),
                SharpnessEnhancement=SharpnessEnhancementType3(
                    ModularTransferFunctionEnhancement=FilterType3(
                        FilterName='ModularTransferFunctionEnhancement',
                        FilterBank=FilterBankType3(
                            Predefined=PredefinedFilterType3(DatabaseName='BILINEAR')),
                        Operation='CONVOLUTION')),
                DynamicRangeAdjustment=DynamicRangeAdjustmentType3(
                    AlgorithmType='NONE',
                    BandStatsSource=1),
                band=i+1) for i in range(bands)])

    def _create_measurement_v3():
        proj_helper = ortho_helper.proj_helper
        rows = bounds[1] - bounds[0]
        cols = bounds[3] - bounds[2]
        if isinstance(proj_helper, PGProjection):
            # fit the time coa polynomial in ortho-pixel coordinates
            plane_projection = _create_plane_projection_v3(proj_helper, bounds)
            return MeasurementType3(PixelFootprint=(rows, cols),
                                    ValidData=((0, 0), (0, cols), (rows, cols), (rows, 0)),
                                    PlaneProjection=plane_projection,
                                    ARPPoly=XYZPolyType3(
                                        X=proj_helper.sicd.Position.ARPPoly.X.get_array(),
                                        Y=proj_helper.sicd.Position.ARPPoly.Y.get_array(),
                                        Z=proj_helper.sicd.Position.ARPPoly.Z.get_array()))
        else:
            return None

    def _create_exploitation_v3():
        proj_helper = ortho_helper.proj_helper
        if isinstance(proj_helper, PGProjection):
            return ExploitationFeaturesType3.from_sicd(
                proj_helper.sicd, proj_helper.row_vector, proj_helper.col_vector)
        else:
            raise ValueError(_proj_helper_text.format(type(proj_helper)))

    pixel_type = pixel_type.upper()
    # validate bounds and get pixel coordinates rectangle
    bounds, ortho_pixel_corners = ortho_helper.bounds_to_rectangle(bounds)
    # construct appropriate SIDD elements
    prod_create = ProductCreationType3.from_sicd(ortho_helper.proj_helper.sicd, product_class)
    prod_create.Classification.ISMCATCESVersion = '201903'
    prod_create.Classification.compliesWith = 'USGov'
    if not isinstance(remap_function, remap.RemapFunction):
        raise TypeError("Input 'remap_function' must be a remap.RemapFunction.")

    # Display requires more product specifics
    display = _create_display_v3()
    # GeoData
    llh_corners = ortho_helper.proj_helper.ortho_to_llh(ortho_pixel_corners)
    geo_data = GeoDataType3(ImageCorners=llh_corners[:, :2], ValidData=llh_corners[:, :2])
    # Measurement
    measurement = _create_measurement_v3()
    # ExploitationFeatures
    exploit_feats = _create_exploitation_v3()
    return SIDDType3(ProductCreation=prod_create,
                     GeoData=geo_data,
                     Display=display,
                     Measurement=measurement,
                     ExploitationFeatures=exploit_feats)


#########################
# Version 2 element creation

def create_sidd_structure_v2(ortho_helper, bounds, product_class, pixel_type, 
                             remap_function=remap.get_registered_remap('nrl') ):
    """
    Create a SIDD version 2.0 structure based on the orthorectification helper
    and pixel bounds.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
    bounds : numpy.ndarray|list|tuple
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.
    product_class : str
        A descriptive name for the product class. Examples -
        :code:`Dynamic Image, Amplitude Change Detection, Coherent Change Detection`
    pixel_type : str
        Must be one of `MONO8I, MONO16I` or `RGB24I`.
    remap_function: RemapFunction
        Must be an instantiation of the RemapFunction class see sarpy/visualation/remap

    Returns
    -------
    SIDDType2
    """

    def _create_display_v2():
        if pixel_type in ('MONO8I', 'MONO16I'):
            bands = 1
        elif pixel_type == 'RGB24I':
            bands = 3
        else:
            raise ValueError('pixel_type must be one of MONO8I, MONO16I, RGB24I. Got {}'.format(pixel_type))

        return ProductDisplayType2(
            PixelType=pixel_type,
            NumBands=bands,
            NonInteractiveProcessing=[NonInteractiveProcessingType2(
                ProductGenerationOptions=ProductGenerationOptionsType2(
                    DataRemapping=NewLookupTableType(
                        LUTName=remap_function.name.upper(),
                        Predefined=PredefinedLookupType(
                            DatabaseName=remap_function.name.upper()))),
                RRDS=RRDSType2(DownsamplingMethod='DECIMATE'),
                band=i+1) for i in range(bands)],
            InteractiveProcessing=[InteractiveProcessingType2(
                GeometricTransform=GeometricTransformType2(
                    Scaling=ScalingType2(
                        AntiAlias=FilterType(
                            FilterName='AntiAlias',
                            FilterBank=FilterBankType(
                                Predefined=PredefinedFilterType(DatabaseName='BILINEAR')),
                            Operation='CONVOLUTION'),
                        Interpolation=FilterType(
                            FilterName='Interpolation',
                            FilterBank=FilterBankType(
                                Predefined=PredefinedFilterType(DatabaseName='BILINEAR')),
                            Operation='CORRELATION')),
                    Orientation=OrientationType2(ShadowDirection='ARBITRARY')),
                SharpnessEnhancement=SharpnessEnhancementType2(
                    ModularTransferFunctionEnhancement=FilterType(
                        FilterName='ModularTransferFunctionEnhancement',
                        FilterBank=FilterBankType(
                            Predefined=PredefinedFilterType(DatabaseName='BILINEAR')),
                        Operation='CONVOLUTION')),
                DynamicRangeAdjustment=DynamicRangeAdjustmentType2(
                    AlgorithmType='NONE',
                    BandStatsSource=1),
                band=i+1) for i in range(bands)])

    def _create_measurement_v2():
        proj_helper = ortho_helper.proj_helper
        rows = bounds[1] - bounds[0]
        cols = bounds[3] - bounds[2]
        if isinstance(proj_helper, PGProjection):
            # fit the time coa polynomial in ortho-pixel coordinates
            plane_projection = _create_plane_projection(proj_helper, bounds)
            return MeasurementType2(PixelFootprint=(rows, cols),
                                    ValidData=((0, 0), (0, cols), (rows, cols), (rows, 0)),
                                    PlaneProjection=plane_projection,
                                    ARPPoly=XYZPolyType(
                                        X=proj_helper.sicd.Position.ARPPoly.X.get_array(),
                                        Y=proj_helper.sicd.Position.ARPPoly.Y.get_array(),
                                        Z=proj_helper.sicd.Position.ARPPoly.Z.get_array()))
        else:
            return None

    def _create_exploitation_v2():
        proj_helper = ortho_helper.proj_helper
        if isinstance(proj_helper, PGProjection):
            return ExploitationFeaturesType2.from_sicd(
                proj_helper.sicd, proj_helper.row_vector, proj_helper.col_vector)
        else:
            raise ValueError(_proj_helper_text.format(type(proj_helper)))

    pixel_type = pixel_type.upper()
    # validate bounds and get pixel coordinates rectangle
    bounds, ortho_pixel_corners = ortho_helper.bounds_to_rectangle(bounds)
    # construct appropriate SIDD elements
    prod_create = ProductCreationType2.from_sicd(ortho_helper.proj_helper.sicd, product_class)
    prod_create.Classification.ISMCATCESVersion = '201903'
    prod_create.Classification.compliesWith = 'USGov'
    if not isinstance(remap_function, remap.RemapFunction):
        raise TypeError("Input 'remap_function' must be a remap.RemapFunction.")

    # Display requires more product specifics
    display = _create_display_v2()
    # GeoData
    llh_corners = ortho_helper.proj_helper.ortho_to_llh(ortho_pixel_corners)
    geo_data = GeoDataType2(ImageCorners=llh_corners[:, :2], ValidData=llh_corners[:, :2])
    # Measurement
    measurement = _create_measurement_v2()
    # ExploitationFeatures
    exploit_feats = _create_exploitation_v2()
    return SIDDType2(ProductCreation=prod_create,
                     GeoData=geo_data,
                     Display=display,
                     Measurement=measurement,
                     ExploitationFeatures=exploit_feats)


##########################
# Version 1 element creation

def create_sidd_structure_v1(ortho_helper, bounds, product_class, pixel_type):
    """
    Create a SIDD version 1.0 structure based on the orthorectification helper
    and pixel bounds.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
    bounds : numpy.ndarray|list|tuple
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.
    product_class : str
        A descriptive name for the product class. Examples -
        :code:`Dynamic Image, Amplitude Change Detection, Coherent Change Detection`
    pixel_type : str
        Must be one of `MONO8I, MONO16I` or `RGB24I`.

    Returns
    -------
    SIDDType1
    """

    def _create_display_v1():
        if pixel_type not in ('MONO8I', 'MONO16I', 'RGB24I'):
            raise ValueError(
                'pixel_type must be one of MONO8I, MONO16I, RGB24I. Got {}'.format(pixel_type))

        return ProductDisplayType1(PixelType=pixel_type)

    def _create_measurement_v1():
        proj_helper = ortho_helper.proj_helper
        if isinstance(proj_helper, PGProjection):
            # fit the time coa polynomial in ortho-pixel coordinates
            plane_projection = _create_plane_projection(proj_helper, bounds)
            return MeasurementType1(PixelFootprint=(bounds[1] - bounds[0], bounds[3] - bounds[2]),
                                    PlaneProjection=plane_projection,
                                    ARPPoly=XYZPolyType(
                                        X=proj_helper.sicd.Position.ARPPoly.X.get_array(),
                                        Y=proj_helper.sicd.Position.ARPPoly.Y.get_array(),
                                        Z=proj_helper.sicd.Position.ARPPoly.Z.get_array()))
        else:
            raise ValueError(_proj_helper_text.format(type(proj_helper)))

    def _create_exploitation_v1():
        proj_helper = ortho_helper.proj_helper
        if isinstance(proj_helper, PGProjection):
            return ExploitationFeaturesType1.from_sicd(
                proj_helper.sicd, proj_helper.row_vector, proj_helper.col_vector)
        else:
            raise ValueError(_proj_helper_text.format(type(proj_helper)))

    pixel_type = pixel_type.upper()
    # validate bounds and get pixel coordinates rectangle
    bounds, ortho_pixel_corners = ortho_helper.bounds_to_rectangle(bounds)
    # construct appropriate SIDD elements
    prod_create = ProductCreationType1.from_sicd(ortho_helper.proj_helper.sicd, product_class)

    # Display requires more product specifics
    display = _create_display_v1()
    # GeographicAndTarget
    llh_corners = ortho_helper.proj_helper.ortho_to_llh(ortho_pixel_corners)
    geographic = GeographicAndTargetType1(
        GeographicCoverage=GeographicCoverageType1(Footprint=llh_corners[:, :2],
                                                   GeographicInfo=GeographicInformationType1()),)
    # Measurement
    measurement = _create_measurement_v1()
    # ExploitationFeatures
    exploit_feats = _create_exploitation_v1()

    return SIDDType1(ProductCreation=prod_create,
                     Display=display,
                     GeographicAndTarget=geographic,
                     Measurement=measurement,
                     ExploitationFeatures=exploit_feats)


##########################
# Switchable version SIDD structure

def create_sidd_structure(ortho_helper, bounds, product_class, pixel_type, 
                          version=3, remap_function=remap.get_registered_remap('nrl')):
    """
    Create a SIDD structure, with version specified, based on the orthorectification
    helper and pixel bounds.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
    bounds : numpy.ndarray|list|tuple
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.
    product_class : str
        A descriptive name for the product class. Examples -
        :code:`Dynamic Image, Amplitude Change Detection, Coherent Change Detection`
    pixel_type : str
        Must be one of `MONO8I, MONO16I` or `RGB24I`.
    version : int
        The SIDD version, must be either 1, 2, or 3.
    remap_function: RemapFunction
        Must be an instantiation of the RemapFunction class see sarpy/visualation/remap

    Returns
    -------
    SIDDType1|SIDDType2|SIDDType3
    """
    if not isinstance(remap_function, remap.RemapFunction):
        raise TypeError("Input 'remap_function' must be a remap.RemapFunction.")
    if version not in [1, 2, 3]:
        raise ValueError('version must be 1, 2, or 3. Got {}'.format(version))

    if version == 1:
        return create_sidd_structure_v1(ortho_helper, bounds, product_class, pixel_type)
    elif version == 2:
        return create_sidd_structure_v2(ortho_helper, bounds, product_class, pixel_type, remap_function=remap_function )
    else:
        return create_sidd_structure_v3(ortho_helper, bounds, product_class, pixel_type, remap_function=remap_function )
