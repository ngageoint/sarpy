# -*- coding: utf-8 -*-
"""
Common functionality for creating the SIDD structure from a SICD structure and
OrthorectificationHelper.
"""

import numpy

from sarpy.io.complex.utils import two_dim_poly_fit, get_im_physical_coords
from sarpy.processing.ortho_rectify import OrthorectificationHelper, ProjectionHelper, \
    PGProjection
# agnostic to version
from sarpy.io.product.sidd2_elements.ProductCreation import ProductCreationType
from sarpy.io.product.sidd2_elements.Measurement import PlaneProjectionType, ProductPlaneType
# version 2 elements
from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.io.product.sidd2_elements.Display import ProductDisplayType as ProductDisplayType2, \
    NonInteractiveProcessingType, ProductGenerationOptionsType, RRDSType
from sarpy.io.product.sidd2_elements.GeoData import GeoDataType as GeoDataType2
from sarpy.io.product.sidd2_elements.Measurement import MeasurementType as MeasurementType2
from sarpy.io.product.sidd2_elements.ExploitationFeatures import ExploitationFeaturesType as ExploitationFeaturesType2
from sarpy.io.product.sidd2_elements.blocks import ReferencePointType, Poly2DType, XYZPolyType
# version 1 elements
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.product.sidd1_elements.Display import ProductDisplayType as ProductDisplayType1
from sarpy.io.product.sidd1_elements.GeographicAndTarget import GeographicAndTargetType as GeographicAndTargetType1, \
    GeographicCoverageType as GeographicCoverageType1
from sarpy.io.product.sidd1_elements.Measurement import MeasurementType as MeasurementType1
from sarpy.io.product.sidd1_elements.ExploitationFeatures import ExploitationFeaturesType as ExploitationFeaturesType1


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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
    # fit this at the ortho_grid coordinates
    sidd_timecoa_coeffs, residuals, rank, sing_values = two_dim_poly_fit(
        ortho_grid[:, :, 0] - bounds[0], ortho_grid[:, :, 1] - bounds[2], timecoa_values,
        x_order=use_order, y_order=use_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
    import logging
    logging.warning('The time_coa_fit details:\nroot mean square residuals = {}\nrank = {}\nsingular values = {}'.format(residuals, rank, sing_values))
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

    return PlaneProjectionType(
        ReferencePoint=ReferencePointType(ECEF=proj_helper.reference_point,
                                          Point=(-bounds[0], -bounds[2])),
        SampleSpacing=(proj_helper.row_spacing, proj_helper.col_spacing),
        TimeCOAPoly=_fit_timecoa_poly(proj_helper, bounds),
        ProductPlane=ProductPlaneType(RowUnitVector=proj_helper.row_vector,
                                      ColUnitVector=proj_helper.col_vector))


#########################
# Version 2 element creation

def create_sidd_structure_v2(ortho_helper, bounds, product_class, pixel_type):
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

        return ProductDisplayType2(PixelType=pixel_type, NumBands=bands)

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
            raise ValueError('Unhandled projection helper type {}'.format(type(proj_helper)))

    pixel_type = pixel_type.upper()
    # validate bounds and get pixel coordinates rectangle
    bounds, ortho_pixel_corners = ortho_helper.bounds_to_rectangle(bounds)
    # construct appropriate SIDD elements
    prod_create = ProductCreationType.from_sicd(ortho_helper.proj_helper.sicd, product_class)
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
            raise ValueError('Unhandled projection helper type {}'.format(type(proj_helper)))

    def _create_exploitation_v1():
        proj_helper = ortho_helper.proj_helper
        if isinstance(proj_helper, PGProjection):
            return ExploitationFeaturesType1.from_sicd(
                proj_helper.sicd, proj_helper.row_vector, proj_helper.col_vector)
        else:
            raise ValueError('Unhandled projection helper type {}'.format(type(proj_helper)))

    pixel_type = pixel_type.upper()
    # validate bounds and get pixel coordinates rectangle
    bounds, ortho_pixel_corners = ortho_helper.bounds_to_rectangle(bounds)
    # construct appropriate SIDD elements
    prod_create = ProductCreationType.from_sicd(ortho_helper.proj_helper.sicd, product_class)
    prod_create.Classification.DESVersion = 4

    # Display requires more product specifics
    display = _create_display_v1()
    # GeographicAndTarget
    llh_corners = ortho_helper.proj_helper.ortho_to_llh(ortho_pixel_corners)
    geographic = GeographicAndTargetType1(GeographicCoverage=GeographicCoverageType1(Footprint=llh_corners[:, :2]))
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

def create_sidd_structure(ortho_helper, bounds, product_class, pixel_type, version=2):
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
        The SIDD version, must be either 1 or 2.

    Returns
    -------
    SIDDType1|SIDDType2
    """

    if version not in [1, 2]:
        raise ValueError('version must be 1 or 2. Got {}'.format(version))

    if version == 1:
        return create_sidd_structure_v1(ortho_helper, bounds, product_class, pixel_type)
    else:
        return create_sidd_structure_v2(ortho_helper, bounds, product_class, pixel_type)
