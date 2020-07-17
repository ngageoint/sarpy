# -*- coding: utf-8 -*-
"""
Common functionality for converting SIDD metadata.
"""

from datetime import datetime
import numpy

from ..complex.utils import two_dim_poly_fit, get_im_physical_coords
from .sidd2_elements.SIDD import SIDDType
from .sidd2_elements.ProductCreation import ProductCreationType
from .sidd2_elements.Display import ProductDisplayType
from .sidd2_elements.GeoData import GeoDataType
from .sidd2_elements.Measurement import MeasurementType, PlaneProjectionType, ProductPlaneType
from .sidd2_elements.ExploitationFeatures import ExploitationFeaturesType
from .sidd2_elements.blocks import ReferencePointType, Poly2DType
from sarpy.processing.ortho_rectify import OrthorectificationHelper, NearestNeighborMethod, \
    BivariateSplineMethod, ProjectionHelper, PGProjection


def create_display(pixel_type):
    """
    Create the ProductDisplay version 2.0 structure.

    Parameters
    ----------
    pixel_type : str
        Must be one of `MONO8I, MONO16I` or `RGB24I`.

    Returns
    -------
    ProductDisplayType
    """

    pixel_type = pixel_type.upper()

    if pixel_type in ('MONO8I', 'MONO16I'):
        bands = 1
    elif pixel_type == 'RGB24I':
        bands = 3
    else:
        raise ValueError('pixel_type must be one of MONO8I, MONO16I, RGB24I. Got {}'.format(pixel_type))

    return ProductDisplayType(PixelType=pixel_type, NumBands=bands)


def fit_timecoa_poly(proj_helper, bounds):
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
    sidd_timecoa_coeffs = two_dim_poly_fit(
        ortho_grid[:, :, 0] - bounds[0], ortho_grid[:, :, 1] - bounds[2], timecoa_values,
        x_order=use_order, y_order=use_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
    return Poly2DType(Coefs=sidd_timecoa_coeffs)


def _create_measurement_from_pgprojection(proj_helper, bounds):
    """
    Construct the Measurement version 2.0 structure.

    Parameters
    ----------
    proj_helper : PGProjection
    bounds : numpy.ndarray
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.

    Returns
    -------
    MeasurementType
    """

    # create plane_projection
    # fit the time coa polynomial in ortho-pixel coordinates
    plane_projection = PlaneProjectionType(
        ReferencePoint=ReferencePointType(ECEF=proj_helper.reference_point,
                                          Point=(-bounds[0], -bounds[2])),
        SampleSpacing=(proj_helper.row_spacing, proj_helper.col_spacing),
        TimeCOAPoly=fit_timecoa_poly(proj_helper, bounds),
        ProductPlane=ProductPlaneType(RowUnitVector=proj_helper.row_vector,
                                      ColUnitVector=proj_helper.col_vector))

    return MeasurementType(PixelFootprint=(bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1),
                           PlaneProjection=plane_projection,
                           ARPPoly=Poly2DType(Coefs=proj_helper.sicd.Position.ARPPoly.get_array()))


def create_measurement(ortho_helper, bounds):
    """
    Construct the Measurement version 2.0 structure.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
    bounds : numpy.ndarray
        The orthorectification pixel bounds of the form `(min row, max row, min col, max col)`.

    Returns
    -------
    MeasurementType
    """

    proj_helper = ortho_helper.proj_helper

    if isinstance(proj_helper, PGProjection):
        return _create_measurement_from_pgprojection(proj_helper, bounds)
    else:
        return None


def create_sidd_shell(ortho_helper, bounds, product_class, pixel_type):
    """
    Create a SIDD version 2.0 basic shell based on the orthorectification helper
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
    SIDDType
    """

    # validate bounds and get pixel coordinates rectangle
    bounds, pixel_corners = ortho_helper.bounds_to_rectangle(bounds)

    # construct appropriate SIDD elements
    prod_create = ProductCreationType.from_sicd(ortho_helper.proj_helper.sicd, product_class)
    # Display requires more product specifics
    display = create_display(pixel_type)
    # GeoData
    llh_corners = ortho_helper.proj_helper.ortho_to_llh(pixel_corners)
    geo_data = GeoDataType(ImageCorners=llh_corners[:, :2])
    # Measurement
    measurement = create_measurement(ortho_helper, bounds)
    # ExploitationFeatures
    exploit_feats = ExploitationFeaturesType.from_sicd(ortho_helper.proj_helper.sicd)
    return SIDDType(ProductCreation=prod_create,
                    GeoData=geo_data,
                    Display=display,
                    Measurement=measurement,
                    ExploitationFeatures=exploit_feats)
