"""
Basic image registration tools
"""

__classification__ = "UNCLASSIFIED"
__author__ = 'Thomas McCullough'

import logging
from typing import Sequence

import numpy
from scipy.optimize import minimize

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.geometry.geocoords import ecf_to_geodetic
from sarpy.geometry.point_projection import ground_to_image


logger = logging.getLogger(__name__)


def best_physical_location_fit(structs, locs, **minimization_args):
    """
    Given a collection of SICD and/or SIDDs and a collection of image coordinates, 
    each of which identifies the pixel location of the same feature in the 
    respective image, determine the (best fit) geophysical location of this feature. 

    This assumes that any adjustable parameters used for the SICD/SIDD projection 
    model have already been applied (via :func:`define_coa_projection`).

    Parameters
    ----------
    structs : Sequence[SICDType|SIDDType1|SIDDType2]
        The collection of sicds/sidds, of length `N`
    locs : numpy.ndarray|list|tuple
        The image coordinate collection, of shape `(N, 2)`
    minimization_args
        The keyword arguments (after `args` argument) passed through to
        :func:`scipy.optimize.minimize`. This will default to `'Powell'` 
        optimization, which seems generally much more reliable for this 
        problem than the steepest descent based approaches.

    Returns
    -------
    ecf_location : numpy.ndarray
        The location in question, in ECF coordinates
    residue : float
        The mean square residue of the physical distance between the given
        location and the image locations projected into the surface of
        given HAE value
    result
        The minimization result object
    """

    def get_mean_location(hae_value, log_residue=False):
        ecf_locs = numpy.zeros((points, 3), dtype='float64')
        for i, (loc, struct) in enumerate(zip(locs, structs)):
            ecf_locs[i, :] = struct.project_image_to_ground(loc, projection_type='HAE', hae0=hae_value)
        ecf_mean = numpy.mean(ecf_locs, axis=0)
        diff = ecf_locs - ecf_mean
        residue = numpy.sum(diff*diff, axis=1)
        if log_residue:
            logger.info(
                'best physical location residues [m^2]\n{}'.format(residue))

        avg_residue = numpy.mean(residue)
        return ecf_mean, avg_residue

    def average_residue(hae_value):
        return get_mean_location(hae_value)[1]

    points = len(structs)
    if points < 2:
        raise ValueError(
            'At least 2 structs must be present to determine the best location')
    if points != locs.shape[0]:
        raise ValueError(
            'The number of structs must match the number of locations')

    struct0 = structs[0]
    if isinstance(struct0, SICDType):
        h0 = struct0.GeoData.SCP.LLH.HAE
    elif isinstance(struct0, (SIDDType1, SIDDType2)):
        ref_ecf = struct0.Measurement.ReferencePoint.ECEF.get_array()
        h0 = ecf_to_geodetic(ref_ecf)[2]
    else:
        raise TypeError('Got unexpected structure type {}'.format(type(struct0)))

    if 'method' not in minimization_args:
        minimization_args['method'] = 'Powell'

    result = minimize(average_residue, h0, **minimization_args)
    if not result.success:
        raise ValueError('Optimization failed {}'.format(result))

    values = get_mean_location(result.x, log_residue=True)
    return values[0], values[1], result


def _find_best_adjustable_parameters_sicd(sicd, ecf_coords, img_coords, **minimization_args):
    """
    Find the best projection model adjustable parameters (in `'ECF'` coordinate frame)
    to fit the geophyscial coordinate locations to the image coordinate locations.

    Parameters
    ----------
    sicd : SICDType
    ecf_coords : numpy.ndarray
        The geophysical coordinates of shape `(N, 3)` for the identified features
    img_coords : numpy.ndarray
        The image coordinates of shape `(N, 2)` for the identified features
    minimization_args
        The keyword arguments (after `args` argument) passed through to
        :func:`scipy.optimize.minimize`. This will default to `'Powell'`
        optimization, which seems generally much more reliable for this
        problem than the steepest descent based approaches.

    Returns
    -------
    delta_arp : numpy.ndarray
    delta_varp : numpy.ndarray
    delta_range : float
    residue : float
        Average pixel coordinate distance across the features between projected
        and observed pixel locations
    result
        The minimization result
    """

    row = sicd.Grid.Row.UVectECF.get_array()
    col = sicd.Grid.Col.UVectECF.get_array()

    def get_params(perturb):
        da = perturb[0]*row + perturb[1]*col
        dv = perturb[2:5]
        dr = perturb[5]
        return da, dv, dr

    def get_sq_residue(perturb):
        da, dv, dr = get_params(perturb)
        img_proj, _, _ = ground_to_image(
            ecf_coords, sicd, max_iterations=100, use_structure_coa=False,
            delta_arp=da, delta_varp=dv, range_bias=dr,
            adj_params_frame='ECF')
        diff = (img_proj - img_coords)
        return numpy.sum(diff*diff, axis=1)

    def average_residue(perturb):
        res = get_sq_residue(perturb)
        return numpy.mean(res)

    if 'method' not in minimization_args:
        minimization_args['method'] = 'Powell'

    p0 = numpy.zeros((6, ), dtype='float64')
    result = minimize(average_residue, p0, **minimization_args)
    if not result.success:
        raise ValueError('Optimization failed {}'.format(result))

    logger.info(
        'best adjustable parameters residues [pix^2]\n{}'.format(
            get_sq_residue(result.x)))

    delta_arp, delta_varp, delta_range = get_params(result.x)
    return delta_arp, delta_varp, delta_range, result.fun, result


def _find_best_adjustable_parameters(struct, ecf_coords, img_coords, **minimization_args):
    """
    Find the best projection model adjustable parameters (in `'ECF'` coordinate frame)
    to fit the geophyscial coordinate locations to the image coordinate locations.

    Parameters
    ----------
    struct : SICDType|SIDDType1|SIDDType2
    ecf_coords : numpy.ndarray
        The geophysical coordinates of shape `(N, 3)` for the identified features
    img_coords : numpy.ndarray
        The image coordinates of shape `(N, 2)` for the identified features
    minimization_args
        The keyword arguments (after `args` argument) passed through to
        :func:`scipy.optimize.minimize`. This will default to `'Powell'`
        optimization, which seems generally much more reliable for this
        problem than the steepest descent based approaches.

    Returns
    -------
    delta_arp : numpy.ndarray
    delta_varp : numpy.ndarray
    delta_range : float
    residue : float
        Average pixel coordinate distance across the features between projected
        and observed pixel locations
    result
        The minimization result
    """

    def get_params(perturb):
        da = perturb[0:3]
        dv = perturb[3:6]
        dr = perturb[6]
        return da, dv, dr

    def get_sq_residue(perturb):
        da, dv, dr = get_params(perturb)
        img_proj, _, _ = ground_to_image(
            ecf_coords, struct, max_iterations=100, use_structure_coa=False,
            delta_arp=da, delta_varp=dv, range_bias=dr,
            adj_params_frame='ECF')
        diff = (img_proj - img_coords)
        return numpy.sum(diff*diff, axis=1)

    def average_residue(perturb):
        return numpy.mean(get_sq_residue(perturb))

    if 'method' not in minimization_args:
        minimization_args['method'] = 'Powell'

    p0 = numpy.zeros((7,), dtype='float64')
    result = minimize(average_residue, p0, **minimization_args)
    if not result.success:
        raise ValueError('Optimization failed {}'.format(result))

    logger.info(
        'best adjustable parameters residues [pix^2]\n{}'.format(
            get_sq_residue(result.x)))

    delta_arp, delta_varp, delta_range = get_params(result.x)
    return delta_arp, delta_varp, delta_range, result.fun, result


def find_best_adjustable_parameters(struct, ecf_coords, img_coords, **minimization_args):
    """
    Find the best projection model adjustable parameters (in `'ECF'` coordinate frame)
    to fit the geophyscial coordinate locations to the image coordinate locations.

    Parameters
    ----------
    struct : SICDType|SIDDType1|SIDDType2
    ecf_coords : numpy.ndarray
        The geophysical coordinates of shape `(N, 3)` for the identified features
    img_coords : numpy.ndarray
        The image coordinates of shape `(N, 2)` for the identified features
    minimization_args
        The keyword arguments (after `args` argument) passed through to
        :func:`scipy.optimize.minimize`. This will default to `'Powell'`
        optimization, which seems generally much more reliable for this
        problem than the steepest descent based approaches.

    Returns
    -------
    delta_arp : numpy.ndarray
    delta_varp : numpy.ndarray
    delta_range : float
    residue : float
        Average pixel coordinate distance across the features between projected
        and observed pixel locations
    result
        The minimization result
    """

    return _find_best_adjustable_parameters(struct, ecf_coords, img_coords, **minimization_args)
