"""
This module provides common functions for creating kmz products.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"

import logging

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shg

from sarpy.geometry.geocoords import ecf_to_geodetic

WGS84_SEMIMAJOR = 6378137.0
WGS84_SEMIMINOR = 6356752.314245

logger = logging.getLogger(__name__)


def ecef_to_kml_coord(ecf_points):
    """Convert a list of ECEF points to a list of KML coordinates"""
    # TODO apply geoid.  KML expects MSL
    llh_points = ecf_to_geodetic(ecf_points)
    return [f"{lon},{lat},{alt}" for (lat, lon, alt) in llh_points]


def acf_to_ecef(eb_dcx, eb_dcy, uacx, uacy):
    """Convert an ACF pointing vector into an ECEF pointing vector"""
    uacz = np.cross(uacx, uacy)
    eb_dcz = np.sqrt(1 - eb_dcx**2 - eb_dcy**2)
    eb = np.stack((eb_dcx, eb_dcy, eb_dcz)).T

    eb_pointing = eb[:, 0, np.newaxis] * uacx
    eb_pointing += eb[:, 1, np.newaxis] * uacy
    eb_pointing += eb[:, 2, np.newaxis] * uacz

    return eb_pointing


def add_polygon_style(
    kmz_document, name, *, bbggrr, low_aa, low_width, high_aa, high_width, outline="1"
):
    """Add a polygon style to a KML document"""
    opaque = "ff"
    kmz_document.add_style(
        name + "_high",
        line_style={"color": opaque + bbggrr, "width": high_width},
        poly_style={"color": high_aa + bbggrr, "outline": str(outline)},
    )
    kmz_document.add_style(
        name + "_low",
        line_style={"color": opaque + bbggrr, "width": low_width},
        poly_style={"color": low_aa + bbggrr, "outline": str(outline)},
    )
    kmz_document.add_style_map(name, name + "_high", name + "_low")


def add_antenna_styles(kmz_document):
    """Add KML styles used for rendering antenna information"""
    add_polygon_style(
        kmz_document,
        "mechanical_boresight",
        bbggrr="a00000",
        low_aa="70",
        low_width="2.0",
        high_aa="a0",
        high_width="3.5",
        outline="0",
    )

    add_polygon_style(
        kmz_document,
        "electrical_boresight",
        bbggrr="a0a050",
        low_aa="70",
        low_width="2.0",
        high_aa="a0",
        high_width="3.5",
        outline="0",
    )

    add_polygon_style(
        kmz_document,
        "rcv_beam_footprint",
        bbggrr="ffaa55",
        low_aa="70",
        low_width="1.0",
        high_aa="a0",
        high_width="1.5",
    )

    add_polygon_style(
        kmz_document,
        "tx_beam_footprint",
        bbggrr="0000ff",
        low_aa="70",
        low_width="1.0",
        high_aa="a0",
        high_width="1.5",
    )

    add_polygon_style(
        kmz_document,
        "twoway_beam_footprint",
        bbggrr="885588",
        low_aa="70",
        low_width="1.0",
        high_aa="a0",
        high_width="1.5",
    )


def ray_ellipsoid_intersection(
    semiaxis_a, semiaxis_b, semiaxis_c, pos_x, pos_y, pos_z, dir_x, dir_y, dir_z
):
    """
    Compute the intersection of a Ray with an Ellipsoid

    Parameters
    ----------
    semiaxis_a, semiaxis_b, semiaxis_c: float
        Semiaxes of the ellipsoid
    pos_x, pos_y, pos_z: float
        Ray origin
    dir_x, dir_y, dir_z: float
        Ray direction

    Returns
    -------
    intersection_x, intersection_y, intersection_z: float
        Point of intersection
    """
    # Based on https://stephenhartzell.medium.com/satellite-line-of-sight-intersection-with-earth-d786b4a6a9b6
    # >>> pos_x, pos_y, pos_z = symbols('pos_x pos_y pos_z')
    # >>> dir_x, dir_y, dir_z = symbols('dir_x dir_y dir_z')
    # >>> semiaxis_a, semiaxis_b, semiaxis_c = symbols('semiaxis_a semiaxis_b semiaxis_c')
    # >>> distance = symbols('distance')

    # >>> solutions = solve((pos_x + distance*dir_x)**2/semiaxis_a**2 + (pos_y + distance*dir_y)**2/semiaxis_b**2 + (pos_z + distance*dir_z)**2/semiaxis_c**2 - 1, distance)
    # >>> print(solutions[0])

    # black formatted
    distance = (
        -dir_x * pos_x * semiaxis_b**2 * semiaxis_c**2
        - dir_y * pos_y * semiaxis_a**2 * semiaxis_c**2
        - dir_z * pos_z * semiaxis_a**2 * semiaxis_b**2
        - semiaxis_a
        * semiaxis_b
        * semiaxis_c
        * np.sqrt(
            -(dir_x**2) * pos_y**2 * semiaxis_c**2
            - dir_x**2 * pos_z**2 * semiaxis_b**2
            + dir_x**2 * semiaxis_b**2 * semiaxis_c**2
            + 2 * dir_x * dir_y * pos_x * pos_y * semiaxis_c**2
            + 2 * dir_x * dir_z * pos_x * pos_z * semiaxis_b**2
            - dir_y**2 * pos_x**2 * semiaxis_c**2
            - dir_y**2 * pos_z**2 * semiaxis_a**2
            + dir_y**2 * semiaxis_a**2 * semiaxis_c**2
            + 2 * dir_y * dir_z * pos_y * pos_z * semiaxis_a**2
            - dir_z**2 * pos_x**2 * semiaxis_b**2
            - dir_z**2 * pos_y**2 * semiaxis_a**2
            + dir_z**2 * semiaxis_a**2 * semiaxis_b**2
        )
    ) / (
        dir_x**2 * semiaxis_b**2 * semiaxis_c**2
        + dir_y**2 * semiaxis_a**2 * semiaxis_c**2
        + dir_z**2 * semiaxis_a**2 * semiaxis_b**2
    )

    if np.isnan(distance):
        raise ValueError("Ray does not intersect ellipsoid")

    if distance < 0:
        raise ValueError("Ray points away from ellipsoid")

    return (
        pos_x + distance * dir_x,
        pos_y + distance * dir_y,
        pos_z + distance * dir_z,
    )


def ray_intersect_earth(position, direction):
    """Intersect an ECEF ray with the earth

    Parameters
    ----------
    position: list-like
        Ray origin in ECEF
    direction: list-like
        Ray direction in ECEF

    Returns
    -------
    intersection_x, intersection_y, intersection_z: float
        Point of intersection in ECEF

    """
    point_ecf = ray_ellipsoid_intersection(
        WGS84_SEMIMAJOR,
        WGS84_SEMIMAJOR,
        WGS84_SEMIMINOR,
        position[0],
        position[1],
        position[2],
        direction[0],
        direction[1],
        direction[2],
    )
    return point_ecf


def add_los_polygon(kmz_doc, parent_placemark, apc_coords, ground_coords):
    """Add a polygon connecting a path above the earth to a path on the earth"""
    # complex 3d polygons don't always render nicely.  So, we'll manually triangluate it.
    mg = kmz_doc.add_multi_geometry(par=parent_placemark)
    # Highlight the starting point
    kmz_doc.add_line_string(
        coords=" ".join([apc_coords[0], ground_coords[0]]),
        par=mg,
        altitudeMode="absolute",
    )
    for idx in range(len(apc_coords) - 1):
        coords = [
            apc_coords[idx],
            ground_coords[idx],
            apc_coords[idx + 1],
            apc_coords[idx],
        ]
        kmz_doc.add_polygon(" ".join(coords), par=mg, altitudeMode="absolute")
        coords = [
            ground_coords[idx],
            ground_coords[idx + 1],
            apc_coords[idx + 1],
            ground_coords[idx],
        ]
        kmz_doc.add_polygon(" ".join(coords), par=mg, altitudeMode="absolute")


def make_beam_footprints(
    aiming_metadata,
    labelled_indices,
    array_gain_poly,
    element_gain_poly,
    contour_level=-3,
):
    """Attempt to make beam footprints for the labelled slowtimes"""

    def _find_central_contour(max_dc, eb_dcx, eb_dcy):
        Ns = 201
        dcs = np.linspace(-max_dc, max_dc, Ns)
        dcx, dcy = np.meshgrid(dcs, dcs, indexing="ij")
        gain_pattern = array_gain_poly(dcx, dcy) + element_gain_poly(dcx + eb_dcx, dcy + eb_dcy)
        contour_sets = plt.contour(dcx, dcy, gain_pattern, levels=[contour_level])
        plt.close()  # close the figure created by contour

        # We don't know the validity range of the gain polynomials and may have gone
        # outside, resulting in multiple contours

        # Only keep contours that form a closed shape
        try:
            paths = contour_sets.get_paths()
        except AttributeError:
            # matplotlib deprecated collections attribute in 3.8
            paths = contour_sets.collections[0].get_paths()

        polygons = [
            polygon
            for path in paths
            for polygon in path.to_polygons(closed_only=False)
            if np.array_equal(polygon[0], polygon[-1]) and len(polygon) > 1  # only consider closed polygons
        ]
        if polygons:
            # Keep contour closest to center
            return min(
                polygons, key=lambda vertices: np.linalg.norm(np.mean(vertices, axis=0))
            )
        return None


    result = {}
    for name, pvp_index in labelled_indices.items():
        try:
            eb_dcx = aiming_metadata["raw"]["eb_dcx"][pvp_index]
            eb_dcy = aiming_metadata["raw"]["eb_dcy"][pvp_index]
            uacx = aiming_metadata["raw"]["uacx"][pvp_index]
            uacy = aiming_metadata["raw"]["uacy"][pvp_index]
            time = aiming_metadata["raw"]["times"][pvp_index]
            apc_pos = aiming_metadata["raw"]["positions"][pvp_index]

            contour = _find_central_contour(1, eb_dcx, eb_dcy)
            for n in range(10):
                next_contour = _find_central_contour(8**(-n-1), eb_dcx, eb_dcy)
                if next_contour is None and contour is not None:
                    break
                if contour is not None and next_contour is not None:
                    cpoly = shg.Polygon(contour)
                    ncpoly = shg.Polygon(next_contour)
                    if np.isclose(cpoly.intersection(ncpoly).area, ncpoly.area, atol=0, rtol=0.05):
                        # close enough
                        contour = next_contour
                        break
                contour = next_contour

            if contour is None:
                raise ValueError("unable to find contour")

            delta_dcx = contour[:, 0]
            delta_dcy = contour[:, 1]

            contour_pointing = acf_to_ecef(
                delta_dcx + eb_dcx,
                delta_dcy + eb_dcy,
                uacx,
                uacy,
            )

            contour_earth_ecf = []
            for along in contour_pointing:
                contour_earth_ecf.append(ray_intersect_earth(apc_pos, along))
            result[name] = {"time": time, "contour": contour_earth_ecf}
        except Exception as exc:
            logger.warning(
                f"Exception while calculating {name} beam footprint of {aiming_metadata['antpat_id']}"
            )
            logger.warning(exc, exc_info=True)

    return result
