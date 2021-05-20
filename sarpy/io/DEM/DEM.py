"""
Establish base expected functionality for digital elevation model handling.
"""

import numpy
from typing import List

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class DEMInterpolator(object):
    """
    Abstract DEM class presenting base required functionality.
    """

    def get_elevation_hae(self, lat, lon, block_size=50000):
        """
        Get the elevation value relative to the WGS-84 ellipsoid.

        Parameters
        ----------
        lat : numpy.ndarray|list|tuple|int|float
        lon : numpy.ndarray|list|tuple|int|float
        block_size : int|None
            If `None`, then the entire calculation will proceed as a single block.
            Otherwise, block processing using blocks of the given size will be used.
            The minimum value used for this is 50,000, and any smaller value will be
            replaced with 50,000. Default is 50,000.

        Returns
        -------
        numpy.ndarray
            the elevation relative to the WGS-84 ellipsoid.
        """

        raise NotImplementedError

    def get_elevation_geoid(self, lat, lon, block_size=50000):
        """
        Get the elevation value relative to the geoid.

        Parameters
        ----------
        lat : numpy.ndarray|list|tuple|int|float
        lon : numpy.ndarray|list|tuple|int|float
        block_size : int|None
            If `None`, then the entire calculation will proceed as a single block.
            Otherwise, block processing using blocks of the given size will be used.
            The minimum value used for this is 50,000, and any smaller value will be
            replaced with 50,000. Default is 50,000.

        Returns
        -------
        numpy.ndarray
            the elevation relative to the geoid
        """

        raise NotImplementedError

    def get_max_hae(self, lat_lon_box=None):
        """
        Get the maximum dem value with respect to HAE, which should be assumed
        **approximately** correct. This may possibly be with respect to some
        Area of Interest.

        Parameters
        ----------
        lat_lon_box : None|numpy.ndarray
            None or any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """

        raise NotImplementedError

    def get_min_hae(self, lat_lon_box=None):
        """
        Get the minimum dem value with respct to HAE, which should be assumed
        **approximately** correct. This may possibly be with respect to some
        Area of Interest.

        Parameters
        ----------
        lat_lon_box : None|numpy.ndarray
            None or any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """

        raise NotImplementedError

    def get_max_geoid(self, lat_lon_box=None):
        """
        Get the maximum dem value with respect to the geoid, which should be assumed
        **approximately** correct. This may possibly be with respect to some
        Area of Interest.

        Parameters
        ----------
        lat_lon_box : None|numpy.ndarray
            None or any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """

        raise NotImplementedError

    def get_min_geoid(self, lat_lon_box=None):
        """
        Get the minimum dem value with respect to geoid, whihc should be assumed
        **approximately** correct. This may possibly be with respect to some
        Area of Interest.

        Parameters
        ----------
        lat_lon_box : None|numpy.ndarray
            None or any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """

        raise NotImplementedError


class DEMList(object):
    """
    Abstract class for creating a searchable list of applicable DEM files of a
    given type.
    """

    def get_file_list(self, lat_lon_box):
        """
        This will return the list of files associated with covering the
        `lat_lon_box` using a DEM. Extraneous files (i.e. with region not overlapping
        the provided box) should NOT be returned, and files should be returned in
        order of preference.

        .. Note: It should be considered the user's responsibility to ensure
            that necessary DEM files will be found by this methodology, and
            regions lacking DEM file(s) should be assumed to have elevation
            at WGS-84 mean sea level.

        Parameters
        ----------
        lat_lon_box : numpy.ndarray|list|tuple
            The bounding box of the form `[lat min, lat max, lon min, lon max]`.

        Returns
        -------
        List[str]
        """

        raise NotImplementedError
