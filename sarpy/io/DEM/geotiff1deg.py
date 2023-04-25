"""
Classes and methods for parsing and using digital elevation models in GeoTIFF format.

This code makes the following assumptions.
    1. The GeoTIFF files tile the earth with one degree offsets in both latitude and longitude.
    2. There is one pixel of overlap between adjacent tiles.
    3. The south-west corner of each tile is at an integer (degrees) latitude and longitude.
    4. The latitude and longitude of south-west corner point is encoded in the GeoTIFF filename.
    5. The antimerdian is at W180 rather than at E180 so that valid longitude values are (-180 <= lon < 180) degrees.
    6. The GeoTIFF tag 34737 (GeoAsciiParamsTag) indicates the reference surface (EGM2008 or WGS84).
"""

import logging
import os.path
import warnings
from PIL import Image, TiffTags
from scipy.interpolate import RegularGridInterpolator

import numpy as np
from sarpy.io.DEM.DEM import DEMList
from sarpy.io.DEM.DEM import DEMInterpolator
from sarpy.io.DEM.geoid import GeoidHeight

logger = logging.getLogger(__name__)

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"

Image.MAX_IMAGE_PIXELS = None  # get rid of decompression bomb checking


class GeoTIFF1DegInterpolator(DEMInterpolator):
    """
     This class contains methods used to read DEM data from GeoTIFF files and interpolate the height values, as needed.

     Args
     ----
     root_dir: str | pathlib.Path
         The root directory of the GeoTIFF DEM data.
    filename_format: str | list[str]
        format string used to construct GeoTIFF filename from a Lat/Lon pair (see the GeoTIFF1DegList docstring).
     geoid_dir: str or pathlib.Path
        Directory containing the geoid files in PGM format.
     missing_error: bool (default: False)
         Optional flag indicating whether an exception will be raised when missing DEM data files are encountered.
         If True then a ValueError will be raised when a needed data file does not exist in the root_dir.
         If False then a DEM value of zero will be silently used when a needed data file does not exist in the root_dir.
     interp_method: str (default: "linear")
         Optional interpolation method. Any scipy.interpolate.RegularGridInterpolator method is valid here.

     """
    __slots__ = ('_root_dir', '_geod_file', '_interp_method', '_ref_surface',
                 '_geotiff_list_obj', "_bounding_box_cache")

    def __init__(self, root_dir,  filename_format, geoid_dir='', *, missing_error=False, interp_method="linear"):
        self._root_dir = str(root_dir)
        self._geoid_dir = str(geoid_dir)
        self._interp_method = interp_method
        self._ref_surface = "Unknown"
        self._geotiff_list_obj = GeoTIFF1DegList(root_dir, filename_format, missing_error=missing_error)
        self._bounding_box_cache = {}

        # get the geoid object - we prefer egm2008*.pgm files, but in reality, it makes very little difference.
        self._geoid_obj = None
        if self._geoid_dir and os.path.isdir(self._geoid_dir):
            search_files = ('egm2008-1.pgm', 'egm2008-2_5.pgm', 'egm2008-5.pgm',
                            'egm96-5.pgm', 'egm96-15.pgm', 'egm84-15.pgm', 'egm84-30.pgm')
            self._geoid_obj = GeoidHeight.from_directory(geoid_dir, search_files=search_files)

    @property
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, val):
        self._interp_method = str(val)

    def get_elevation_native(self, lat, lon, block_size=50000):
        """
        Get the elevation value relative to the reference surface specified in the GeooTIFF GeoAsciiParamsTag tag.

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
        if block_size is not None:
            warnings.warn("Block processing is not yet implemented.  Full size processing will be used.")

        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)

        if lat.shape != lon.shape:
            raise ValueError("The lat and lon arrays are not the same shape.")

        lat_lon_pairs = np.stack([lat.flatten(), lon.flatten()], axis=-1)
        unique_sw_corners = np.unique(np.floor(lat_lon_pairs), axis=0)

        # Get the list of filenames for the unique SW corners, if they exist.
        filename_info = []
        for sw_lat, sw_lon in unique_sw_corners:
            # Adding a fractional offset to the otherwise integer SW corner Lat/Lon values
            # will guarantee that no more than one filename will be found.
            files = self._geotiff_list_obj.find_dem_files(sw_lat + 0.1, sw_lon + 0.1)
            if files:
                filename_info.append({"filename": files[0], "sw_lat": sw_lat, "sw_lon": sw_lon})

        height = np.zeros(lat.size)
        for info in filename_info:
            filename = info["filename"]
            tile_sw_lat = info["sw_lat"]
            tile_sw_lon = info["sw_lon"]
            tile_ne_lat = tile_sw_lat + 1
            tile_ne_lon = tile_sw_lon + 1

            # Note: the dem_data must have dtype=np.float64 otherwise the interpolator
            # created by RegularGridInterpolator will raise a TypeError exception.
            with Image.open(filename) as img:
                tiff_tags = {TiffTags.TAGS[key]: val for key, val in img.tag.items()}
                dem_data = np.asarray(img, dtype=np.float64)

            geo_ascii_params_tag = tiff_tags.get('GeoAsciiParamsTag', ('',))[0]
            this_ref_surface = ('EGM84' if 'EGM84' in geo_ascii_params_tag else
                                'EGM96' if 'EGM96' in geo_ascii_params_tag else
                                'EGM2008' if 'EGM2008' in geo_ascii_params_tag else
                                'EGM2020' if 'EGM2020' in geo_ascii_params_tag else
                                'WGS84' if 'WGS84' in geo_ascii_params_tag else
                                'Unknown')
            if self._ref_surface == "Unknown":
                self._ref_surface = this_ref_surface
            elif self._ref_surface != this_ref_surface:
                raise ValueError(f'Reference surface missmatch in file {filename}')    # pragma: no cover

            tile_num_lats, tile_num_lons = dem_data.shape
            tile_lats = np.linspace(tile_ne_lat, tile_sw_lat, tile_num_lats)
            tile_lons = np.linspace(tile_sw_lon, tile_ne_lon, tile_num_lons)

            # Old versions of scipy.interpolate require that axis samples be in strictly ascending order.
            # Unfortunately, the tile_lats are in strictly descending order.  To get the interpolator to
            # work regardless of package version, we will negate tile_lats and the lat-part of lat_lon_pairs.
            neg_tile_lats = -tile_lats
            neg_lat_lon_pairs = [(-lat, lon) for lat, lon in lat_lon_pairs]

            interp = RegularGridInterpolator((neg_tile_lats, tile_lons), dem_data, method=self._interp_method,
                                             bounds_error=False, fill_value=np.nan)

            interp_height = interp(neg_lat_lon_pairs)
            mask = np.logical_not(np.isnan(interp_height))
            height[mask] = interp_height[mask]

        return height.reshape(lat.shape)

    def get_elevation_hae(self, lat, lon, block_size=50000):
        """
        Get the elevation value relative to the WGS84 ellipsoid.

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
        height_native = self.get_elevation_native(lat, lon, block_size=block_size)

        if self._ref_surface == "WGS84":
            return height_native
        elif self._ref_surface.startswith("EGM"):
            if self._geoid_obj is None:
                raise ValueError("The geoid_dir parameter was not defined so geoid calculations are disabled.")

            return height_native + self._geoid_obj.get(lat, lon, block_size=block_size)
        else:
            raise ValueError(f"The reference surface is {self._ref_surface}, which is not supported")

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
        height_native = self.get_elevation_native(lat, lon, block_size)

        if self._ref_surface.startswith("EGM"):
            return height_native
        elif self._ref_surface == "WGS84":
            if self._geoid_obj is None:
                raise ValueError("The geoid_dir parameter was not defined so geoid calculations are disabled.")

            return height_native - self._geoid_obj.get(lat, lon, block_size=block_size)
        else:
            raise ValueError(f"The reference surface is {self._ref_surface}, which is not supported.")

    def get_max_hae(self, lat_lon_box=None):
        """
        Get the maximum dem value with respect to HAE, which should be assumed **approximately** correct.

        Parameters
        ----------
        lat_lon_box : list | numpy.ndarray
            Any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """
        result = self.get_min_max_native(lat_lon_box)

        if result['ref_surface'] == 'ellipsoid':
            return result['max']['height']
        else:
            return self.get_elevation_hae(result['max']['lat'], result['max']['lon'])[0]

    def get_min_hae(self, lat_lon_box=None):
        """
        Get the minimum dem value with respect to HAE, which should be assumed **approximately** correct.

        Parameters
        ----------
        lat_lon_box : list | numpy.ndarray
            Any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """
        result = self.get_min_max_native(lat_lon_box)

        if result['ref_surface'] == 'ellipsoid':
            return result['min']['height']
        else:
            return self.get_elevation_hae(result['min']['lat'], result['min']['lon'])[0]

    def get_max_geoid(self, lat_lon_box=None):
        """
        Get the maximum dem value with respect to the geoid, which should be assumed **approximately** correct.

        Parameters
        ----------
        lat_lon_box : list | numpy.ndarray
            Any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """
        result = self.get_min_max_native(lat_lon_box)

        if result['ref_surface'] == 'geoid':
            return result['max']['height']
        else:
            return self.get_elevation_geoid(result['max']['lat'], result['max']['lon'])[0]

    def get_min_geoid(self, lat_lon_box=None):
        """
        Get the minimum dem value with respect to geoid, which should be assumed **approximately** correct.

        Parameters
        ----------
        lat_lon_box : list | numpy.ndarray
            Any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """
        result = self.get_min_max_native(lat_lon_box)

        if result['ref_surface'] == 'geoid':
            return result['min']['height']
        else:
            return self.get_elevation_geoid(result['min']['lat'], result['min']['lon'])[0]

    def get_min_max_native(self, lat_lon_box):
        """
        Get the minimum and maximum dem value with respect to the native reference surface of the DEM.

        Parameters
        ----------
        lat_lon_box : List | numpy.ndarray
            The bounding box to search `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        dict: {"box": lat_lon_box,
               "ref_surface": ref_surface,
               "min": {"lat": lat_deg, "lon": lon_deg, "height": height},
               "max": {"lat": lat_deg, "lon": lon_deg, "height": height}
              }
        """
        if self._bounding_box_cache.get("box", []) == lat_lon_box:
            # If we have already done this calculation, don't do it again.
            return self._bounding_box_cache

        box_lat_min, box_lat_max, box_lon_min, box_lon_max = lat_lon_box

        filename_info = []
        for sw_lat in np.arange(np.floor(box_lat_min), np.floor(box_lat_max)+1):
            for sw_lon in np.arange(np.floor(box_lon_min), np.floor(box_lon_max) + 1):
                files = self._geotiff_list_obj.find_dem_files(sw_lat + 0.1, sw_lon + 0.1)
                if files:
                    filename_info.append({"filename": files[0], "sw_lat": sw_lat, "sw_lon": sw_lon})

        # Initialize so that the global min and max occur at the same lat/lon and have a value of zero.
        # These values will be returned if the bounding box is completely outside the available DEM tiles.
        ref_surface = 'geoid'
        global_min_lat = box_lat_min
        global_max_lat = box_lat_min
        global_min_lon = box_lon_min
        global_max_lon = box_lon_min
        global_min = np.inf if filename_info else 0
        global_max = -np.inf if filename_info else 0

        for info in filename_info:
            filename = info["filename"]
            tile_sw_lat = info["sw_lat"]
            tile_sw_lon = info["sw_lon"]
            tile_ne_lat = tile_sw_lat + 1

            with Image.open(filename) as img:
                tiff_tags = {TiffTags.TAGS[key]: val for key, val in img.tag.items()}
                dem_data = np.asarray(img, dtype=np.float64)

            tile_num_lats = tiff_tags['ImageLength'][0]
            tile_num_lons = tiff_tags['ImageWidth'][0]
            ref_surface = 'geoid' if 'EGM' in tiff_tags.get('GeoAsciiParamsTag', ('',))[0] else 'ellipsoid'

            # Lat index is in descending order, so calculate the offset from the north edge (lowest index)
            lat_start_offset = max(0, tile_ne_lat - box_lat_max)
            lat_stop_offset = min(1, tile_ne_lat - box_lat_min)

            # Lon index is in ascending order, so calculate the offset from the west edge (lowest index)
            lon_start_offset = max(0, box_lon_min - tile_sw_lon)
            lon_stop_offset = min(1, box_lon_max - tile_sw_lon)

            # Lat index is in descending order, so start at the north edge (lowest index)
            row_start = int(np.ceil(lat_start_offset * (tile_num_lats - 1)))
            row_stop = int(np.floor(lat_stop_offset * (tile_num_lats - 1)))

            # Lon index is in ascending order, so start at the west edge (lowest index)
            col_start = int(np.ceil(lon_start_offset * (tile_num_lons - 1)))
            col_stop = int(np.floor(lon_stop_offset * (tile_num_lons - 1)))

            dem_slice = dem_data[row_start:row_stop+1, col_start:col_stop+1]
            max_index = np.unravel_index(np.argmax(dem_slice), shape=dem_slice.shape)
            min_index = np.unravel_index(np.argmin(dem_slice), shape=dem_slice.shape)

            if global_max < dem_slice[max_index]:
                global_max = dem_slice[max_index]
                global_max_lat = tile_ne_lat - lat_start_offset - max_index[0] / (tile_num_lats - 1)
                global_max_lon = tile_sw_lon + lon_start_offset + max_index[1] / (tile_num_lons - 1)

            if global_min > dem_slice[min_index]:
                global_min = dem_slice[min_index]
                global_min_lat = tile_ne_lat - lat_start_offset - min_index[0] / (tile_num_lats - 1)
                global_min_lon = tile_sw_lon + lon_start_offset + min_index[1] / (tile_num_lons - 1)

        self._bounding_box_cache = {"box": lat_lon_box,
                                    "ref_surface": ref_surface,
                                    "min": {"lat": global_min_lat, "lon": global_min_lon, "height": float(global_min)},
                                    "max": {"lat": global_max_lat, "lon": global_max_lon, "height": float(global_max)}
                                    }

        return self._bounding_box_cache


# ---------------------------------------------------------------------------------------------------------------------
# GeoTIFF1DegList
# ---------------------------------------------------------------------------------------------------------------------
class GeoTIFF1DegList(DEMList):
    """
    GeoTIFF subclass of sarpy.io.DEM.DEMList

    This class contains methods used to determine which GeoTIFF files are needed to cover
    a specified geodetic bounding box.

    Args
    ----
    root_dir: str or pathlib.Path
        The root directory of the GeoTIFF DEM data.
    filename_format: str | list[str]
        format string used to construct GeoTIFF filename from a Lat/Lon pair, as described in the Notes section.
    missing_error: bool (default: False)
        Optional flag indicating whether an exception will be raised when missing DEM data files are encountered.
        If True then a ValueError will be raised when a needed data file does not exist in root_dir.
        If False then a DEM value of zero will be silently used when a needed data file does not exist in root_dir.

    Notes
    -----
    The DEM files must have the SW corner Lat/Lon encoded in their filenames.  The filename_format argument contains a
    list of format strings that will be joined together by the os.path.join() to produce a path, relative to root_dir,
    to the desired GeoTIFF DEM file.  The following arguments will be provided to the format string.
        lat = int(numpy.floor(lat))
        lon = int(numpy.floor(lon))
        abslat = int(abs(numpy.floor(lat)))
        abslon = int(abs(numpy.floor(lon)))
        ns = 's' if lat < 0 else 'n'
        NS = 'S' if lat < 0 else 'N'
        ew = 'w' if lon < 0 else 'e'
        EW = 'W' if lon < 0 else 'E'
        ver = optional character string used to specify a version number

    When specifying an f-string argument, a width field and fill character should always be included.
    For example, "{ns:1s}", "{ew:1s}", "{ver:2s}", "{abslat:02d}", "{abslon:03d}".

    Examples:

        Some high resolution GeoTIFF DEM files have used the following filename format.
            ["tdt_{ns:1s}{abslat:02}{ew:1s}{abslon:03}_{ver:2s}", "DEM",
             "TDT_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_{ver:2s}_DEM.tif"]

        Some lower resolution GeoTIFF DEM files have used the following filename format.
            ["TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_V{ver:2s}_C", "DEM",
            "TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_DEM.tif"]

    """

    __slots__ = ('_root_dir', "_filename_format", '_missing_error')

    def __init__(self, root_dir, filename_format, *, missing_error=False):
        if not os.path.exists(str(root_dir)):
            raise ValueError(f"The top level directory ({str(root_dir)}) does not exist.")

        if isinstance(filename_format, str):
            filename_format = [filename_format]    # pragma: no cover

        self._root_dir = str(root_dir)
        self._filename_format = os.path.join(self._root_dir, *filename_format)
        self._missing_error = bool(missing_error)

    def filename_from_lat_lon(self, lat, lon, ver=None):
        """
        This method will return the filename of the GeoTIFF file that contains the specified
        latitude/longitude and version number.

        """
        pars = {
            "lat": int(np.floor(lat)),
            "lon": int(np.floor(lon)),
            "abslat": int(abs(np.floor(lat))),
            "abslon": int(abs(np.floor(lon))),
            "ns": 's' if lat < 0 else 'n',
            "ew": 'w' if lon < 0 else 'e',
            "NS": 'S' if lat < 0 else 'N',
            "EW": 'W' if lon < 0 else 'E',
            "ver": ver,
        }

        class SkipMissing(dict):
            def __missing__(self, key):
                return f'{{{key}}}'

        return self._filename_format.format_map(SkipMissing(**pars))

    def find_dem_files(self, lat, lon):
        """
        Return a list of filenames of GeoTIFF files that contain DEM data for the specified Lat/Lon point.
        Since DEM files overlap, there might be more than one file that contains the specified Lat/Lon point.

        Args
        ----
        lat: int | float
            The latitude in degrees (-90 <= lat <= 90)
        lon: int | float
            The longitude in degrees (-180 <= lon < 180)

        Returns
        -------
        filenames: list(str)
            A list of filenames of DEM data files, if they exists, otherwise []

        """
        msg = [] if -90.0 <= lat <= 90.0 else ["The latitude value must be between [-90, +90]"]
        msg += [] if -180.0 <= lon < 180.0 else ["The longitude value must be between [-180, +180)"]
        if msg:
            raise ValueError('\n'.join(msg))

        sw_lats = [89 if lat == 90 else np.floor(lat)]      # The latitude of the south-west corner in integer degrees
        sw_lons = [np.floor(lon)]                           # The longitude of the south-west corner in integer degrees

        if lat == np.floor(lat) and np.abs(lat) < 90:
            # lat is an integer, so it is in the overlap region of at least two files
            sw_lats.append(np.floor(lat)-1)

        if lon == np.floor(lon):
            # lon is an integer, so it is in the overlap region of at least two files.
            sw_lons.append(179 if lon == -180 else np.floor(lon)-1)

        filenames = []
        for sw_lat in sw_lats:
            for sw_lon in sw_lons:
                new_filenames = []
                for ver in ['01', '02']:
                    filename = self.filename_from_lat_lon(int(sw_lat), int(sw_lon), ver)

                    if os.path.isfile(filename):
                        new_filenames.append(filename)
                filenames.extend(new_filenames)

                if not new_filenames:
                    msg = f'Missing expected DEM file for tile with lower left lat/lon corner ({sw_lat}, {sw_lon})'
                    if self._missing_error:
                        raise ValueError(msg)
                    else:
                        logger.warning(
                            msg + '\n\tThis should result in the assumption that the altitude in\n\t'
                                  'that section is zero relative to the reference surface.')

        return filenames

    def get_file_list(self, lat_lon_box):
        """
        This will return the list of files associated with covering the `lat_lon_box` using a DEM.

        If the bounding box spans the antimeridian (180th meridian), then the maximum longitude
        will be less than the minimum longitude.

        Args
        ----
        lat_lon_box : numpy.ndarray | list | tuple
            The bounding box of the form `[lat min, lat max, lon min, lon max]` in degrees.

        Returns
        -------
        filenames: List[str]
            A list of filenames, without duplication, of the files needed to cover the bounding box.
        """
        filenames = []

        lat_min, lat_max, lon_min, lon_max = lat_lon_box

        msg = ["The minimum latitude value must be between [-90, +90]"] if not (-90.0 <= lat_min <= 90.0) else []
        msg += ["The maximum latitude value must be between [-90, +90]"] if not (-90.0 <= lat_max <= 90.0) else []
        msg += ["The minimum longitude value must be between [-180, +180)"] if not (-180.0 <= lon_min < 180.0) else []
        msg += ["The maximum longitude value must be between [-180, +180)"] if not (-180.0 <= lon_max < 180.0) else []
        if msg:
            raise ValueError('\n'.join(msg))

        if lon_max < lon_min:
            lon_max += 360

        for lat_inc in np.arange(np.ceil(lat_max) - np.floor(lat_min)):
            lat = lat_min + lat_inc

            for lon_inc in np.arange(np.ceil(lon_max) - np.floor(lon_min)):
                lon = (lon_min + lon_inc + 180) % 360 - 180

                new_filenames = self.find_dem_files(lat, lon)
                new_unique_filenames = [file for file in new_filenames if file not in filenames]
                filenames.extend(new_unique_filenames)

        return filenames
