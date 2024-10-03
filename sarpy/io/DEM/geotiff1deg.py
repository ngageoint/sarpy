"""
Classes and methods for parsing and using digital elevation models (DEM) in GeoTIFF format.

This code makes the following assumptions.
    1. The GeoTIFF files tile the earth with one degree offsets in both latitude and longitude.
    2. There is one pixel of overlap between adjacent tiles.
    3. The south-west corner of each tile is at an integer (degrees) latitude and longitude.
    4. The latitude and longitude of south-west corner points is encoded in the GeoTIFF filename.
    5. The anti-meridian is at W180 rather than at E180 so that valid longitude values are (-180 <= lon < 180) degrees.
"""
import glob
import logging
import pathlib
import warnings

import numpy as np
try:
    from PIL import Image
    from PIL import TiffTags
    Image.MAX_IMAGE_PIXELS = None  # get rid of decompression bomb checking
except ImportError:
    Image = None
    TiffTags = None

from scipy.interpolate import RegularGridInterpolator

from sarpy.io.DEM.DEM import DEMList
from sarpy.io.DEM.DEM import DEMInterpolator
from sarpy.io.DEM.geoid import GeoidHeight

logger = logging.getLogger(__name__)

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"



class GeoTIFF1DegReader:
    """Class to read in a GeoTIFF file, if necessary, and cache the data."""

    def __init__(self, filename):
        self._filename = filename
        self._dem_data = None
        self._tiff_tags = None

    @property
    def filename(self):
        return self._filename

    @property
    def dem_data(self):
        if self._dem_data is None:
            self._read()  # pragma no cover
        return self._dem_data

    @property
    def tiff_tags(self):
        if self._tiff_tags is None:
            self._read()  # pragma no cover
        return self._tiff_tags

    def _read(self):
        # Note: the dem_data must have dtype=np.float64 otherwise the interpolator
        # created by RegularGridInterpolator will raise a TypeError exception.
        if Image is None or TiffTags is None:
            raise ImportError("Reading GeoTIFF DEM requires the PIL library")

        with Image.open(self._filename) as img:
            self._tiff_tags = {TiffTags.TAGS[key]: val for key, val in img.tag.items()}
            self._dem_data = np.asarray(img, dtype=np.float64)


class GeoTIFF1DegInterpolator(DEMInterpolator):
    """
    This class contains methods used to read DEM data from GeoTIFF files and interpolate the height values, as needed.

    Args
    ----
    dem_filename_pattern : str
        This is a format string that provides a glob pattern that will uniquely specify a DEM file from
        the Lat/Lon of the SW corner of the DEM tile.  See the GeoTIFF1DegList docstring for more details.
    ref_surface: str (default: "EGM2008")
        A case-insensitive string specifying the DEM reference surface. (eg., "WGS84" | "EGM2008" | "EGM96" | "EGM84")
    geoid_path: str | pathlib.Path | None (default: None)
        Optional filename of a specific Geoid file or a directory containing geoid files to choose from.
        If a directory is specified, then one or more of the following geoid files (in order of preference)
        will be chosen from this directory.

            'egm2008-1.pgm', 'egm2008-2_5.pgm', 'egm2008-5.pgm',
            'egm96-5.pgm', 'egm96-15.pgm', 'egm84-15.pgm', 'egm84-30.pgm'
    missing_error: bool (default: False)
        Optional flag indicating whether an exception will be raised when missing DEM data files are encountered.
        If True then a ValueError will be raised when a needed DEM data file can not be found.
        If False then a DEM value of zero will be used when a needed DEM data file is not found.
    interp_method: str (default: 'linear')
        Optional interpolation method. Any scipy.interpolate.RegularGridInterpolator method is valid here.
    max_readers: init (default: 4)
        Optional maximum number of DEM file readers.  A DEM file reader will read a DEM file and cache the results.
        DEM file readers can use a lot of memory (~8 bytes x number-of-DEM-samples), but will make processing faster.

     """
    __slots__ = ('_geoid_path', '_interp_method', '_ref_surface', '_geotiff_list_obj',
                 '_bounding_box_cache', '_max_readers', '_readers')

    def __init__(self, dem_filename_pattern, ref_surface='EGM2008', geoid_path=None, *,
                 missing_error=False, interp_method="linear", max_readers=4):
        self._geoid_path = pathlib.Path(geoid_path) if geoid_path else None
        self._interp_method = str(interp_method)
        self._ref_surface = str(ref_surface).upper()
        self._geotiff_list_obj = GeoTIFF1DegList(dem_filename_pattern, missing_error=missing_error)
        self._bounding_box_cache = {}
        self._max_readers = max(1, int(max_readers))
        self._readers = []

        # get the geoid object - we prefer egm2008*.pgm files, but in reality, it makes very little difference.
        if self._geoid_path and self._geoid_path.is_file():
            self._geoid_obj = GeoidHeight(str(self._geoid_path))
        elif self._geoid_path and self._geoid_path.is_dir():
            search_files = ('egm2008-1.pgm', 'egm2008-2_5.pgm', 'egm2008-5.pgm',
                            'egm96-5.pgm', 'egm96-15.pgm', 'egm84-15.pgm', 'egm84-30.pgm')
            self._geoid_obj = GeoidHeight.from_directory(str(self._geoid_path), search_files=search_files)
        else:
            self._geoid_obj = None

    @property
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, val):
        self._interp_method = str(val)

    def _read_dem_file(self, filename):
        """
        Get the DEM values and TIFF tags from the reader cache, if possible,
        otherwise create a new reader object in the reader cache and
        return its DEM data and TIFF tags.
        """
        for rdr in self._readers:
            if filename == rdr.filename:
                reader = rdr
                break
        else:
            if len(self._readers) >= self._max_readers:
                self._readers.pop(0)

            reader = GeoTIFF1DegReader(filename)
            self._readers.append(reader)

        return reader.tiff_tags, reader.dem_data

    def get_elevation_native(self, lat, lon, block_size=None):
        """
        Get the elevation value relative to the DEM file's reference surface.

        Parameters
        ----------
        lat : numpy.ndarray | list | tuple | int | float
        lon : numpy.ndarray | list | tuple | int | float
        block_size : int | None (default: None)
            Block processing is not supported; this argument is present to maintain a common interface with
            the DEMInterpolator parent class.  A value other than None will result in a warning.

        Returns
        -------
        numpy.ndarray
            The elevation relative to the reference surface of the DEM.
        """
        if block_size is not None:
            warnings.warn("Block processing is not implemented.  Full size processing will be used.")  # pragma nocover

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

            tiff_tags, dem_data = self._read_dem_file(filename)

            gpars = tiff_tags.get('GeoAsciiParamsTag', ('',))[0].upper()
            implied_ref_surface = ('EGM84' if any([p in gpars for p in ['EGM84', 'EGM 84', 'EGM-84']]) else
                                   'EGM96' if any([p in gpars for p in ['EGM96', 'EGM 96', 'EGM-96']]) else
                                   'EGM2008' if any([p in gpars for p in ['EGM2008', 'EGM 2008', 'EGM-2008']]) else
                                   'EGM2020' if any([p in gpars for p in ['EGM2020', 'EGM 2020', 'EGM-2020']]) else
                                   'WGS84' if any([p in gpars for p in ['WGS84', 'WGS 84', 'WGS-84']]) else
                                   'Unknown')
            if ((self._ref_surface.startswith('EGM') and implied_ref_surface.startswith('WGS')) or
                    (self._ref_surface.startswith('WGS') and implied_ref_surface.startswith('EGM'))):
                msg = (f"{filename}\n"
                       f"The GeoAsciiParamsTag tag implies that the reference surface is {implied_ref_surface},\n"
                       f"but the explicit reference surface was defined to be {self._ref_surface}.\n"
                       f"This might cause the elevation values to be calculated incorrectly.\n")
                logger.warning(msg)

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

    def get_elevation_hae(self, lat, lon, block_size=None):
        """
        Get the elevation value relative to the WGS84 ellipsoid.

        Parameters
        ----------
        lat : numpy.ndarray | list | tuple | int | float
        lon : numpy.ndarray | list | tuple | int | float
        block_size : int | None (default: None)
            Block processing is not supported; this argument is present to maintain a common interface with
            the DEMInterpolator parent class.  A value other than None will result in a warning.

        Returns
        -------
        numpy.ndarray
            The elevation relative to the ellipsoid
        """
        height_native = self.get_elevation_native(lat, lon, block_size=block_size)

        if self._ref_surface.startswith('WGS'):
            return height_native
        elif self._ref_surface.startswith("EGM"):
            if self._geoid_obj is None:
                raise ValueError("The geoid_dir parameter was not defined so geoid calculations are disabled.")

            return height_native + self._geoid_obj.get(lat, lon, block_size=block_size)
        else:
            raise ValueError(f"The reference surface is {self._ref_surface}, which is not supported")

    def get_elevation_geoid(self, lat, lon, block_size=None):
        """
        Get the elevation value relative to the geoid.

        Parameters
        ----------
        lat : numpy.ndarray | list | tuple | int | float
        lon : numpy.ndarray | list | tuple | int | float
        block_size : int | None (default: None)
            Block processing is not supported; this argument is present to maintain a common interface with
            the DEMInterpolator parent class.  A value other than None will result in a warning.

        Returns
        -------
        numpy.ndarray
            the elevation relative to the geoid
        """
        height_native = self.get_elevation_native(lat, lon, block_size)

        if self._ref_surface.startswith("EGM"):
            return height_native
        elif self._ref_surface.startswith('WGS'):
            if self._geoid_obj is None:
                raise ValueError("The geoid_dir parameter was not defined so geoid calculations are disabled.")

            return height_native - self._geoid_obj.get(lat, lon, block_size=block_size)
        else:
            raise ValueError(f"The reference surface is {self._ref_surface}, which is not supported.")

    def get_max_hae(self, lat_lon_box=None):
        """
        Get the maximum dem value with respect to the ellipsoid, which should be assumed **approximately** correct.

        Parameters
        ----------
        lat_lon_box : list | numpy.ndarray
            Any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """
        result = self.get_min_max_native(lat_lon_box)

        if self._ref_surface.startswith('WGS'):
            return result['max']['height']
        else:
            return self.get_elevation_hae(result['max']['lat'], result['max']['lon'])[0]

    def get_min_hae(self, lat_lon_box=None):
        """
        Get the minimum dem value with respect to the ellipsoid, which should be assumed **approximately** correct.

        Parameters
        ----------
        lat_lon_box : list | numpy.ndarray
            Any area of interest of the form `[lat min lat max, lon min, lon max]`.

        Returns
        -------
        float
        """
        result = self.get_min_max_native(lat_lon_box)

        if self._ref_surface.startswith('WGS'):
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

        if self._ref_surface.startswith('EGM'):
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

        if self._ref_surface.startswith('EGM'):
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
        dict
            A dictionary describing the results of the search::

                {"box": lat_lon_box,
                 "min": {"lat": lat_deg, "lon": lon_deg, "height": height},
                 "max": {"lat": lat_deg, "lon": lon_deg, "height": height}}

        """
        if np.array_equal(self._bounding_box_cache.get("box", []), lat_lon_box):
            # If we have already done this calculation then don't do it again.
            return self._bounding_box_cache

        box_lat_min, box_lat_max, box_lon_min, box_lon_max = lat_lon_box

        filename_info = []
        for sw_lat in np.arange(np.floor(box_lat_min), np.floor(box_lat_max) + 1):
            for sw_lon in np.arange(np.floor(box_lon_min), np.floor(box_lon_max) + 1):
                files = self._geotiff_list_obj.find_dem_files(sw_lat + 0.1, sw_lon + 0.1)
                if files:
                    filename_info.append({"filename": files[0], "sw_lat": sw_lat, "sw_lon": sw_lon})

        # Initialize so that the global min and max occur at the same lat/lon and have a value of zero.
        # These values will be returned if the bounding box is completely outside the available DEM tiles.
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

            tiff_tags, dem_data = self._read_dem_file(filename)

            tile_num_lats = tiff_tags['ImageLength'][0]
            tile_num_lons = tiff_tags['ImageWidth'][0]

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

            dem_slice = dem_data[row_start:row_stop + 1, col_start:col_stop + 1]
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
    dem_filename_pattern : str
        This is a format string that specifies the glob pattern that will uniquely specify a DEM file from
        the Lat/Lon of the SW corner of the DEM tile.  See the note below for more details.
    missing_error: bool (default: False)
        Optional flag indicating whether an exception will be raised when missing DEM data files are encountered.
        If True then a ValueError will be raised when a needed DEM data file can not be found.
        If False then a DEM value of zero will be used when a needed DEM data file is not found.

    Notes
    -----
    The DEM files must have the SW corner Lat/Lon encoded in their filenames.
    The dem_filename_pattern argument contains a format string that, when populated,
    will create a glob pattern that will specify the desired DEM file.  The following
    arguments are provided to the format string::

        lat = int(numpy.floor(lat))
        lon = int(numpy.floor(lon))
        abslat = int(abs(numpy.floor(lat)))
        abslon = int(abs(numpy.floor(lon)))
        ns = 's' if lat < 0 else 'n'
        NS = 'S' if lat < 0 else 'N'
        ew = 'w' if lon < 0 else 'e'
        EW = 'W' if lon < 0 else 'E'

    An example (with Linux file separators):
        "/dem_root/tdt_{ns}{abslat:02}{ew}{abslon:03}_*/DEM/TDT_{NS}{abslat:02}{EW}{abslon:03}_*_DEM.tif"

    will match filenames like:
        "/dem_root/tdt_n45e013_02/DEM/TDT_N45E013_02_DEM.tif"
        "/dem_root/tdt_s09w140_01/DEM/TDT_S09W140_01_DEM.tif"

    """

    __slots__ = ('_dem_filename_pattern', '_missing_error', '_pattern_to_filename')

    def __init__(self, dem_filename_pattern, missing_error=False):
        self._dem_filename_pattern = str(dem_filename_pattern)
        self._missing_error = bool(missing_error)
        self._pattern_to_filename = {}

    @staticmethod
    def filename_from_lat_lon(lat, lon, pattern):
        """
        This method will return the filename glob of the GeoTIFF file that contains the specified latitude/longitude.

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
        }

        class SkipMissing(dict):
            def __missing__(self, key):
                return f'{{{key}}}'

        return pattern.format_map(SkipMissing(**pars))

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

        sw_lats = [89 if lat == 90 else np.floor(lat)]  # The latitude of the south-west corner in integer degrees
        sw_lons = [np.floor(lon)]  # The longitude of the south-west corner in integer degrees

        if lat == np.floor(lat) and np.abs(lat) < 90:
            # lat is an integer, so it is in the overlap region of at least two files
            sw_lats.append(np.floor(lat) - 1)

        if lon == np.floor(lon):
            # lon is an integer, so it is in the overlap region of at least two files.
            sw_lons.append(179 if lon == -180 else np.floor(lon) - 1)

        filenames = []
        for sw_lat in sw_lats:
            for sw_lon in sw_lons:
                glob_pattern = self.filename_from_lat_lon(int(sw_lat), int(sw_lon), self._dem_filename_pattern)
                if glob_pattern not in self._pattern_to_filename:
                    for filename in glob.glob(glob_pattern):
                        if pathlib.Path(filename).is_file():
                            # The glob should not return more than one filename,
                            # but if it does then keep only the first.
                            self._pattern_to_filename[glob_pattern] = filename
                            break
                    else:
                        self._pattern_to_filename[glob_pattern] = None
                        msg = f'Missing expected DEM file for tile with lower left lat/lon corner ({sw_lat}, {sw_lon})'
                        if self._missing_error:
                            raise ValueError(msg)
                        else:
                            logger.warning(
                                msg + '\n\tThis should result in the assumption that the altitude in\n\t'
                                    'that section is zero relative to the reference surface.')
                if self._pattern_to_filename[glob_pattern] is not None:
                    filenames.append(self._pattern_to_filename[glob_pattern])

        return filenames

    def get_file_list(self, lat_lon_box):
        """
        This will return the list of files associated with covering the `lat_lon_box` using a DEM.

        If the bounding box spans the anti-meridian (180th meridian), then the maximum longitude
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
