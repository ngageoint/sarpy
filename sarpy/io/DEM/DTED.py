# -*- coding: utf-8 -*-
"""
Classes and methods for parsing and using digital elevation models in DTED format.
"""

import logging
import os
import struct

import numpy

from sarpy.compliance import integer_types, string_types
from sarpy.io.DEM.DEM import DEMList, DEMInterpolator
from sarpy.io.DEM.utils import argument_validation
from sarpy.io.DEM.geoid import GeoidHeight

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#######
# module variables
_SUPPORTED_DTED_FILE_TYPES = {
    'DTED1': {'fext': '.dtd'},
    'DTED2': {'fext': '.dtd'},
    'SRTM1': {'fext': '.dt1'},
    'SRTM2': {'fext': '.dt2'},
    'SRTM2F': {'fext': '.dt2'}}


def get_default_prioritization():
    """
    Gets the default prioritization of the DTED types.

    Returns
    -------
    List[str]
    """

    # TODO: what should this actually be?
    return 'DTED2', 'DTED1', 'SRTM2F', 'SRTM2', 'SRTM1'


def get_lat_lon_box(lats, lons):
    """
    Gets the lat/lon bounding box, as appropriate.

    Parameters
    ----------
    lats : numpy.ndarray|list|tuple|float|int
    lons : numpy.ndarray|list|tuple|float|int

    Returns
    -------
    numpy.ndarray
    """

    def get_min_max(inp, lon=False):
        if isinstance(inp, (int, float, numpy.number)):
            return inp, inp
        else:
            min_val, max_val = numpy.min(inp), numpy.max(inp)
            if not lon:
                return numpy.min(inp), numpy.max(inp)
            # check for 180/-180 crossing
            if not (min_val < -90 and max_val > 90):
                return min_val, max_val
            inp = numpy.array(inp).flatten()
            min_val = numpy.min(inp[inp >= 0])
            max_val = numpy.max(inp[inp <= 0])
            return min_val, max_val

    out = numpy.zeros((4, ), dtype='float64')
    out[:2] = get_min_max(lats)
    out[2:] = get_min_max(lons, lon=True)
    return out


class DTEDList(DEMList):
    """
    The DEM directory structure is assumed to look like the following:

        * For DTED<1,2>: `<root_dir>/dted/<1, 2>/<lon_string>/<lat_string>.dtd`

        * For SRTM<1,2,2F>: `<root_dir>/srtm/<1, 2, 2f>/<lon_string>/<lat_string>.dt<1,2,2>`

    Here `<lat_string>` corresponds to a string of the form `X##`, where
    `X` is one of 'N' or 'S', and `##` is the zero-padded formatted string for
    the integer value `floor(lat)`.

    Similarly, `<lon_string>` corresponds to a string of the form `Y##`, where
    `Y` is one of 'E' or 'W', and `###` is the zero-padded formatted string for
    the integer value `floor(lon)`.

    <lon_string>, <lat_string> corresponds to the origin in the lower left corner
    of the DEM tile.
    """

    __slots__ = ('_root_dir', )

    def __init__(self, root_directory):
        """

        Parameters
        ----------
        root_directory : str
        """

        self._root_dir = root_directory

    @property
    def root_dir(self):
        """
        str: the root directory
        """

        return self._root_dir

    def _get_directory_stem(self, dem_type):
        if dem_type.startswith('DTED'):
            return os.path.join(self._root_dir, dem_type[:4].lower(), dem_type[-1])
        elif dem_type.startswith('SRTM'):
            return os.path.join(self._root_dir, dem_type[:4].lower(), dem_type[4:].lower())
        else:
            raise ValueError('Unhandled dem_type {}'.format(dem_type))


    def _get_file_list(self, lat_lon_list, dem_type):
        """
        Helper method for getting the file list for a specified type.

        Parameters
        ----------
        lat_lon_list : list
        dem_type : str

        Returns
        -------
        List[str]
        """

        def get_box(la, lo):
            x = 'n' if la >= 0 else 's'
            y = 'e' if lo >= 0 else 'w'
            return '{0:s}{1:03d}'.format(y, abs(lo)), '{0:s}{1:02d}'.format(x, abs(la))

        # get the directory search stem
        dstem = self._get_directory_stem(dem_type)
        if not os.path.isdir(dstem):
            return  # nothing to be done

        # get file extension
        fext = _SUPPORTED_DTED_FILE_TYPES[dem_type]['fext']

        for entry in lat_lon_list:
            if entry[2] is not None:
                # we already found the file
                continue
            lonstr, latstr = get_box(entry[0], entry[1])
            fil = os.path.join(dstem, lonstr, latstr + fext)
            if os.path.isfile(fil):
                entry[2] = fil

    def get_file_list(self, lat_lon_box, dem_type=None):
        """
        Get the file list required for the given coordinates.

        Parameters
        ----------
        lat_lon_box : numpy.ndarray|list|tuple
            The bounding box of the form `[lat min, lat max, lon min, lon max]`.
        dem_type : None|str|List[str]
            The prioritized list of dem types to check. If `None`, then
            :func:`get_default_prioritization` is used. Each entry must be one
            of ("DTED1", "DTED2", "SRTM1", "SRTM2", "SRTM2F")

        Returns
        -------
        List[str]
        """

        # let's construct the list of lats that we must match
        lat_start = int(numpy.floor(lat_lon_box[0]))
        lat_end = int(numpy.ceil(lat_lon_box[1]))
        if (lat_start > lat_end) or (lat_start < -90) or (lat_start > 90):
            raise ValueError('Got malformed latitude in bounding box {}'.format(lat_lon_box))
        if lat_start == lat_end:
            lat_list = [lat_start, ]
        else:
            lat_list = list(range(lat_start, lat_end, 1))

        # let's construct the list of lons that we must match.
        lon_start = int(numpy.floor(lat_lon_box[2]))
        lon_end = int(numpy.ceil(lat_lon_box[3]))
        if (lon_start < -180) or (lon_end < -180) or (lon_start > 180) or (lon_end > 180):
            raise ValueError('Got malformed longitude in bounding box {}'.format(lat_lon_box))
        if lon_start > lon_end:
            if not (lon_end < 0 < lon_start):
                # this is assumed to NOT be a 180/-180 boundary crossing
                raise ValueError(
                    'We have minimum longitude greater than maximum longitude {}'.format(lat_lon_box))
            else:
                # we have a 180/-180 boundary crossing
                lon_list = list(range(lon_start, 180, 1)) + list(range(-180, lon_end, 1))
        else:
            if lon_start == lon_end:
                lon_list = [lon_start, ]
            else:
                lon_list = list(range(lon_start, lon_end, 1))

        # construct our workspace
        lat_lon_list = []
        for corner_lat in lat_list:
            for corner_lon in lon_list:
                lat_lon_list.append([corner_lat, corner_lon, None])

        # validate the dem types list
        if dem_type is None:
            dem_type = get_default_prioritization()
        elif isinstance(dem_type, string_types):
            dem_type = [dem_type, ]
        # loop over the prioritized list of types and check
        for entry in dem_type:
            if not isinstance(entry, string_types):
                raise TypeError(
                    'Got entry {} of dem_type, this is required to be of string type'.format(entry))
            # validate dem_type options
            this_entry = entry.upper()
            if this_entry not in _SUPPORTED_DTED_FILE_TYPES:
                raise ValueError(
                    'Got dem_type {}, but it must be one of the supported '
                    'types {}'.format(entry, list(_SUPPORTED_DTED_FILE_TYPES.keys())))
            self._get_file_list(lat_lon_list, this_entry)  # NB: this modifies lat_lon_list in place

        # extract files and warn about missing entries
        files = []
        missing_boxes = []
        for entry in lat_lon_list:
            if entry[2] is not None:
                files.append(entry[2])
            else:
                missing_boxes.append('({}, {})'.format(entry[0], entry[1]))

        if len(missing_boxes) > 0:
            logging.warning(
                'Missing expected DEM files for squares with lower left lat/lon corner {}. '
                'This Should result in the assumption that the altitude in that section is '
                'given by Mean Sea Level.'.format(missing_boxes))
        return files


class DTEDReader(object):
    """
    Reader/interpreter for DTED files, generally expected to be a helper class.
    As such, some implementation choices have been made for computational efficiency,
    and not user convenience.
    """

    __slots__ = ('_file_name', '_origin', '_spacing', '_bounding_box', '_shape', '_mem_map')

    def __init__(self, file_name):
        self._file_name = file_name

        with open(self._file_name, 'rb') as fi:
            # NB: DTED is always big-endian
            # the first 80 characters are header
            # characters 80:728 are data set identification record
            # characters 728:3428 are accuracy record
            # the remainder is data records, but DTED is not quite a raster
            header = struct.unpack('>80s', fi.read(80))[0].decode('utf-8')

        if header[:3] != 'UHL':
            raise IOError('File {} does not appear to be a DTED file.'.format(self._file_name))

        lon = float(header[4:7]) + float(header[7:9])/60. + float(header[9:11])/3600.
        lon = -lon if header[11] == 'W' else lon
        lat = float(header[12:15]) + float(header[15:17])/60. + float(header[17:19])/3600.
        lat = -lat if header[19] == 'S' else lat

        self._origin = numpy.array([lon, lat], dtype=numpy.float64)
        self._spacing = numpy.array([float(header[20:24]), float(header[24:28])], dtype=numpy.float64)/36000.
        self._shape = numpy.array([int(header[47:51]), int(header[51:55])], dtype=numpy.int64)
        self._bounding_box = numpy.zeros((4, ), dtype=numpy.float64)
        self._bounding_box[0::2] = self._origin
        self._bounding_box[1::2] = self._origin + self._spacing*(self._shape - 1)

        # starting at 3428, the rest of the file is data records, but not quite a raster
        #   each "row" is a data record with 8 extra bytes at the beginning,
        #   and 4 extra (checksum) at the end - look to MIL-PRF-89020B for an explanation
        # To enable memory map usage, we will spoof it as a raster and adjust column indices
        shp = (int(self._shape[0]), int(self._shape[1]) + 6)
        self._mem_map = numpy.memmap(self._file_name,
                                     dtype=numpy.dtype('>i2'),
                                     mode='r',
                                     offset=3428,
                                     shape=shp)
    @property
    def origin(self):
        """
        numpy.ndarray: The origin of this DTED, of the form `[longitude, latitude]`.
        """

        return numpy.copy(self._origin)

    @property
    def bounding_box(self):
        """
        numpy.ndarray: The bounding box of the form
        `[longitude min, longitude max, latitude min, latitude max]`.
        """

        return numpy.copy(self._bounding_box)

    def __getitem__(self, item):
        def new_col_int(val, begin):
            if val is None:
                if begin:
                    return 4
                else:
                    return -2
            return val + 4 if val >= 0 else val - 2

        # we need to manipulate in the second dimension
        if isinstance(item, tuple):
            if len(item) > 2:
                raise ValueError('Cannot slice on more than 2 dimensions')
            it = item[1]
            if isinstance(it, integer_types):
                it1 = new_col_int(it, True)
            elif isinstance(it, slice):
                start = new_col_int(it.start, True)
                stop = new_col_int(it.stop, False)
                it1 = slice(start, stop, it.step)
            elif isinstance(item[1], numpy.ndarray):
                it1 = numpy.copy(item[1])
                it1[it1 >= 0] += 4
                it1[it1 < 0] -= 2
            else:
                raise ValueError('Cannot slice using {}'.format(type(item[1])))
            data = self._mem_map.__getitem__((item[0], it1))
        else:
            data = self._mem_map[item, 4:-2]
        return self._repair_values(data)

    @staticmethod
    def _repair_values(elevations):
        """
        This is a helper method for repairing the weird entries in a DTED.
        The array is modified in place.

        Parameters
        ----------
        elevations : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        elevations = numpy.copy(elevations)
        # BASED ON MIL-PRF-89020B SECTION 3.11.1, 3.11.2
        # There are some byte-swapping details that are poorly explained.
        # The following steps appear to correct for the "complemented" values.
        # Find negative voids and repair them
        neg_voids = (elevations < -15000)
        elevations[neg_voids] = numpy.abs(elevations[neg_voids]) - 32768
        # Find positive voids and repair them
        pos_voids = (elevations > 15000)
        elevations[pos_voids] = 32768 - elevations[pos_voids]
        return elevations

    def _linear(self, ix, dx, iy, dy):
        # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
        a = (1 - dx) * self._lookup_elevation(ix, iy) + dx * self._lookup_elevation(ix + 1, iy)
        b = (1 - dx) * self._lookup_elevation(ix, iy+1) + dx * self._lookup_elevation(ix+1, iy+1)
        return (1 - dy) * a + dy * b

    def _lookup_elevation(self, ix, iy):
        # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
        t_ix = numpy.copy(ix)
        t_ix[t_ix >= self._shape[0]] = self._shape[0] - 1
        t_ix[t_ix < 0] = 0

        # adjust iy to account for 8 extra bytes at the beginning of each column
        t_iy = iy + 4
        t_iy[t_iy >= self._shape[1]+4] = self._shape[1] + 3
        t_iy[t_iy < 4] = 4

        return self._repair_values(self._mem_map[t_ix, t_iy])

    def in_bounds(self, lat, lon):
        """
        Determine which of the given points are inside the extent of this DTED.

        Parameters
        ----------
        lat : numpy.ndarray
        lon : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            boolean array of the same shape as lat/lon
        """

        return (lon >= self._bounding_box[0]) & (lon <= self._bounding_box[1]) & \
               (lat >= self._bounding_box[2]) & (lat <= self._bounding_box[3])

    def _get_elevation(self, lat, lon):
        # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray

        # we implicitly require that lat/lon make sense and are contained in this DTED

        # get indices
        fx = (lon - self._origin[0])/self._spacing[0]
        fy = (lat - self._origin[1])/self._spacing[1]

        # get integer indices via floor
        ix = numpy.cast[numpy.int32](numpy.floor(fx))
        iy = numpy.cast[numpy.int32](numpy.floor(fy))
        return self._linear(ix, fx-ix, iy, fy-iy)

    def get_elevation(self, lat, lon, block_size=50000):
        """
        Interpolate the elevation values for lat/lon. This is relative to the EGM96
        geoid by DTED specification.

        Parameters
        ----------
        lat : numpy.ndarray
        lon : numpy.ndarray
        block_size : None|int
            If `None`, then the entire calculation will proceed as a single block.
            Otherwise, block processing using blocks of the given size will be used.
            The minimum value used for this is 50,000, and any smaller value will be
            replaced with 50,000. Default is 50,000.

        Returns
        -------
        numpy.ndarray
            Elevation values of the same shape as lat/lon.
        """

        o_shape, lat, lon = argument_validation(lat, lon)

        out = numpy.full(lat.shape, numpy.nan, dtype=numpy.float64)
        if block_size is None:
            boolc = self.in_bounds(lat, lon)
            if numpy.any(boolc):
                out[boolc] = self._get_elevation(lat[boolc], lon[boolc])
        else:
            block_size = min(50000, int(block_size))
            start_block = 0
            while start_block < lat.size:
                end_block = min(lat.size, start_block + block_size)
                lat1 = lat[start_block:end_block]
                lon1 = lon[start_block:end_block]
                boolc = self.in_bounds(lat1, lon1)
                out1 = numpy.full(lat1.shape, numpy.nan, dtype=numpy.float64)
                out1[boolc] = self._get_elevation(lat1[boolc], lon[boolc])
                out[start_block:end_block] = out1
                start_block = end_block

        if o_shape == ():
            return float(out[0])
        else:
            return numpy.reshape(out, o_shape)

    def _find_overlap(self, lat_lon_box):
        """
        Gets the overlap slice argument for the lat/lon bounding box.

        Parameters
        ----------
        lat_lon_box : None|numpy.ndarray
            Of the form `[

        Returns
        -------
        (slice, slice)
        """

        if lat_lon_box is None:
            first_row = 0
            last_row = self._shape[0]
            first_col = 0
            last_col = self._shape[1]
        else:
            first_row = (lat_lon_box[2] - self._origin[0])/self._spacing[0]
            last_row = (lat_lon_box[3] - self._origin[0])/self._spacing[0]
            first_col = (lat_lon_box[0] - self._origin[1])/self._spacing[1]
            last_col = (lat_lon_box[1] - self._origin[1])/self._spacing[1]
            if first_row > self._shape[0] or last_row < 0 or first_col > self._shape[1] or last_col < 0:
                return None
            first_row = int(max(0, numpy.floor(first_row)))
            last_row = int(min(self._shape[0], numpy.ceil(last_row)))
            first_col = int(max(0, numpy.floor(first_col)))
            last_col = int(min(self._shape[1], numpy.ceil(last_col)))
        return slice(first_row, last_row, 1), slice(first_col, last_col, 1)

    def get_max(self, lat_lon_box=None):
        """
        Gets the maximum observed DEM value, possibly contained in the given
        rectangular area of interest.

        Parameters
        ----------
        lat_lon_box : None|numpy.ndarray
            None or of the form `[lat min, lat max, lon min, lon max]`.

        Returns
        -------
        float|None
        """

        arg = self._find_overlap(lat_lon_box)
        if arg is None:
            return None
        return numpy.max(self.__getitem__(arg))

    def get_min(self, lat_lon_box=None):
        """
        Gets the minimum observed DEM value, possibly contained in the given
        rectangular area of interest.

        Parameters
        ----------
        lat_lon_box : None|numpy.ndarray
            None or of the form `[lat min, lat max, lon min, lon max]`.

        Returns
        -------
        float|None
        """

        arg = self._find_overlap(lat_lon_box)
        if arg is None:
            return None
        return numpy.min(self.__getitem__(arg))


class DTEDInterpolator(DEMInterpolator):
    """
    DEM Interpolator using DTED/SRTM files for the DEM information.
    """

    __slots__ = ('_readers', '_geoid', '_ref_geoid')

    def __init__(self, files, geoid_file, lat_lon_box=None):
        if isinstance(files, str):
            files = [files, ]
        # get a reader object for each file
        self._readers = [DTEDReader(fil) for fil in files]

        # get the geoid object - we should prefer egm96 .pgm files, since that's the DTED spec
        #   in reality, it makes very little difference, though
        if isinstance(geoid_file, str):
            if os.path.isdir(geoid_file):
                geoid_file = GeoidHeight.from_directory(geoid_file, search_files=('egm96-5.pgm', 'egm96-15.pgm'))
            else:
                geoid_file = GeoidHeight(geoid_file)
        if not isinstance(geoid_file, GeoidHeight):
            raise TypeError(
                'geoid_file is expected to be the path where one of the standard '
                'egm .pgm files can be found, or an instance of GeoidHeight reader. '
                'Got {}'.format(type(geoid_file)))
        self._geoid = geoid_file

        self._lat_lon_box = lat_lon_box

        if len(self._readers) == 0:
            self._ref_geoid = 0
        else:
            # use the origin of the first reader
            ref_point = self._readers[0].origin
            self._ref_geoid = float(self._geoid.get(ref_point[1], ref_point[0]))
        self._max_geoid = None
        self._min_geoid = None

    @classmethod
    def from_coords_and_list(cls, lat_lon_box, dted_list, dem_type=None, geoid_file=None):
        """
        Construct a `DTEDInterpolator` from a coordinate collection and `DTEDList` object.

        .. Note:: This depends on using :func:`DTEDList.get_file_list`
            to get the relevant file list.

        Parameters
        ----------
        lat_lon_box : numpy.ndarray|list|tuple
            Of the form `[lat min, lat max, lon min, lon max]`.
        dted_list : DTEDList|str
            The dted list object or root directory
        dem_type : None|str|List[str]
            The DEM type or list of DEM types in order of priority.
        geoid_file : None|str|GeoidHeight
            The `GeoidHeight` object, an egm file name, or root directory containing
            one of the egm files in the sub-directory "geoid". If `None`, then default
            to the root directory of `dted_list`.

        Returns
        -------
        DTEDInterpolator
        """

        if isinstance(dted_list, string_types):
            dted_list = DTEDList(dted_list)
        if not isinstance(dted_list, DTEDList):
            raise ValueError(
                'dted_list os required to be a path (directory) or DTEDList instance.')

        # default the geoid argument to the root directory of the dted_list
        if geoid_file is None:
            geoid_file = dted_list.root_dir

        return cls(dted_list.get_file_list(lat_lon_box, dem_type=dem_type), geoid_file, lat_lon_box=lat_lon_box)

    @classmethod
    def from_reference_point(cls, ref_point, dted_list, dem_type=None, geoid_file=None, pad_value=0.1):
        """
        Construct a DTEDInterpolator object by padding around the reference point by
        `pad_value` latitude degrees (1 degree ~ 111 km or 69 miles).

        .. Note:: The degeneracy at the poles is not handled, because DTED are not
            defined there anyways.

        Parameters
        ----------
        ref_point : numpy.ndarray|list|tuple
            This is assumed to be of the form `[lat, lon, ...]`, and entries
            beyond the first two are ignored.
        dted_list : DTEDList|str
            The dted list object or root directory
        dem_type : None|str|List[str]
            The DEM type or list of DEM types in order of priority.
        geoid_file : None|str|GeoidHeight
            The `GeoidHeight` object, an egm file name, or root directory containing
            one of the egm files in the sub-directory "geoid". If `None`, then default
            to the root directory of `dted_list`.
        pad_value : float
            The degree value to pad by.

        Returns
        -------
        DTEDInterpolator
        """

        pad_value = float(pad_value)
        if pad_value > 0.5:
            pad_value = 0.5
        if pad_value < 0.05:
            pad_value = 0.05
        lat_diff = pad_value
        lat_max = min(ref_point[0] + lat_diff, 90)
        lat_min = max(ref_point[0] - lat_diff, -90)

        lon_diff = min(15, lat_diff/(numpy.sin(numpy.deg2rad(ref_point[0]))))
        lon_max = ref_point[1] + lon_diff
        if lon_max > 180:
            lon_max -= 360
        lon_min = ref_point[1] - lon_diff
        if lon_min < -180:
            lon_min += 360

        return cls.from_coords_and_list(
            [lat_min, lat_max, lon_min, lon_max], dted_list, dem_type=dem_type, geoid_file=geoid_file)

    @property
    def geoid(self):  # type: () -> GeoidHeight
        """
        GeoidHeight: Get the geoid height calculator
        """

        return self._geoid

    def _get_elevation_geoid_from_reader(self, reader, lat, lon):
        mask = reader.in_bounds(lat, lon)
        values = numpy.full(lat.shape, numpy.nan, dtype=numpy.float64)
        if numpy.any(mask):
            # noinspection PyProtectedMember
            values[mask] = reader._get_elevation(lat[mask], lon[mask])
        return mask, values

    def _get_elevation_geoid(self, lat, lon):
        out = numpy.full(lat.shape, numpy.nan, dtype=numpy.float64)
        remaining = numpy.ones(lat.shape, dtype=numpy.bool)
        for reader in self._readers:
            if not numpy.any(remaining):
                break
            mask, values = self._get_elevation_geoid_from_reader(reader, lat[remaining], lon[remaining])
            if numpy.any(mask):
                work_mask = numpy.copy(remaining)
                work_mask[remaining] = mask  # mask as a subset of remaining
                out[work_mask] = values[mask]
                remaining[work_mask] = False
        return out

    def get_elevation_hae(self, lat, lon, block_size=50000):
        """
        Get the elevation value relative to the WGS-84 ellipsoid.

        .. Note:: DTED elevation is relative to the egm96 geoid, and we are simply adding
            values determined by a geoid calculator. Using a the egm2008 model will result
            in only minor differences.

        Parameters
        ----------
        lat : numpy.ndarray|list|tuple|int|float
        lon : numpy.ndarray|list|tuple|int|float
        block_size : None|int
            If `None`, then the entire calculation will proceed as a single block.
            Otherwise, block processing using blocks of the given size will be used.
            A minimum value of 50000 will be enforced here.

        Returns
        -------
        numpy.ndarray
            The elevation relative to the WGS-84 ellipsoid.
        """

        return self.get_elevation_geoid(lat, lon, block_size=block_size) + \
            self._geoid.get(lat, lon, block_size=block_size)

    def get_elevation_geoid(self, lat, lon, block_size=50000):
        """
        Get the elevation value relative to the geoid.

        .. Note:: DTED elevation is relative to the egm96 geoid, though using the egm2008
            model will result in only minor differences.

        Parameters
        ----------
        lat : numpy.ndarray|list|tuple|int|float
        lon : numpy.ndarray|list|tuple|int|float
        block_size : None|int
            If `None`, then the entire calculation will proceed as a single block.
            Otherwise, block processing using blocks of the given size will be used.
            The minimum value used for this is 50000, and any smaller value will be
            replaced with 50000. Default is 50000.

        Returns
        -------
        numpy.ndarray
            the elevation relative to the geoid
        """

        o_shape, lat, lon = argument_validation(lat, lon)

        if block_size is None:
            out = self._get_elevation_geoid(lat, lon)
        else:
            block_size = min(50000, int(block_size))
            out = numpy.full(lat.shape, numpy.nan, dtype=numpy.float64)
            start_block = 0
            while start_block < lat.size:
                end_block = min(lat.size, start_block+block_size)
                out[start_block:end_block] = self._get_elevation_geoid(
                    lat[start_block:end_block], lon[start_block:end_block])
                start_block = end_block
        out[numpy.isnan(out)] = 0.0  # set missing values to geoid=0 (MSL)

        if o_shape == ():
            return float(out[0])
        else:
            return numpy.reshape(out, o_shape)

    def _get_ref_geoid(self, lat_lon_box):
        if lat_lon_box is None:
            # use the origin of the first reader
            ref_point = self._readers[0].origin
            return float(self._geoid.get(ref_point[1], ref_point[0]))
        else:
            return float(self._geoid.get(lat_lon_box[0], lat_lon_box[2]))

    def get_max_hae(self, lat_lon_box=None):
        return self.get_max_geoid(lat_lon_box=lat_lon_box) + self._get_ref_geoid(lat_lon_box)

    def get_min_hae(self, lat_lon_box=None):
        return self.get_min_geoid(lat_lon_box=lat_lon_box) + self._get_ref_geoid(lat_lon_box)

    def get_max_geoid(self, lat_lon_box=None):
        if len(self._readers) < 1:
            return self._get_ref_geoid(lat_lon_box)

        obs_maxes = [reader.get_max(lat_lon_box=lat_lon_box) for reader in self._readers]
        return float(max(value for value in obs_maxes if value is not None))

    def get_min_geoid(self, lat_lon_box=None):
        if len(self._readers) < 1:
            return self._get_ref_geoid(lat_lon_box)

        obs_mins = [reader.get_min(lat_lon_box=lat_lon_box) for reader in self._readers]
        return float(min(value for value in obs_mins if value is not None))

