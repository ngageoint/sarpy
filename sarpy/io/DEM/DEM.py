# -*- coding: utf-8 -*-

import os
import logging
import numpy
import struct
from typing import List

from sarpy.compliance import int_func, integer_types
from . import _argument_validation
from .geoid import GeoidHeight, find_geoid_file_from_dir


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

    def get_max_dem(self):
        """
        Get the maximum dem entry.

        Returns
        -------
        float
        """

        raise NotImplementedError

    def get_min_dem(self):
        """
        Get the minimum dem entry.

        Returns
        -------
        float
        """

        raise NotImplementedError


class DTEDList(object):
    """
    The DEM directory structure is assumed to look like the following:

        * For DTED<1,2>: `<root_dir>/dted/<1, 2>/<lon_string>/<lat_string>.dtd`

        * For SRTM<1,2,2F>: `<root_dir>/srtm/<1, 2, 2f>/<lon_string>/<lat_string>.dt<1,2,2>`

    Here `<lon_string>` corresponds to a string of the form `X###`, where
    `X` is one of 'N' or 'S', and `###` is the zero-padded formatted string for
    the integer value `floor(lon)`.

    Similarly, `<lat_string>` corresponds to a string of the form `Y##`, where
    `Y` is one of 'E' or 'W', and `##` is the zero-padded formatted string for
    the integer value `floor(lat)`.

    <lon_string>, <lat_string> corresponds to the origin in the lower left corner
    of the DEM tile.
    """

    __slots__ = ('_root_dir', )

    def __init__(self, root_directory=None):
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

    def get_file_list(self, lat, lon, dem_type):
        """
        Get the file list required for the given coordinates.

        Parameters
        ----------
        lat : numpy.ndarray|list|tuple|int|float
        lon : numpy.ndarray|list|tuple|int|float
        dem_type : str
            the DTED type - one of ("DTED1", "DTED2", "SRTM1", "SRTM2", "SRTM2F")

        Returns
        -------
        List[str]
        """

        def get_box(la, lo):
            la = int(numpy.floor(la))
            lo = int(numpy.floor(lo))
            if lo > 180:
                lo -= 360
            x = 'n' if la >= 0 else 's'
            y = 'e' if lo >= 0 else 'w'
            return '{0:s}{1:03d}{2:s}{3:02d}'.format(y, lo, x, la)

        # validate dem_type options
        dem_type = dem_type.upper()
        if dem_type not in _SUPPORTED_DTED_FILE_TYPES:
            raise ValueError(
                'dem_type must be one of the supported types {}'.format(list(_SUPPORTED_DTED_FILE_TYPES.keys())))
        if dem_type.startswith('DTED'):
            dstem = os.path.join(self._root_dir, dem_type[:4].lower(), dem_type[-1])
        elif dem_type.startswith('SRTM'):
            dstem = os.path.join(self._root_dir, dem_type[:4].lower(), dem_type[4:].lower())
        else:
            raise ValueError('Unhandled dem_type {}'.format(dem_type))

        if not os.path.isdir(dstem):
            raise IOError(
                "Based on configured of root_dir, it is expected that {} type dem "
                "files will lie below {}, which doesn't exist".format(dem_type, dstem))
        # get file extension
        fext = _SUPPORTED_DTED_FILE_TYPES[dem_type]['fext']

        # move data to numpy arrays
        if not isinstance(lat, numpy.ndarray):
            lat = numpy.array(lat)
        if not isinstance(lon, numpy.ndarray):
            lon = numpy.array(lon)

        files = []
        missing_boxes = []
        for box in set(get_box(*pair) for pair in zip(numpy.reshape(lat, (-1, )), numpy.reshape(lon, (-1, )))):
            fil = os.path.join(dstem, box[:4], box[4:] + fext)
            if os.path.isfile(fil):
                files.append(fil)
            else:
                missing_boxes.append(fil)
        if len(missing_boxes) > 0:
            logging.warning(
                'Missing required dem files {}. This will result in getting missing values '
                'for some points during any interpolation'.format(missing_boxes))
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
        self._bounding_box = numpy.zeros((2, 2), dtype=numpy.float64)
        self._bounding_box[0, :] = self._origin
        self._bounding_box[1, :] = self._origin + self._spacing*self._shape

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
                it1 = slice(start, stop, step=it.step)
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
        Determine which of the given points are inside the extent of the DTED

        Parameters
        ----------
        lat : numpy.ndarray
        lon : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            boolean array of the same shape as lat/lon
        """

        return (lat >= self._bounding_box[1][0]) & (lat <= self._bounding_box[1][1]) & \
               (lon >= self._bounding_box[0][0]) & (lon <= self._bounding_box[0][1])

    def _get_elevation(self, lat, lon):
        # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray

        # we implicitly require that lat/lon make sense and are contained in this DTED

        # get indices
        fx = (lon - self._origin[0])/self._spacing[0]
        fy = (lat - self._origin[1])/self._spacing[1]

        ix = numpy.cast[numpy.int32](numpy.floor(fx))
        iy = numpy.cast[numpy.int32](numpy.floor(fy))

        dx = fx - ix
        dy = fy - iy
        return self._linear(ix, dx, iy, dy)

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

        o_shape, lat, lon = _argument_validation(lat, lon)

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


class DTEDInterpolator(DEMInterpolator):
    """
    DEM Interpolator using DTED/SRTM files for the DEM information.
    """

    __slots__ = ('_readers', '_geoid')

    def __init__(self, files, geoid_file):
        if isinstance(files, str):
            files = [files, ]
        # get a reader object for each file
        self._readers = [DTEDReader(fil) for fil in files]

        # get the geoid object - we should prefer egm96 .pgm files, since that's the DTED spec
        #   in reality, it makes very little difference, though
        if isinstance(geoid_file, str):
            if os.path.isdir(geoid_file):
                geoid_file = GeoidHeight(
                    find_geoid_file_from_dir(geoid_file, search_files=('egm96-5.pgm', 'egm96-15.pgm')))
            else:
                geoid_file = GeoidHeight(geoid_file)
        if not isinstance(geoid_file, GeoidHeight):
            raise TypeError(
                'geoid_file is expected to be the path where one of the standard '
                'egm .pgm files can be found, or an instance of GeoidHeight reader. '
                'Got {}'.format(type(geoid_file)))
        self._geoid = geoid_file

    @classmethod
    def from_coords_and_list(cls, lats, lons, dted_list, dem_type, geoid_file=None):
        """
        Construct a `DTEDInterpolator` from a coordinate collection and `DTEDList` object.

        .. Note:: This depends on using `DTEDList.get_file_list(lats, lons, dted_type)`
            to get the relevant file list.

        Parameters
        ----------
        lats : numpy.ndarray|list|tuple|int|float
        lons : numpy.ndarray|list|tuple|int|float
        dted_list : None|DTEDList|str
            The dted list object or root directory
        dem_type : str
            The DEM type.
        geoid_file : None|str|GeoidHeight
            The `GeoidHeight` object, an egm file name, or root directory containing
            one of the egm files in the sub-directory "geoid". If `None`, then default
            to the root directory of `dted_list`.

        Returns
        -------
        DTEDInterpolator
        """

        if isinstance(dted_list, str):
            dted_list = DTEDList(dted_list)
        if not isinstance(dted_list, DTEDList):
            raise ValueError(
                'dted_list os required to be a path (directory) or DTEDList instance.')

        # default the geoid argument to the root directory of the dted_list
        if geoid_file is None:
            geoid_file = dted_list.root_dir

        return cls(dted_list.get_file_list(lats, lons, dem_type), geoid_file)

    @property
    def geoid(self):  # type: () -> GeoidHeight
        """
        GeoidHeight: Get the geoid height calculator
        """

        return self._geoid

    def _get_elevation_geoid(self, lat, lon):
        out = numpy.full(lat.shape, numpy.nan, dtype=numpy.float64)
        remaining = numpy.ones(lat.shape, dtype=numpy.bool)
        for reader in self._readers:
            if not numpy.any(remaining):
                break
            t_lat = lat[remaining]
            t_lon = lon[remaining]
            this = reader.in_bounds(t_lat, t_lon)
            if numpy.any(this):
                # noinspection PyProtectedMember
                out[remaining[this]] = reader._get_elevation(t_lat[this], t_lon[this])
                remaining[remaining[this]] = False
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
            The minimum value used for this is 50,000, and any smaller value will be
            replaced with 50,000. Default is 50,000.

        Returns
        -------
        numpy.ndarray
            the elevation relative to the WGS-84 ellipsoid.
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
            The minimum value used for this is 50,000, and any smaller value will be
            replaced with 50,000. Default is 50,000.

        Returns
        -------
        numpy.ndarray
            the elevation relative to the geoid
        """

        o_shape, lat, lon = _argument_validation(lat, lon)

        if block_size is None:
            out = self._get_elevation_geoid(lat, lon)
        else:
            block_size = min(50000, int(block_size))
            out = numpy.full(lat.shape, numpy.nan, dtype=numpy.float64)
            start_block = 0
            while start_block < lat.size:
                end_block = min(lat.size, start_block+block_size)
                out[start_block:end_block] = self._get_elevation_geoid(lat[start_block:end_block], lon[start_block:end_block])
                start_block = end_block

        if o_shape == ():
            return float(out[0])
        else:
            return numpy.reshape(out, o_shape)

    def get_max_dem(self):
        """
        Get the maximum DTED entry - note that this is relative to the geoid.

        Returns
        -------
        float
        """

        return float(max(numpy.max(reader[:, :]) for reader in self._readers))

    def get_min_dem(self):
        """
        Get the minimum DTED entry - note that this is relative to the geoid.

        Returns
        -------
        float
        """

        return float(min(numpy.min(reader[:, :]) for reader in self._readers))
