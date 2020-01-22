# -*- coding: utf-8 -*-

import os
import numpy
import struct


from copy import copy  # TODO: HIGH - usage should be replace with numpy.copy() method
import numpy as np
from scipy.interpolate import interpn

from .dem_log import dem_logger

__classification__ = "UNCLASSIFIED"

#######
# module variables
_SUPPORTED_FILE_TYPES = {
    'DTED1': {'fext': '.dtd'},
    'DTED2': {'fext': '.dtd'},
    'SRTM1': {'fext': '.dt1'},
    'SRTM2': {'fext': '.dt2'},
    'SRTM2F': {'fext': '.dt2'}}


class DTEDList(object):
    """
    The DEM directory structure is assumed to look like the following:
    For DTED<1,2>: `<root_dir>/dted/<1, 2>/<lon_string>/<lat_string>.dtd`
    For SRTM<1,2,2F>: `<root_dir>/srtm/<1, 2, 2f>/<lon_string>/<lat_string>.dt<1,2,2>`

    Here `<lon_string>` corresponds to a string of the form `X###`, where
    `X` is one of 'N' or 'S', and `###` is the zero-padded formatted string for
    the integer value `floor(lon)`.

    Similarly, `<lat_string>` corresponds to a string of the form `Y##`, where
    `Y` is one of 'E' or 'W', and `##` is the zero-padded formatted string for
    the integer value `floor(lat)`.

    <lon_string>, <lat_string> corresponds to the upper left? lower left? corner
    of the DEM tile.
    """
    # TODO: complete the doc string - which corner?

    __slots__ = ('_root_dir', )

    def __init__(self, root_directory):
        self._root_dir = root_directory

    def get_file_list(self, lat, lon, dem_type):
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
        if dem_type not in _SUPPORTED_FILE_TYPES:
            raise ValueError(
                'dem_type must be one of the supported types {}'.format(list(_SUPPORTED_FILE_TYPES.keys())))
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
        fext = _SUPPORTED_FILE_TYPES[dem_type]['fext']

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
            raise ValueError('Missing required dem files {}'.format(missing_boxes))
        # TODO: is automatically fetching these from some source feasible?
        return files


class DTEDReader(object):
    __slots__ = ('_file_name', '_origin', '_spacing', '_bounding_box', '_shape')

    def __init__(self, file_name):
        self._file_name = file_name
        self._origin = None
        self._spacing = None
        self._bounding_box = None
        self._shape = None

    def _read_dted_header(self):
        def convert_format(str_in):
            lon = float(str_in[4:7]) + float(str_in[7:9])/60. + float(str_in[9:11])/3600.
            lon = -lon if str_in[11] == 'W' else lon
            lat = float(str_in[12:15]) + float(str_in[15:17])/60. + float(str_in[17:19])/3600.
            lat = -lat if str_in[19] == 'S' else lat
            return lon, lat

        # the header is only 80 characters long
        with open(self._file_name, 'rb') as fi:
            # NB: DTED is always big-endian
            header = struct.unpack('>80s', fi.read(80))[0].decode('utf-8')

        if header[:3] != 'UHL':
            raise IOError('File {} does not appear to be a DTED file.'.format(self._file_name))

        # DTED is always in Lat/Lon order
        self._origin = numpy.array(convert_format(header), dtype=numpy.float64)
        self._spacing = numpy.array([float(header[20:24]), float(header[24:28])], dtype=numpy.float64)/36000.
        self._shape = numpy.array([int(header[47:51]), int(header[51:55])], dtype=numpy.int64)
        self._bounding_box = numpy.zeros((2, 2), dtype=numpy.float64)
        self._bounding_box[0, :] = self._origin
        self._bounding_box[1, :] = self._origin + self._spacing*self._shape

    def _get_mem_map(self):
        # the first 80 characters are header
        # characters 80:728 are data set identification record
        # characters 728:3428 are accuracy record
        # the remainder is data records...each row has 8 extra bytes at the beginning and 4 extra at the end...why?
        shp = (int(self._shape[0]), int(self._shape[1]) + 4 + 2)  # extra bytes...why?
        return numpy.memmap(self._file_name,
                            dtype=numpy.dtype('>i2'),
                            mode='r',
                            offset=3428,
                            shape=shp)

    def _linear(self, ix, dx, iy, dy, mem_map):
        a = (1 - dx) * self._get_elevations(ix, iy, mem_map) + dx * self._get_elevations(ix + 1, iy, mem_map)
        b = (1 - dx) * self._get_elevations(ix, iy+1, mem_map) + dx * self._get_elevations(ix+1, iy+1, mem_map)
        return (1 - dy) * a + dy * b

    def _get_elevations(self, ix, iy, mem_map):
        t_ix = numpy.copy(ix)
        t_ix[t_ix >= self._shape[0]] = self._shape[0] - 1
        t_ix[t_ix < 0] = 0

        # adjust iy to account for 8 extra bytes at the beginning of each column
        t_iy = iy + 4
        t_iy[t_iy >= self._shape[1]+4] = self._shape[1] + 3
        t_iy[t_iy < 4] = 4

        elevations = mem_map[t_ix, t_iy]

        # BASED ON MIL-PRF-89020B SECTION 3.11.1, 3.11.2
        # There is some byte-swapping nonsense that is poorly explained.
        # The following steps appear to correct for the "complemented" values.
        neg_voids = (elevations < -15000.0)  # Find negative voids
        elevations[neg_voids] = np.abs(elevations[neg_voids]) - 32768.0  # And fill them in (2**15 = 32768)
        pos_voids = (elevations > 15000.0)  # Find positive voids
        elevations[pos_voids] = 32768.0 - elevations[pos_voids]  # And fill them in
        return elevations

    def in_bounds(self, lat, lon):
        return (lat >= self._bounding_box[1][0]) & (lat <= self._bounding_box[1][1]) & \
               (lon >= self._bounding_box[0][0]) & (lon <= self._bounding_box[0][1])

    def interp(self, lat, lon):
        mem_map = self._get_mem_map()

        # get indices
        fx = (lon - self._origin[0])/self._spacing[0]
        fy = (lat - self._origin[1])/self._spacing[1]

        ix = numpy.cast[numpy.int32](numpy.floor(fx))
        iy = numpy.cast[numpy.int32](numpy.floor(fy))

        dx = fx - ix
        dy = fy - iy
        return self._linear(ix, dx, iy, dy, mem_map)


class DTEDInterpolator(object):
    __slots__ = ('_readers', )

    def __init__(self, files):
        if isinstance(files, str):
            files = [files, ]
        # get a reader object for each file
        self._readers = [DTEDReader(fil) for fil in files]

    def interp(self, lat, lon):
        if not isinstance(lat, numpy.ndarray):
            lat = numpy.array(lat)
        if not isinstance(lon, numpy.ndarray):
            lon = numpy.array(lon)
        if lat.shape != lon.shape:
            raise ValueError(
                'lat and lon must have the same shape, got '
                'lat.shape = {}, lon.shape = {}'.format(lat.shape, lon.shape))
        o_shape = lat.shape
        lat = numpy.reshape(lat, (-1, ))
        lon = numpy.reshape(lon, (-1, ))

        out = numpy.full(lat.shape, numpy.nan, dtype=numpy.float64)
        remaining = numpy.ones(lat.shape, dtype=numpy.bool)
        for reader in self._readers:
            if not numpy.any(remaining):
                break
            t_lat = lat[remaining]
            t_lon = lon[remaining]
            this = reader.in_bounds(t_lat, t_lon)
            if numpy.any(this):
                out[remaining[this]] = reader.interp(t_lat[this], t_lon[this])
                remaining[remaining[this]] = False

        if o_shape == ():
            return float(out[0])
        else:
            return numpy.reshape(out, o_shape)

    @classmethod
    def from_coords_and_list(cls, lats, lons, dted_list, dem_type):
        """
        Construct a DTEDInterpolator from a coordinate collection and DTEDList object.

        Parameters
        ----------
        lats : numpy.ndarray|list|tuple|int|float
        lons : numpy.ndarray|list|tuple|int|float
        dted_list : DTEDList|str
            the dtedList object or root directory
        dem_type : str
            the DEM type.

        Returns
        -------
        DTEDInterpolator
        """

        if isinstance(dted_list, str):
            dted_list = DTEDList(dted_list)
        if not isinstance(dted_list, DTEDList):
            raise ValueError(
                'dted_list os required to be a path (directory) or DTEDList instance.')

        return cls(dted_list.get_file_list(lats, lons, dem_type))


# TODO: REFACTOR the below, or just kill it?


class DEM(object):
    """Class for handling DEM files."""

    def __init__(self, coordinates=[], masterpath='', dempaths=[], dem_type='DTED1', dem_buffer=1. / 60,
                 lonlat=False, log_to_console=True, log_to_file=None, log_level=None):
        """
        :param coordinates: A point/list of LAT/LON coordinates to use for grabbing a DEM. REQUIRES masterpath!
        :param masterpath: Top Level directory were DEMs are held. REQUIRES coordinates!
        :param dempaths: Specify exactly the file(s) needed instead of using coords. Can be used with coords/masterpath
        :param dem_type: One of ['DTED1', 'DTED2', 'SRTM1', 'SRTM2', 'SRTM2F']
        :param dem_buffer: Ensures that coordinates (see above) on the edge of have enough DEM points for proper
            interpolation. Default 1/60 degrees (1 minute).
        :param lonlat:
        :param log_to_console: boolean condition for writing log messages to console
        :param log_to_file: None or path for log file.
        :param log_level: logging level.
        """

        # TODO: HIGH - mutable default arguments.
        #   Something is totally ridiculous for the logging. What is the intent?

        self.logger = dem_logger('ELEVATOR', level=log_level, logfile=log_to_file, log_to_console=log_to_console)

        self.avaiable_dems = ['DTED1', 'DTED2', 'SRTM1', 'SRTM2',
                              'SRTM2F']  # TODO: HIGH - this should be a property...fix typo.

        # Add any user specified dempaths to self
        if len(dempaths) == 0:  # TODO: MEDIUM - fix this up for better behavior?
            self.dempaths = []
        else:  # If there are dempaths do some stuff
            self.dempaths = dempaths
            if 'list' not in str(type(self.dempaths)):  # Ensure type == list for reading single or multiple files
                self.dempaths = [dempaths]
            for dempath in self.dempaths:
                if not os.path.exists(dempath):  # If the user path does not exist, remove it.
                    self.dempaths.remove(dempath)
                    self.logger.warning('Could not locate user specified file {}'.format(dempath))

        # If coordinates are specified, find the right DEMS and use those
        if coordinates != []:  # TODO: MEDIUM - again, use property?
            if lonlat:  # If coordinates are [LON,LAT] instead of [LAT/LON]
                self.logger.debug('Switching Coordinates')
                coordinates = self.geo_swap(coordinates)  # Then swap to ensure correctness
            self.include(coordinates, masterpath, dem_type, dem_buffer)

        ndems = len(self.dempaths)
        self.origin = np.zeros([ndems, 2]).astype(str)  # DDMMSS [LAT, LON] origin for LOWER LEFT corner of the data
        self.origindd = np.zeros([ndems, 2])  # [LAT, LON] Origin in decimal degrees
        self.delta = np.zeros([ndems, 2])  # [LAT, LON] spacing in tenths of arcseconds
        self.deltadd = np.zeros([ndems, 2])  # [LAT, LON] spacing in decimal degrees
        self.deltam = np.zeros([ndems, 2])  # [LAT, LON] spacing in meters
        self.lat_list_1D = []  # List to hold 1-D LAT arrays
        self.lon_list_1D = []  # List to hold 1-D LON arrays
        self.lat_list_2D = []  # List to hold 2-D LAT matrices
        self.lon_list_2D = []  # List to hold 2-D LON matrices
        self.elevation_list = []  # List to hold 2-D elevation matrices
        # TODO: HIGH - why load all this crap in memory?

        if self.dempaths != []:
            # If there are files in the list and init_read is set
            # TODO: MEDIUM - at least make this hidden. Should handle the condition seamlessly.
            self.read_dempath()

        if len(self.elevation_list) > 0:
            self.join_dems()  # Join multiple dems if they exist then join the DEMs together
            # TODO: MEDIUM - at least make this hidden. Should handle the condition seamlessly.

    def include(self, coordinates, masterpath, dem_type, dem_buffer):
        """
        Find DEM file (or files) to read based on a coordinate (or coordinates). Assumed to be in (lat,lon) format.
        :param coordinates:
        :param masterpath:
        :param dem_type:
        :param dem_buffer:
        :return: A list of valid files to pass to the actual reader
        """

        self.logger.info('Including filepaths.')
        if len(coordinates) > 0:
            if dem_type in self.avaiable_dems:  # Check DEM type
                if dem_type.upper() in ['DTED', 'SRTM1', 'SRTM2']:  # Check to see if DTED
                    suffix = '.dt{}'.format(dem_type[-1])  # Append the D,1,2 to the suffix?
                elif 'SRTM2F' in dem_type:
                    suffix = '.dt2'
                # TODO: HIGH - orphaned todo below. We should have a map.
                # Add elif statements for other DEM formats
            else:
                self.logger.critical('Cannot read DEM type {}'.format(dem_type))  # TODO: HIGH - raise ValueError.
                return

            if not os.path.exists(masterpath):
                self.logger.warning('Master DEM path --  {}  -- does not appear to exist.'.format(masterpath))
            else:
                coordinates = np.array(coordinates)  # Make sure coordinates are an Nx2 array, not a list
                if len(coordinates.shape) == 1:
                    coordinates = np.array([coordinates])  # If coordinates is a 1D array, convert to 2D
                elif len(coordinates.shape) >= 3:
                    self.logger.warning('3-Dimensional array detected. May cause unknown errors!')
                lat_range = np.array([np.floor(np.min(coordinates[:, 0]) - dem_buffer),
                                      np.ceil(np.max(coordinates[:, 0]) + dem_buffer)])  # Full LAT range
                if lat_range[0] < -90.0:  # Set limits on LAT range
                    lat_range[0] = -90.0
                if lat_range[1] >= 90.0:
                    lat_range[1] = 90.0
                # In order to handle longitudes properly at the international dateline (180 meets -180)
                # set new range from 0 -> 360 for ease of creating DEM range. Then just reset for dempath
                lon_360 = copy(coordinates[:, 1])
                #                hemi_w = (lon_360 < 0)
                lon_360[(lon_360 < 0)] += 360.0
                lon_range = np.array(
                    [np.floor(np.min(lon_360) - dem_buffer), np.ceil(np.max(lon_360) + dem_buffer)])  # Full LON

                if (lat_range.size * lon_range.size > 25.0) and (lat_range.size * lon_range.size <= 100.0):
                    self.logger.warning('Coordinate range extends beyond 25 square degrees.')
                elif (lat_range.size * lon_range.size >= 100.0):
                    self.logger.warning('Coordinate range too large (>100 square degrees).')
                    self.logger.warning('Quitting DEM reading.')
                    return

                for lat in np.arange(lat_range[0], lat_range[1], 1):
                    for lon in np.arange(lon_range[0], lon_range[1], 1):
                        if lon >= 180.0:
                            lon -= 360.0
                        if np.sign(lat) >= 0:
                            lat_hemi = 'n'  # Northern hemisphere
                        else:
                            lat_hemi = 's'  # Southern hemisphere
                        if np.sign(lon) >= 0:
                            lon_hemi = 'e'  # Eastern hemisphere
                        else:
                            lon_hemi = 'w'  # Western hemisphere
                        lat_short = lat_hemi + str(np.abs(lat).astype(int)).zfill(2)  # TODO: MEDIUM - format string
                        lon_short = lon_hemi + str(np.abs(lon).astype(int)).zfill(3)

                        if dem_type[0:4].upper() == 'DTED':
                            include_path = os.path.join(masterpath, dem_type[0:4].lower(), dem_type[-1], lon_short,
                                                        lat_short + suffix)
                        elif dem_type[0:4].upper() == 'SRTM' and dem_type[-1].upper() == "F":
                            include_path = os.path.join(masterpath, dem_type[0:4].lower(), dem_type[-2:].lower(),
                                                        lon_short, lat_short + suffix)

                        if os.path.exists(include_path):
                            if include_path not in self.dempaths:
                                self.dempaths.append(include_path)
                        else:
                            self.logger.warning('include() could not find file {}'.format(include_path))
                self.logger.debug('include_path {}'.format(include_path))
        else:
            self.logger.critical('No DEM files used. Could not find appropriate files.')

    def read_dempath(self):
        """Read all files within DEM.dempaths"""

        self.logger.info('Going to read {} DEM(s).'.format(len(self.dempaths)))

        if len(self.dempaths) < 1:
            self.logger.warning('Nothing to read.')
            return []
        for i, dempath in enumerate(self.dempaths):  # Cycle through list of dempaths
            self.logger.debug('Reading DEM file {}'.format(dempath))
            try:
                dem_specs, geos_1D, geos_2D, elevations = read_dted(dempath)  # Read each file one at a time.
            except:
                self.logger.warning('read_dted failed for {}'.format(dempath))  # Print warning
                continue  # Then skip the rest of the loop and continue on
            self.origin[i, :] = dem_specs[0]
            self.origindd[i, :] = dem_specs[1]
            self.delta[i, :] = dem_specs[2]
            self.deltadd[i, :] = dem_specs[3]
            self.deltam[i, :] = dem_specs[4]
            self.lat_list_1D.append(geos_1D[0])
            self.lon_list_1D.append(geos_1D[1])
            self.lat_list_2D.append(geos_2D[0])
            self.lon_list_2D.append(geos_2D[1])
            self.elevation_list.append(elevations)

    def join_dems(self):
        """Take all the DEMs in dempath and stick them together."""

        # TODO: MEDIUM - review this functionality. This looks gross?
        self.logger.info('Joining DEMs together if necessary.')
        olon_360 = copy(self.origindd[:, 1])  # Get the lon origins
        self.logger.debug('DEM origins: {}'.format(self.origindd))
        self.logger.debug('DEM LON min/max: {}/{}'.format(np.min(self.origindd[:, 1]), np.max(self.origindd[:, 1])))
        if np.max(self.origindd[:, 1]) == 179.0 and np.min(self.origindd[:, 1]) == -180.0:
            # DEM range crosses the dateline
            olon_360[(olon_360 < 0)] += 360.0  # Convert to -180 to 180 -> 0 to 360

        # indices labels which files go together in LAT space and LON space.
        indices = np.array([np.max(self.origindd[:, 0]) - self.origindd[:, 0],  # Get LAT rows
                            olon_360 - np.min(olon_360)]).astype(int)  # Get LON cols
        nlats = np.unique(indices[0, :]).shape[0]  # Get number of lat rows
        nlons = np.unique(indices[1, :]).shape[0]  # Get number of lon rows

        i = 0
        while i < nlats:
            j = 0
            while j < nlons:
                # To fill grid from top left to bottom right, match i,j to indices from above
                image_index = np.where((indices[0, :] == i) & (indices[1, :] == j))[0][0].astype(int)  # Get index
                if j == 0:  # Get the values for the first LON row
                    self.lons_1D = self.lon_list_1D[image_index]
                    elevrow = self.elevation_list[image_index]
                else:  # Remove the common columns and join, ex. (73...74, 74....75), so remove the first 74 row
                    if np.sign(self.lons_1D[1]) == 1 and np.sign(
                            self.lon_list_1D[image_index][1]) == -1:  # If at dateline
                        self.lons_1D = np.hstack(
                            (self.lons_1D[0:-1], (360.0 + self.lon_list_1D[image_index])))  # Convert
                    else:
                        self.lons_1D = np.hstack((self.lons_1D[0:-1], self.lon_list_1D[image_index]))
                    elevrow = np.vstack((elevrow[0:-1, :], self.elevation_list[image_index]))
                j += 1
            if i == 0:  # Get the values for the first LAT col
                self.dem = elevrow
                self.lats_1D = self.lat_list_1D[image_index]
            else:  # Remove the common rows and join, ex. (34...35, 35....36), so remove the first 36 col
                self.lats_1D = np.hstack((self.lat_list_1D[image_index][0:-1], self.lats_1D))
                self.dem = np.hstack((elevrow[:, 0:-1], self.dem))
            i += 1

    def elevate(self, coord, method='linear', lonlat=False):
        """
        Calculate elevation (i.e. DEM value) at geographic coordinate(s).

        Parameters
        ----------
        coord : numpy.ndarray
        method : str
        lonlat : bool
            Is this [Longitude, Latitude] order?

        Returns
        -------

        """

        # TODO: MEDIUM - what is the "real" intent of this whole method?

        if hasattr(self, 'dem'):  # Check to see if dem is an attribute
            # TODO: HIGH - this doesn't cut it
            self.logger.warning('There are no DEMs to interpolate from.')  # If not, no dems have been read in
            return np.zeros(coord.shape[0]) + -1234.5
        if np.max(self.origindd[:, 1]) == 179.0 and np.min(
                self.origindd[:, 1]) == -180.0:  # If DEM range crosses the dateline
            coord[(coord < 0)] += 360.0  # Convert to -180 to 180 -> 0 to 360
        # interpolated_elevation = interpn((1-D LON array, 1-D LAT array), 2-D elev array, coord array)
        coord = np.array(coord)  # Ensure coord is an array for the following line to work
        if len(coord.shape) == 1:  # If only one point is specified
            coord = coord.reshape((1, 2))  # Then make sure its a 2-D array of 1 by 2 [[x,y]]
        if not lonlat:  # If coords are given at LAT/LON (y,x), then switch to LON/LAT (x,y)
            coord = self.geo_swap(coord)  # Convert [[lat,lon],...] to [[lon, lat], ...]
        # The following is to ensure interpn evaluates all the good, valid coordinates instead
        # of throwing the baby out with the bath water.
        elev = np.zeros(coord.shape[0]) + -1234.5  # Create a list of dummy elevations the same length as input list
        # Get all valid coordinates
        in_bounds = np.where((coord[:, 1] > np.min(self.lats_1D)) & (coord[:, 1] < np.max(self.lats_1D)) &
                             (coord[:, 0] > np.min(self.lons_1D)) & (coord[:, 0] < np.max(self.lons_1D)))[0]

        self.logger.debug('Coord LAT range: {}'.format((np.min(coord[:, 1]), np.max(coord[:, 1]))))
        self.logger.debug('Coord LON range: {}'.format((np.min(coord[:, 0]), np.max(coord[:, 0]))))
        self.logger.debug('DEM LAT range: {}'.format((np.min(self.lats_1D), np.max(self.lats_1D))))
        self.logger.debug('DEM LON range: {}'.format((np.min(self.lons_1D), np.max(self.lons_1D))))
        self.logger.debug('Coord shape: {}'.format(coord.shape))
        self.logger.debug('Elev size: {}'.format(elev.size))
        self.logger.debug('In_bounds size: {}'.format(in_bounds.size))
        if in_bounds.size < elev.size:
            self.logger.warning('Some points may be outside of DEM boundary. Check coordinate list.')
        if in_bounds.size > 0:  # If there are any valid points, then do try the interpolation
            try:
                self.logger.info('Interpolating elevation points.')
                elev[in_bounds] = interpn((self.lons_1D, self.lats_1D), self.dem, coord[in_bounds, :], method=method)
            except Exception as err:
                self.logger.critical('Interpolation error: {}'.format(err))
        good_heights = np.where(elev > -1234.5)[0]  # Do stats on valid points only.
        if good_heights.size > 0:  # If there are good points then print stats
            emin = np.round(np.min(elev[good_heights]), 2)
            emean = np.round(np.mean(elev[good_heights]), 2)
            emax = np.round(np.max(elev[good_heights]), 2)
            self.logger.info('Elevation stats (min/mean/max): {}/{}/{}'.format(emin, emean, emax))
        else:
            self.logger.info('No valid points found.')
        return elev

    def geo_swap(self, incoord):
        """
        Provides a way for users to input LON/LAT coords instead of LAT/LON.

        Parameters
        ----------
        incoord : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        # TODO: MEDIUM - this should return a consistent shape as input
        incoord = np.array(incoord)  # Ensure input coords are numpy arrays
        if len(incoord.shape) == 1:  # If only one point is specified
            incoord = incoord.reshape((1, 2))  # Then make sure its a 2-D array of 1 by 2 [[x,y]]
        return incoord[:, ::-1]


def calculateEarthRadius(lat_deg):
    """
    Get the radius of the earth at the given latitude.

    Parameters
    ----------
    lat_deg : numpy.ndarray|int|float

    Returns
    -------
    numpy.ndarray|float
    """

    major = 6378137.0  # semi-major axis of the earth
    minor = 6356752.3142  # semi-minor axis of the earth
    lat_cos = np.cos(np.deg2rad(lat_deg))
    lat_sin = np.sin(np.deg2rad(lat_deg))

    return np.sqrt(((major*major*lat_cos)**2 + (minor*minor*lat_sin)**2)/((major*lat_cos)**2 + (minor*lat_sin)**2))


def read_dted(demfile):
    """
    Read a single DTED1 or DTED2 file.

    Parameters
    ----------
    demfile : str

    Returns
    -------

    """
    # TODO: populate docstring - should this even be a method like this?

    def convert_format(str_in):
        lon = float(str_in[4:7]) + float(str_in[7:9])/60. + float(str_in[9:11])/3600.
        lon = -lon if str_in[11] == 'W' else lon
        lat = float(str_in[12:15]) + float(str_in[15:17])/60. + float(str_in[17:19])/3600.
        lat = -lat if str_in[19] == 'S' else lat
        return lat, lon

    with open(demfile, 'rb') as df:  # Open file safely using "with open"
        entire_file = df.read()  # Read the whole file

    try:  # Make sure file is bytes-decodable and not empty
        uhl = entire_file[0:80].decode()  # User header label
        # dsi = entire_file[80:728]  # Data set identification record, not needed, here for completeness
        # acc = entire_file[728:3428]  # Accuracy record, not needed, here for completeness
        data = entire_file[3428:]  # Data records
    except Exception as error:  # If file cannot be read like a standard DTED, then error out.
        print(error)
        print('Could not decode file. Check for corrupted file.')
        return None

    if not uhl[0:3] == 'UHL':  # If the first three bytes are not UHL, then it is not a DTED file
        print('File does not appear to be in the DTED standard.')
        return None

    origin = [uhl[12:20], uhl[4:12]]  # Get DDMMSSH
    origindd = np.array(convert_format(uhl[:20]), dtype=np.float64)

    delta = np.array([uhl[24:28], uhl[20:24]]).astype(float)  # Get 0.1" spacing
    deltadd = delta / 36000.0  # Convert to DD spacing
    earth_radius = calculateEarthRadius(origindd[0])  # Get Earth radius at LAT
    deltam = [np.cos(np.deg2rad(origindd[0])) * np.deg2rad(deltadd[1]) * earth_radius,
                   np.deg2rad(deltadd[0]) * earth_radius]  # Convert spacing to meters

    nlon = int(uhl[47:51])  # Number of longitude points, read > to string > to int
    nlat = int(uhl[51:55])  # Number of latitude points, read > to string > to int

    lats_1D = origindd[0] + np.arange(0, nlat, 1) * deltadd[0]  # An array of elevation latitudes
    lons_1D = origindd[1] + np.arange(0, nlon, 1) * deltadd[1]  # An array of elevation longitudes

    # TODO: why the 2-d version? This is just a waste of memory?
    lats_2D, lons_2D = np.meshgrid(lats_1D, lons_1D)  # Create arrays for LAT/LON
    elevations = np.zeros(lats_2D.shape)  # Array in matrix form for creating DEM images

    row_bytes = nlat*2 + 12  # Number of bytes per data row
    for i in range(nlon):
        data_row = data[i * row_bytes:(i + 1) * row_bytes]
        elevations[i, :] = np.ndarray(shape=(nlat,), dtype='>i2', buffer=data_row[8:row_bytes - 4])

    # BASED ON MIL-PRF-89020B SECTION 3.11.1, 3.11.2
    # There is some byte-swapping nonsense that is poorly explained.
    # The following steps appear to correct for the "complemented" values.
    neg_voids = (elevations < -15000.0)  # Find negative voids
    elevations[neg_voids] = np.abs(elevations[neg_voids]) - 32768.0  # And fill them in (2**15 = 32768)
    pos_voids = (elevations > 15000.0)  # Find positive voids
    elevations[pos_voids] = 32768.0 - elevations[pos_voids]  # And fill them in

    return [origin, origindd, delta, deltadd, deltam], [lats_1D, lons_1D], [lats_2D, lons_2D], elevations

