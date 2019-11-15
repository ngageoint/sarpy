import os
from copy import copy  # TODO: HIGH - usage should be replace with numpy.copy() method
import numpy as np
from scipy.interpolate import interpn

from .dem_log import dem_logger
from .readers import read_dted

__classification__ = "UNCLASSIFIED"


# TODO: HIGH - there seems to be some assumption about directory structure. Let's be explicit.

class DEM(object):
    """Class for handling DEM files."""

    def __init__(self, coordinates=[], masterpath='', dempaths=[], dem_type='DTED1', dem_buffer=1./60,
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

        self.avaiable_dems = ['DTED1', 'DTED2', 'SRTM1', 'SRTM2', 'SRTM2F']  # TODO: HIGH - this should be a property...fix typo.

        # Add any user specified dempaths to self
        if len(dempaths) == 0:  # TODO: MEDIUM - fix this up for better behavior?
            self.dempaths = []
        else: # If there are dempaths do some stuff
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
        self.origindd = np.zeros([ndems, 2])            # [LAT, LON] Origin in decimal degrees
        self.delta = np.zeros([ndems, 2])               # [LAT, LON] spacing in tenths of arcseconds
        self.deltadd = np.zeros([ndems, 2])             # [LAT, LON] spacing in decimal degrees
        self.deltam = np.zeros([ndems, 2])              # [LAT, LON] spacing in meters
        self.lat_list_1D = []                           # List to hold 1-D LAT arrays
        self.lon_list_1D = []                           # List to hold 1-D LON arrays
        self.lat_list_2D = []                           # List to hold 2-D LAT matrices
        self.lon_list_2D = []                           # List to hold 2-D LON matrices
        self.elevation_list = []                        # List to hold 2-D elevation matrices
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
                lat_range = np.array([np.floor(np.min(coordinates[:, 0])-dem_buffer),
                                      np.ceil(np.max(coordinates[:, 0])+dem_buffer)])  # Full LAT range
                if lat_range[0] < -90.0:  # Set limits on LAT range
                    lat_range[0] = -90.0
                if lat_range[1] >= 90.0:
                    lat_range[1] = 90.0
                # In order to handle longitudes properly at the international dateline (180 meets -180)
                # set new range from 0 -> 360 for ease of creating DEM range. Then just reset for dempath
                lon_360 = copy(coordinates[:, 1])
#                hemi_w = (lon_360 < 0)
                lon_360[(lon_360 < 0)] += 360.0
                lon_range = np.array([np.floor(np.min(lon_360)-dem_buffer), np.ceil(np.max(lon_360)+dem_buffer)])  # Full LON

                if (lat_range.size*lon_range.size > 25.0) and (lat_range.size*lon_range.size <= 100.0):
                    self.logger.warning('Coordinate range extends beyond 25 square degrees.')
                elif (lat_range.size*lon_range.size >= 100.0):
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
                            include_path = os.path.join(masterpath, dem_type[0:4].lower(), dem_type[-1], lon_short, lat_short + suffix)
                        elif dem_type[0:4].upper() == 'SRTM' and dem_type[-1].upper() == "F":
                            include_path = os.path.join(masterpath, dem_type[0:4].lower(), dem_type[-2:].lower(), lon_short, lat_short + suffix)

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
        indices = np.array([np.max(self.origindd[:,0]) - self.origindd[:,0],  # Get LAT rows
                            olon_360 - np.min(olon_360)]).astype(int)   # Get LON cols
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
                    if np.sign(self.lons_1D[1]) == 1 and np.sign(self.lon_list_1D[image_index][1]) == -1:  # If at dateline
                        self.lons_1D = np.hstack((self.lons_1D[0:-1], (360.0 + self.lon_list_1D[image_index])))  # Convert
                    else:
                        self.lons_1D = np.hstack((self.lons_1D[0:-1], self.lon_list_1D[image_index]))
                    elevrow = np.vstack((elevrow[0:-1, :], self.elevation_list[image_index]))
                j += 1
            if i == 0:  # Get the values for the first LAT col
                self.dem = elevrow
                self.lats_1D = self.lat_list_1D[image_index]
            else:  # Remove the common rows and join, ex. (34...35, 35....36), so remove the first 36 col
                self.lats_1D = np.hstack((self.lat_list_1D[image_index][0:-1],self.lats_1D))
                self.dem = np.hstack((elevrow[:, 0:-1], self.dem))
            i += 1

    def elevate(self, coord, method='linear', lonlat=False):
        """
        Calculate elevation (i.e. DEM value) at geographic coordinate(s).

        :param coord: [lat, lon] or [[lat, lon]]
        :param method: passed to `scipy.interpolate.interpn` method
        :param lonlat:
        :return:
        """

        # TODO: MEDIUM - the line immediately below this :(
        #   What is the intent of this whole method?

        if not 'dem' in dir(self):  # Check to see if dem is an attribute
            self.logger.warning('There are no DEMs to interpolate from.')  # If not, no dems have been read in
            return np.zeros(coord.shape[0]) + -1234.5
        if np.max(self.origindd[:, 1]) == 179.0 and np.min(self.origindd[:,1]) == -180.0:  # If DEM range crosses the dateline
            coord[(coord < 0)] += 360.0  # Convert to -180 to 180 -> 0 to 360
        # interpolated_elevation = interpn((1-D LON array, 1-D LAT array), 2-D elev array, coord array)
        coord = np.array(coord)  # Ensure coord is an array for the following line to work
        if len(coord.shape) == 1:  # If only one point is specified
            coord = coord.reshape((1,2))  # Then make sure its a 2-D array of 1 by 2 [[x,y]]
        if not lonlat:  # If coords are given at LAT/LON (y,x), then switch to LON/LAT (x,y)
            coord = self.geo_swap(coord)  # Convert [[lat,lon],...] to [[lon, lat], ...]
        # The following is to ensure interpn evaluates all the good, valid coordinates instead
        # of throwing the baby out with the bath water.
        elev = np.zeros(coord.shape[0]) + -1234.5  # Create a list of dummy elevations the same length as input list
        # Get all valid coordinates
        in_bounds = np.where((coord[:, 1] > np.min(self.lats_1D)) & (coord[:, 1] < np.max(self.lats_1D)) &
                             (coord[:, 0] > np.min(self.lons_1D)) & (coord[:, 0] < np.max(self.lons_1D)))[0]
        
        self.logger.debug('Coord LAT range: {}'.format((np.min(coord[:,1]),np.max(coord[:,1]))))
        self.logger.debug('Coord LON range: {}'.format((np.min(coord[:,0]),np.max(coord[:,0]))))
        self.logger.debug('DEM LAT range: {}'.format((np.min(self.lats_1D),np.max(self.lats_1D))))
        self.logger.debug('DEM LON range: {}'.format((np.min(self.lons_1D),np.max(self.lons_1D))))
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
            emin = np.round(np.min(elev[good_heights]),2)
            emean = np.round(np.mean(elev[good_heights]),2)
            emax = np.round(np.max(elev[good_heights]),2)
            self.logger.info('Elevation stats (min/mean/max): {}/{}/{}'.format(emin, emean, emax))
        else:
            self.logger.info('No valid points found.')
        return elev

    def geo_swap(self, incoord):
        """
        Provides a way for users to input LON/LAT coords instead of LAT/LON.
        :param incoord:
        :return:
        """

        # TODO: MEDIUM - Why??? :(

        incoord = np.array(incoord)  # Ensure input coords are numpy arrays
        if len(incoord.shape) == 1:  # If only one point is specified
            incoord = incoord.reshape((1, 2))  # Then make sure its a 2-D array of 1 by 2 [[x,y]]
        outcoord = np.array(copy(incoord))
        outcoord[:, 0] = incoord[:, 1]
        outcoord[:, 1] = incoord[:, 0]
        return outcoord
