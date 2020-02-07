"""
Readers file to house the various formatted DEM file readers
"""

import numpy as np

from .geodesy import calculateEarthRadius

__classification__ = "UNCLASSIFIED"


def read_dted(demfile):
    '''
    Read a single DTED1 or DTED2 file
    '''
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
#    hemisphere = [uhl[11], uhl[19]]  # Get hemisphere
#    print(demfile)
#    print(uhl)
#    print(origin)
    origindd = np.array([int(origin[0][0:3]) +
                              int(origin[0][3:5]) / 60. +
                              int(origin[0][5:7]) / 3600.,
                              int(origin[1][0:3]) +
                              int(origin[1][3:5]) / 60. +
                              int(origin[1][5:7]) / 3600.])  # Convert to decimal degrees (dd)
    if origin[0][-1] == 'S':
        origindd[0] *= -1  # Get correct DD origin if in the South
    if origin[1][-1] == 'W':
        origindd[1] *= -1  # Get correct DD origin if in the West

    delta = np.array([uhl[24:28], uhl[20:24]]).astype(float)  # Get 0.1" spacing
    deltadd = delta / 36000.0  # Convert to DD spacing
    earth_radius = calculateEarthRadius(origindd[0])  # Get Earth radius at LAT
    deltam = [np.cos(np.deg2rad(origindd[0])) * np.deg2rad(deltadd[1]) * earth_radius,
                   np.deg2rad(deltadd[0]) * earth_radius]  # Convert spacing to meters

    nlon = int(uhl[47:51])  # Number of longitude points, read > to string > to int
    nlat = int(uhl[51:55])  # Number of latitude points, read > to string > to int

    lats_1D = origindd[0] + np.arange(0, nlat, 1) * deltadd[0]  # An array of elevation latitudes
    lons_1D = origindd[1] + np.arange(0, nlon, 1) * deltadd[1]  # An array of elevation longitudes

    lats_2D, lons_2D = np.meshgrid(lats_1D, lons_1D)  # Create arrays for LAT/LON
    elevations = np.zeros(lats_2D.shape)  # Array in matrix form for creating DEM images

    # dem_info = []  # List containing line record data from DEM
    row_bytes = nlat*2 + 12  # Number of bytes per data row
    for i in range(nlon):
        data_row = data[i * row_bytes:(i + 1) * row_bytes]
        # [252_8, data block coiunt, LON count, LAT count, Checksum] , not needed, here for completeness
        # dem_info.append([data_row[0], data_row[1:4], data_row[4:6], data_row[6:8], data_row[row_bytes - 4:-1]])
        elevations[i, :] = np.ndarray(shape=(nlat,), dtype='>i2', buffer=data_row[8:row_bytes - 4])

    # BASED ON MIL-PRF-89020B SECTION 3.11.1, 3.11.2
    # There is some byte-swapping nonsense that is poorly explained.
    # The following steps appear to correct for the "complemented" values.
    neg_voids = (elevations < -15000.0)  # Find negative voids
    elevations[neg_voids] = np.abs(elevations[neg_voids]) - 32768.0  # And fill them in (2**15 = 32768)
    pos_voids = (elevations > 15000.0)  # Find positive voids
    elevations[pos_voids] = 32768.0 - elevations[pos_voids]  # And fill them in

    return [origin, origindd, delta, deltadd, deltam], [lats_1D, lons_1D], [lats_2D,lons_2D], elevations

