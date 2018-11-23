'''
Module for reading Cosmo Skymed HDF5 imagery.  This was ported
directly from NGA's MATLAB SAR Toolbox.
'''
# SarPy imports
from .sicd import MetaNode
from . import Reader as ReaderSuper  # Reader superclass
from . import sicd
from ...geometry import geocoords as gc
from ...geometry import point_projection as point
from .utils import chipper

# Python standard library imports
import copy
import datetime
import os
import re
# External dependencies
import numpy as np
import h5py
# We prefer numpy.polynomial.polynomial over numpy.polyval/polyfit since its coefficient
# ordering is consistent with SICD, and because it supports 2D polynomials.
from numpy.polynomial import polynomial as poly
from scipy.misc import comb
from scipy.constants import speed_of_light
from scipy.optimize import fsolve

__classification__ = "UNCLASSIFIED"
__author__ = ["Jarred Barber", "Wade Schwartzkopf"]
__email__ = "jpb5082@gmail.com"


def raised_cos_fun(n, coef):
    '''
    Reproduces hamming functionality from the MATLAB Signal Processing
    Toolbox, but allows for arbitrary coefficients of raised cosine.
    '''
    w = coef - (1 - coef) * np.cos(2 * np.pi * np.arange(0, np.ceil(n / 2)) /
                                   (n - 1))
    if n % 2 == 0:
        w = [w, w[::-1]]
    else:
        w = [w, w[:-1][::-1]]
    return np.concatenate(w)


def datenum_w_frac(datestring, as_datetime=False):
    '''
    MATLAB's datenum function won't handle precise times down under a
    millisecond, because 1) It won't accept a format with more than 3 .FFF in
    the string description of the date format and 2) the resulting serial
    date number is stored in days from 00-JAN-0000 and just doesn't have
    enough bits to handle fractional seconds to the level we want.  Here we
    handle the fractional seconds separately so we can read date with the
    precision we need.
    '''
    epoch = datetime.datetime.strptime('2000-01-01 00:00:00',
                                       '%Y-%m-%d %H:%M:%S')
    if '.' in datestring:
        date, frac = datestring.split('.')
    else:
        date = datestring
        frac = '0'

    date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    datenum_s = (date - epoch).total_seconds()
    datenum_frac = float('0.' + frac)

    if np.isnan(datenum_frac):
        datenum_frac = 0

    if as_datetime:
        return date + datetime.timedelta(seconds=datenum_frac)
    else:
        return datenum_s, datenum_frac


def populate_meta(root, recurse=True):
    '''
    DFS to merge all attrs into a single dict
    '''

    def f(v):
        if isinstance(v, bytes):
            return v.decode('ascii')
        return v

    meta = {k: f(v) for k, v in root.attrs.items()}

    if recurse:
        try:
            for v in root.values():
                try:
                    meta.update(populate_meta(v))
                except:
                    # Doesn't have attrs
                    pass
        except:
            # Doesn't have values()
            pass
    return meta


def isa(filename):
    """Test to see if file is a product.xml file."""
    try:
        with h5py.File(filename, 'r') as h5:
            if 'CSK' in h5.attrs['Satellite ID'].decode('ascii'):
                return Reader
    except Exception:
        pass


class CSMChipper(chipper.Base):
    def __init__(self, filename, band, meta):
        self.filename = filename

        def complextype(data):
            return data[..., 0] + data[..., 1] * 1j

        self.band = band
        self.complextype = complextype
        self.symmetry = [False, False, True]

        with h5py.File(filename, 'r') as h5:
            lineorder = h5.attrs['Lines Order'].decode('ascii')
            columnorder = h5.attrs['Columns Order'].decode('ascii')

            key = f'S0{self.band+1}'

            self.datasize = np.flip(np.array(h5[key]['SBI'].shape[:2]),0)

        self.symmetry[0] = (columnorder != 'NEAR-FAR')
        self.symmetry[1] = (lineorder == 'EARLY-LATE') ^ (meta.SCPCOA.SideOfTrack=='R')

        # [num_dims datasize] = H5S.get_simple_extent_dims(dspace_id(1)); % All polarizations should be same size
        # % Compute symmetry
        # symmetry=[0 0 1]; % Default for CSM complex data
        # lineorder=get_hdf_attribute(fid,'Lines Order')';
        # columnorder=get_hdf_attribute(fid,'Columns Order')';
        # % Should be EARLY-LATE/NEAR-FAR for all CSM complex data
        # symmetry(1)=~strncmp(columnorder,'NEAR-FAR',8);
        # symmetry(2)=xor(strncmp(lineorder,'EARLY-LATE',10),meta{1}.SCPCOA.SideOfTrack=='R');


    def read_raw_fun(self, dim1rg, dim2rg):
        # TODO: Here we assume one band
        # Assuming spotlight SLC
        if len(dim1rg) == 2:
            dim1rg = list(dim1rg) + [1]
        if len(dim2rg) == 2:
            dim2rg = list(dim2rg) + [1]

        with h5py.File(self.filename, 'r') as h5:
            s1, e1, k1 = dim1rg
            s2, e2, k2 = dim2rg
            key = f'S0{self.band+1}'
            return h5[key]['SBI'][s2:e2:k2, s1:e1:k1, :]


class Reader(ReaderSuper):
    def __init__(self, product_filename):
        self.sicdmeta = meta2sicd(product_filename)
        self.read_chip = [CSMChipper(product_filename, band, self.sicdmeta[band]) for band in range(len(self.sicdmeta))]


def polyshift(a, shift):
    b = np.zeros(a.size)
    for j in range(1, len(a) + 1):
        for k in range(j, len(a) + 1):
            b[j - 1] = b[j - 1] + (
                a[k - 1] * comb(k - 1, j - 1) * np.power(shift, (k - j)))
    return b


def meta2sicd(filename):
    '''
    Extract the CSM metadata into SICD format
    '''
    SECONDS_IN_A_DAY = 24 * 60 * 60
    band_meta = []
    band_shapes = []
    with h5py.File(filename, 'r') as h5:
        h5meta = populate_meta(h5, recurse=False)

        # per-band data
        numbands = len(h5.keys())
        for i in range(numbands):  # "pingpong" mode has multiple polarizations
            groupname = f'/S0{i+1}'
            band_meta.append(populate_meta(h5[groupname]))
            band_shapes.append(h5[groupname]['SBI'].shape[:2])

    ## CollectionInfo
    output_meta = MetaNode()
    output_meta.CollectionInfo = MetaNode()
    output_meta.CollectionInfo.CollectorName = h5meta['Satellite ID']
    output_meta.CollectionInfo.CoreName = str(h5meta['Programmed Image ID'])
    output_meta.CollectionInfo.CollectType = 'MONOSTATIC'

    output_meta.CollectionInfo.RadarMode = MetaNode()

    if h5meta['Acquisition Mode'] in [
            'HIMAGE', 'PINGPONG', 'WIDEREGION', 'HUGEREGION'
    ]:
        output_meta.CollectionInfo.RadarMode.ModeType = 'STRIPMAP'
    else:
        # case {'ENHANCED SPOTLIGHT','SMART'} # "Spotlight"
        output_meta.CollectionInfo.RadarMode.ModeType = 'DYNAMIC STRIPMAP'

    # output_meta.CollectionInfo.RadarMode.ModeID = h5meta['Multi-Beam ID']
    output_meta.CollectionInfo.Classification = 'UNCLASSIFIED'

    ## ImageCreation
    output_meta.ImageCreation = MetaNode()
    img_create_time = datenum_w_frac(h5meta['Product Generation UTC'], True)
    output_meta.ImageCreation.DateTime = img_create_time
    output_meta.ImageCreation.Profile = 'Prototype'

    ## ImageData
    output_meta.ImageData = MetaNode()  # Just a placeholder
    # Most subfields added below in "per band" section
    # Used for computing SCP later

    ## GeoData
    if h5meta['Ellipsoid Designator'] == 'WGS84':
        output_meta.GeoData = MetaNode()
        output_meta.GeoData.EarthModel = 'WGS_84'

    # Most subfields added below in "per band" section

    ## Grid
    output_meta.Grid = MetaNode()
    if h5meta['Projection ID'] == 'SLANT RANGE/AZIMUTH':
        output_meta.Grid.ImagePlane = 'SLANT'
        output_meta.Grid.Type = 'RGZERO'
    else:
        output_meta.Grid.ImagePlane = 'GROUND'

    output_meta.Grid.Row = MetaNode()
    output_meta.Grid.Col = MetaNode()

    output_meta.Grid.Row.Sgn = -1  # Always true for CSM
    output_meta.Grid.Col.Sgn = -1  # Always true for CSM
    fc = h5meta['Radar Frequency']  # Center frequency
    output_meta.Grid.Row.KCtr = 2 * fc / speed_of_light
    output_meta.Grid.Col.KCtr = 0
    output_meta.Grid.Row.DeltaKCOAPoly = np.atleast_2d(0)
    output_meta.Grid.Row.WgtType = MetaNode()
    output_meta.Grid.Col.WgtType = MetaNode()

    output_meta.Grid.Row.WgtType.WindowName = h5meta[
        'Range Focusing Weighting Function'].rstrip().upper()
    if output_meta.Grid.Row.WgtType.WindowName == 'HAMMING':  # The usual CSM weigting
        output_meta.Grid.Row.WgtType.Parameter = MetaNode()
        output_meta.Grid.Row.WgtType.Parameter.name = 'COEFFICIENT'
        val = h5meta['Range Focusing Weighting Coefficient']
        output_meta.Grid.Row.WgtType.Parameter.value = str(val)
        output_meta.Grid.Row.WgtFunct = raised_cos_fun(512, val)

    output_meta.Grid.Col.WgtType.WindowName = h5meta[
        'Azimuth Focusing Weighting Function'].rstrip().upper()
    if output_meta.Grid.Col.WgtType.WindowName == 'HAMMING':  # The usual CSM weigting
        output_meta.Grid.Col.WgtType.Parameter = MetaNode()
        output_meta.Grid.Col.WgtType.Parameter.name = 'COEFFICIENT'
        val = h5meta['Azimuth Focusing Weighting Coefficient']
        output_meta.Grid.Col.WgtType.Parameter.value = str(val)
        output_meta.Grid.Col.WgtFunct = raised_cos_fun(512, val)

    # More subfields added below in "per band" section

    ## Timeline
    [collectStart,
     collectStartFrac] = datenum_w_frac(h5meta['Scene Sensing Start UTC'])
    [collectEnd,
     collectEndFrac] = datenum_w_frac(h5meta['Scene Sensing Stop UTC'])
    # We loose a bit of precision when assigning the SICD CollectStart
    # field, since a MATLAB serial date number just doesn't have enough
    # bits to handle the full precision given in the TSX XML. However, all
    # relative times within the SICD metadata structure will be computed at
    # full precision.
    output_meta.Timeline = MetaNode()
    output_meta.Timeline.IPP = MetaNode()
    output_meta.Timeline.IPP.Set = MetaNode()
    output_meta.Timeline.CollectStart = datenum_w_frac(
        h5meta['Scene Sensing Start UTC'], True)
    output_meta.Timeline.CollectDuration = datenum_w_frac(
        h5meta['Scene Sensing Stop UTC'], True)
    output_meta.Timeline.CollectDuration = (
        output_meta.Timeline.CollectDuration -
        output_meta.Timeline.CollectStart).total_seconds()
    # output_meta.Timeline.CollectStart = collectStart + (
    #     collectStartFrac / SECONDS_IN_A_DAY)
    # output_meta.Timeline.CollectDuration = np.round(
    #     (collectEnd - collectStart) * SECONDS_IN_A_DAY) + (
    #         collectEndFrac - collectStartFrac)  # Handle fractional seconds
    output_meta.Timeline.IPP.Set.TStart = 0
    output_meta.Timeline.IPP.Set.TEnd = 0  # Apply real value later.  Just a placeholder.
    output_meta.Timeline.IPP.Set.IPPStart = 0
    # More subfields added below in "per band" section

    ## Position
    # Compute polynomial from state vectors
    [ref_time, ref_time_frac] = datenum_w_frac(h5meta['Reference UTC'])
    # Times in SICD are with respect to time from start of collect, but
    # time in CSM are generally with respect to reference time.
    ref_time_offset = np.round(ref_time - collectStart)
    # ref_time_offset = np.round((ref_time - collectStart) *
    #                            SECONDS_IN_A_DAY)  # Convert from days to secs
    
    ref_time_offset += (ref_time_frac - collectStartFrac)  # Handle fractional seconds
    state_vector_T = h5meta['State Vectors Times']  # In seconds
    state_vector_T = state_vector_T + ref_time_offset  # Make with respect to Timeline.CollectStart
    state_vector_pos = h5meta['ECEF Satellite Position']
    # sv2poly.m shows ways to determine best polynomial order, but 5th is almost always best
    polyorder = np.minimum(5, len(state_vector_T) - 1)

    P_x = poly.polyfit(state_vector_T, state_vector_pos[:, 0], polyorder)
    P_y = poly.polyfit(state_vector_T, state_vector_pos[:, 1], polyorder)
    P_z = poly.polyfit(state_vector_T, state_vector_pos[:, 2], polyorder)

    # We don't use these since they are derivable from the position polynomial
    # state_vector_vel = h5meta['ECEF Satellite Velocity']
    # state_vector_acc = h5meta['ECEF Satellite Acceleration']
    # P_vx = polyfit(state_vector_T, state_vector_vel(1,:), polyorder)
    # P_vy = polyfit(state_vector_T, state_vector_vel(2,:), polyorder)
    # P_vz = polyfit(state_vector_T, state_vector_vel(3,:), polyorder)
    # P_ax = polyfit(state_vector_T, state_vector_acc(1,:), polyorder)
    # P_ay = polyfit(state_vector_T, state_vector_acc(2,:), polyorder)
    # P_az = polyfit(state_vector_T, state_vector_acc(3,:), polyorder)
    # Store position polynomial
    # JARRED NOTE: reverse polys?
    output_meta.Position = MetaNode()
    output_meta.Position.ARPPoly = MetaNode()
    output_meta.Position.ARPPoly.X = P_x
    output_meta.Position.ARPPoly.Y = P_y
    output_meta.Position.ARPPoly.Z = P_z

    ## RadarCollection
    pols = []
    output_meta.RadarCollection = MetaNode()
    output_meta.RadarCollection.RcvChannels = MetaNode()
    output_meta.RadarCollection.RcvChannels.ChanParameters = [
        MetaNode() for i in range(numbands)
    ]

    for i in range(numbands):
        pol = band_meta[i]['Polarisation']
        pol_fmt = pol[0:1] + ':' + pol[1:]
        output_meta.RadarCollection.RcvChannels.ChanParameters[
            i].TxRcvPolarization = pol_fmt

    output_meta.RadarCollection.TxPolarization = pol_fmt[0:1]
    # tx_pol = [pols{:}] tx_pol = unique(tx_pol(1:2:end))
    # if numel(tx_pol)==1
    #     output_meta.RadarCollection.TxPolarization = tx_pol
    # else
    #     output_meta.RadarCollection.TxPolarization = 'SEQUENCE'
    #     for i = 1:numel(tx_pol)
    #         output_meta.RadarCollection.TxSequence(i).TxStep = i
    #         output_meta.RadarCollection.TxSequence(i).TxPolarization = tx_pol(i)

    #     # Most subfields added below in "per band" section

    #     ## ImageFormation
    output_meta.ImageFormation = MetaNode()
    output_meta.ImageFormation.RcvChanProc = MetaNode()
    output_meta.ImageFormation.RcvChanProc.NumChanProc = 1
    output_meta.ImageFormation.RcvChanProc.PRFScaleFactor = 1

    # output_meta.ImageFormation.RcvChanProc = struct('NumChanProc', 1,
    #                                                 'PRFScaleFactor', 1)
    output_meta.ImageFormation.ImageFormAlgo = 'RMA'
    output_meta.ImageFormation.TStartProc = 0
    output_meta.ImageFormation.TEndProc = output_meta.Timeline.CollectDuration
    output_meta.ImageFormation.STBeamComp = 'SV'
    output_meta.ImageFormation.ImageBeamComp = 'NO'
    output_meta.ImageFormation.AzAutofocus = 'NO'
    output_meta.ImageFormation.RgAutofocus = 'NO'
    # More subfields added below in "per band" section
    output_meta.RMA = MetaNode()
    output_meta.RMA.RMAlgoType = 'OMEGA_K'
    output_meta.RMA.ImageType = 'INCA'
    output_meta.RMA.INCA = MetaNode()
    output_meta.RMA.INCA.FreqZero = fc
    # These polynomials are used later to determine RMA.INCA.DopCentroidPoly
    t_az_ref = h5meta['Azimuth Polynomial Reference Time']
    t_rg_ref = h5meta['Range Polynomial Reference Time']
    dop_poly_az = h5meta['Centroid vs Azimuth Time Polynomial']
    #     # Jarred note: do we need to strip polys?
    dop_poly_az = dop_poly_az[:(np.argwhere(dop_poly_az != 0.0)[-1, 0] + 1)]
    #     # dop_poly_az=dop_poly_az(1:find(dop_poly_az~=0,1,'last')) # Strip of zero coefficients
    dop_poly_rg = h5meta['Centroid vs Range Time Polynomial']
    dop_poly_rg = dop_poly_rg[:(np.argwhere(dop_poly_rg != 0.0)[-1, 0] + 1)]
    #     # dop_poly_rg=dop_poly_rg(1:find(dop_poly_rg~=0,1,'last'))  # Strip of zero coefficients
    dop_rate_poly_az = h5meta['Doppler Rate vs Azimuth Time Polynomial']
    dop_rate_poly_rg = h5meta['Doppler Rate vs Range Time Polynomial']
    #     # dop_rate_poly_rg=dop_rate_poly_rg(1:find(dop_rate_poly_rg~=0,1,'last'))  # Strip of zero coefficients

    ## SCPCOA
    output_meta.SCPCOA = MetaNode()
    output_meta.SCPCOA.SideOfTrack = h5meta['Look Side'][0:1].upper()
    # Most subfields added below in "per band" section, but we grab this field
    # now so we know to flip from CSM's EARLY-LATE column order to SICD's
    # view-from-above column order.

    ## Process fields specific to each polarimetric band
    band_independent_meta = copy.deepcopy(
        output_meta)  # Values that are consistent across all bands
    grouped_meta = []

    for i in range(numbands):
        output_meta = copy.deepcopy(band_independent_meta)
        ## ImageData
        datasize = band_shapes[i]  # All polarizations should be same size
        output_meta.ImageData = MetaNode()
        output_meta.ImageData.NumCols = datasize[0]
        output_meta.ImageData.NumRows = datasize[1]
        # output_meta.ImageData.FullImage = output_meta.ImageData
        output_meta.ImageData.FirstRow = 0
        output_meta.ImageData.FirstCol = 0
        output_meta.ImageData.PixelType = 'RE16I_IM16I'
        # There are many different options for picking the SCP point.  We chose
        # the point that is closest to the reference zero-doppler and range
        # times in the CSM metadata.
        t_az_first = band_meta[i]['Zero Doppler Azimuth First Time']
        # Zero doppler time of first column
        ss_az_s = band_meta[i]['Line Time Interval']
        # Image column spacing in zero doppler time (seconds)
        output_meta.ImageData.SCPPixel = MetaNode()
        output_meta.ImageData.SCPPixel.Col = int(
            np.round((t_az_ref - t_az_first) / ss_az_s) + 1)
        if output_meta.SCPCOA.SideOfTrack == 'L':
            # Order of columns in SICD goes in reverse time for left-looking
            ss_az_s = -ss_az_s
            output_meta.ImageData.SCPPixel.Col = int(
                output_meta.ImageData.NumCols -
                output_meta.ImageData.SCPPixel.Col - 1)
            # First column in SICD is actually last line in CSM terminology
            t_az_first = band_meta[i]['Zero Doppler Azimuth Last Time']

        t_rg_first = band_meta[i][
            'Zero Doppler Range First Time']  # Range time of first row

        if 'SCS' in h5meta['Product Type']:
            # 'Column Time Interval' does not exist in detected products.
            ss_rg_s = band_meta[i][
                'Column Time Interval']  # Row spacing in range time (seconds)
            output_meta.ImageData.SCPPixel.Row = int(
                round((t_rg_ref - t_rg_first) / ss_rg_s) + 1)
        else:
            raise NotImplementedError('Only complex products supported')

        # How Lockheed seems to pick the SCP:
        output_meta.ImageData.SCPPixel = MetaNode()
        output_meta.ImageData.SCPPixel.Col = datasize[0] // 2
        output_meta.ImageData.SCPPixel.Row = int(np.ceil(datasize[1] / 2) - 1)

        ## GeoData
        # Initially, we just seed this with a rough value.  Later we will put
        # in something more precise.
        latlon = band_meta[i]['Centre Geodetic Coordinates']
        output_meta.GeoData = MetaNode()
        output_meta.GeoData.SCP = MetaNode()
        output_meta.GeoData.SCP.LLH = MetaNode()
        output_meta.GeoData.SCP.ECF = MetaNode()
        output_meta.GeoData.SCP.LLH.Lat = latlon[0]
        output_meta.GeoData.SCP.LLH.Lon = latlon[1]
        # CSM generally gives HAE as zero.  Perhaps we should adjust this to DEM.
        output_meta.GeoData.SCP.LLH.HAE = latlon[2]
        ecf = gc.geodetic_to_ecf(latlon)[0]

        output_meta.GeoData.SCP.ECF.X = ecf[0]
        output_meta.GeoData.SCP.ECF.Y = ecf[1]
        output_meta.GeoData.SCP.ECF.Z = ecf[2]
        # Calling derived_sicd_fields at the end will populate these fields
        # with the sensor model, so we don't need to do it here.
        # latlon=get_hdf_attribute(dset_id(i),'Top Left Geodetic Coordinates')
        # output_meta.GeoData.ImageCorners.ICP.FRFC.Lat=latlon(1)
        # output_meta.GeoData.ImageCorners.ICP.FRFC.Lon=latlon(2)
        # latlon=get_hdf_attribute(dset_id(i),'Bottom Left Geodetic Coordinates')
        # output_meta.GeoData.ImageCorners.ICP.FRLC.Lat=latlon(1)
        # output_meta.GeoData.ImageCorners.ICP.FRLC.Lon=latlon(2)
        # latlon=get_hdf_attribute(dset_id(i),'Bottom Right Geodetic Coordinates')
        # output_meta.GeoData.ImageCorners.ICP.LRLC.Lat=latlon(1)
        # output_meta.GeoData.ImageCorners.ICP.LRLC.Lon=latlon(2)
        # latlon=get_hdf_attribute(dset_id(i),'Top Right Geodetic Coordinates')
        # output_meta.GeoData.ImageCorners.ICP.LRFC.Lat=latlon(1)
        # output_meta.GeoData.ImageCorners.ICP.LRFC.Lon=latlon(2)

        ## Grid
        # output_meta.Grid = MetaNode()
        # output_meta.Grid.Row = MetaNode()
        # output_meta.Grid.Col = MetaNode()

        output_meta.Grid.Row.SS = band_meta[i]['Column Spacing']
        #         # output_meta.Grid.Row.SS=get_hdf_attribute(dset_id(i),'Column Time Interval')*speed_of_light/2 # Exactly equivalent to above
        output_meta.Grid.Col.SS = band_meta[i]['Line Spacing']
        # How Lockheed does it:
        # output_meta.Grid.Col.SS=velocity_mag_CA_SCP * ss_az_s * DRateSFPoly(1,1)
        output_meta.Grid.Row.ImpRespBW = 2 * band_meta[i][
            'Range Focusing Bandwidth'] / speed_of_light
        dop_bw = band_meta[i][
            'Azimuth Focusing Bandwidth']  # Doppler frequency
        # Convert to azimuth spatial bandwidth (cycles per meter)
        output_meta.Grid.Col.ImpRespBW = min(
            dop_bw * abs(ss_az_s), 1
        ) / output_meta.Grid.Col.SS  # Can't have more bandwidth in data than sample spacing
        output_meta.Grid.Row.DeltaK1 = -output_meta.Grid.Row.ImpRespBW / 2
        output_meta.Grid.Row.DeltaK2 = -output_meta.Grid.Row.DeltaK1
        output_meta.Grid.Col.DeltaK1 = -(1 / output_meta.Grid.Col.SS) / 2
        output_meta.Grid.Col.DeltaK2 = -output_meta.Grid.Col.DeltaK1
        if output_meta.Grid.Row.WgtType.WindowName == 'HAMMING':  # The usual CSM weigting
            # Computation of broadening factor for uniform window is:
            # 2 * fzero(@(x) (sin(pi*x)/(pi*x)) - (1/sqrt(2)), .1)
            a = float(output_meta.Grid.Row.WgtType.Parameter.
                      value)  # Generalized Hamming window parameter

            row_broadening_factor = 2.0 * fsolve(
                lambda x: a * (np.sin(np.pi * x) / (np.pi * x)) + ((1 - a) * (np.sin(np.pi * (x - 1)) / (np.pi * (x - 1))) / 2) + ((1 - a) * (np.sin(np.pi * (x + 1)) / (np.pi * (x + 1))) / 2) - a / np.sqrt(2),
                0.1)
            output_meta.Grid.Row.ImpRespWid = row_broadening_factor / output_meta.Grid.Row.ImpRespBW

        if output_meta.Grid.Col.WgtType.WindowName == 'HAMMING':  # The usual CSM weigting
            a = float(output_meta.Grid.Col.WgtType.Parameter.
                      value)  # Generalized Hamming window parameter
            col_broadening_factor = 2.0 * fsolve(
                lambda x: a * (np.sin(np.pi * x) / (np.pi * x)) + ((1 - a) * (np.sin(np.pi * (x - 1)) / (np.pi * (x - 1))) / 2) + ((1 - a) * (np.sin(np.pi * (x + 1)) / (np.pi * (x + 1))) / 2) - a / np.sqrt(2),
                0.1)
            output_meta.Grid.Col.ImpRespWid = col_broadening_factor / output_meta.Grid.Col.ImpRespBW

        ## Timeline
        prf = band_meta[i]['PRF']
        output_meta.Timeline.IPP.Set.IPPEnd = int(
            np.floor(prf * output_meta.Timeline.CollectDuration))
        output_meta.Timeline.IPP.Set.IPPPoly = np.array([0, prf])
        output_meta.Timeline.IPP.Set.TEnd = output_meta.Timeline.CollectDuration

        ## RadarCollection
        # output_meta.RadarCollection.RefFreqIndex=uint32(0) # Absence of this field means all frequencies are true values
        chirp_length = band_meta[i]['Range Chirp Length']
        chirp_rate = abs(band_meta[i]['Range Chirp Rate'])
        bw = chirp_length * chirp_rate
        output_meta.RadarCollection.TxFrequency = MetaNode()
        output_meta.RadarCollection.TxFrequency.Min = fc - (bw / 2)
        output_meta.RadarCollection.TxFrequency.Max = fc + (bw / 2)
        output_meta.RadarCollection.Waveform = MetaNode()
        output_meta.RadarCollection.Waveform.WFParameters = MetaNode()
        output_meta.RadarCollection.Waveform.WFParameters.TxPulseLength = chirp_length
        output_meta.RadarCollection.Waveform.WFParameters.TxRFBandwidth = bw
        output_meta.RadarCollection.Waveform.WFParameters.TxFreqStart = output_meta.RadarCollection.TxFrequency.Min
        output_meta.RadarCollection.Waveform.WFParameters.TxFMRate = chirp_rate
        sample_rate = band_meta[i]['Sampling Rate']
        if np.isnan(band_meta[i]['Reference Dechirping Time']):
            output_meta.RadarCollection.Waveform.WFParameters.RcvDemodType = 'CHIRP'
            output_meta.RadarCollection.Waveform.WFParameters.RcvFMRate = 0
        else:
            output_meta.RadarCollection.Waveform.WFParameters.RcvDemodType = 'STRETCH'

        output_meta.RadarCollection.Waveform.WFParameters.RcvWindowLength = (
            band_meta[i]['Echo Sampling Window Length']) / sample_rate
        output_meta.RadarCollection.Waveform.WFParameters.ADCSampleRate = sample_rate

        ## ImageFormation
        output_meta.ImageFormation.RcvChanProc.ChanIndex = i
        output_meta.ImageFormation.TxFrequencyProc = MetaNode()
        output_meta.ImageFormation.TxFrequencyProc.MinProc = output_meta.RadarCollection.TxFrequency.Min
        output_meta.ImageFormation.TxFrequencyProc.MaxProc = output_meta.RadarCollection.TxFrequency.Max
        output_meta.ImageFormation.TxRcvPolarizationProc = output_meta.RadarCollection.RcvChannels.ChanParameters[
            i].TxRcvPolarization

        ## RMA
        t_rg_scp = t_rg_first + (
            ss_rg_s * float(output_meta.ImageData.SCPPixel.Row)
        )  # Range time to SCP
        output_meta.RMA.INCA.R_CA_SCP = t_rg_scp * speed_of_light / 2
        t_az_scp = t_az_first + (
            ss_az_s * float(output_meta.ImageData.SCPPixel.Col)
        )  # Zero doppler time of SCP
        output_meta.RMA.INCA.TimeCAPoly = np.array([
            t_az_scp + ref_time_offset,  # With respect to start of collect
            ss_az_s / output_meta.Grid.Col.SS
        ])  # Convert zero doppler spacing from sec/pixels to sec/meters
        # Compute DopCentroidPoly/DeltaKCOAPoly

        output_meta.RMA.INCA.DopCentroidPoly = np.zeros((len(dop_poly_rg),
                                                         len(dop_poly_az)))
        # Compute doppler centroid value at SCP
        output_meta.RMA.INCA.DopCentroidPoly[0] = (
            poly.polyval(t_rg_scp - t_rg_ref, dop_poly_rg) + poly.polyval(
                t_az_scp - t_az_ref,dop_poly_az) - 0.5 *
            (dop_poly_az[0] + dop_poly_rg[0]))  # These should be identical
        # Shift 1D polynomials to account for SCP
        dop_poly_az_shifted = polyshift(dop_poly_az, t_az_scp - t_az_ref)
        dop_poly_rg_shifted = polyshift(dop_poly_rg, t_rg_scp - t_rg_ref)
        dop_rate_poly_rg_shifted = polyshift(dop_rate_poly_rg,
                                             t_rg_scp - t_rg_ref)
        # Scale 1D polynomials to from Hz/s^n to Hz/m^n
        dop_poly_az_scaled = dop_poly_az_shifted * np.power(
            ss_az_s / output_meta.Grid.Col.SS, np.arange(0, len(dop_poly_az)))
        dop_poly_rg_scaled = dop_poly_rg_shifted * np.power(
            ss_rg_s / output_meta.Grid.Row.SS, np.arange(0, len(dop_poly_rg)))
        dop_rate_poly_rg_scaled = dop_rate_poly_rg_shifted * np.power(
            ss_rg_s / output_meta.Grid.Row.SS,
            np.arange(0, len(dop_rate_poly_rg)))
        output_meta.RMA.INCA.DopCentroidPoly[1:, 0] = dop_poly_rg_scaled[1:]
        output_meta.RMA.INCA.DopCentroidPoly[0, 1:] = dop_poly_az_scaled[1:]
        # output_meta.RMA.INCA.DopCentroidCOA = True
        output_meta.Grid.Col.DeltaKCOAPoly = output_meta.RMA.INCA.DopCentroidPoly * ss_az_s / output_meta.Grid.Col.SS

        # Compute DRateSFPoly
        # For the purposes of the DRateSFPoly computation, we ignore any
        # changes in velocity or doppler rate over the azimuth dimension.
        # Velocity is derivate of position.
        vel_x = poly.polyval(output_meta.RMA.INCA.TimeCAPoly[0],
                             poly.polyder(P_x))
        vel_y = poly.polyval(output_meta.RMA.INCA.TimeCAPoly[0],
                             poly.polyder(P_y))
        vel_z = poly.polyval(output_meta.RMA.INCA.TimeCAPoly[0],
                             poly.polyder(P_z))
        vm_ca_sq = vel_x**2 + vel_y**2 + vel_z**2  # Magnitude of the velocity squared
        r_ca = np.array(
            [output_meta.RMA.INCA.R_CA_SCP, 1.]
        )  # Polynomial representing range as a function of range distance from SCP
        output_meta.RMA.INCA.DRateSFPoly = -poly.polymul(
            dop_rate_poly_rg_scaled, r_ca) * (
                speed_of_light / (2.0 * fc * vm_ca_sq))  # Assumes a SGN of -1
        output_meta.RMA.INCA.DRateSFPoly = np.array(
            [output_meta.RMA.INCA.DRateSFPoly])#.transpose()
        # TimeCOAPoly
        # TimeCOAPoly=TimeCA+(DopCentroid/dop_rate)
        # Since we can't evaluate this equation analytically, we will evaluate
        # samples of it across our image and fit a 2D polynomial to it.
        # From radarsat.py
        POLY_ORDER = 2  # Order of polynomial which we want to compute in each dimension
        grid_samples = POLY_ORDER + 1
        coords_az_m = np.linspace(
            -output_meta.ImageData.SCPPixel.Col * output_meta.Grid.Col.SS,
            (output_meta.ImageData.NumCols -
             output_meta.ImageData.SCPPixel.Col) * output_meta.Grid.Col.SS,
            grid_samples)
        coords_rg_m = np.linspace(
            -output_meta.ImageData.SCPPixel.Row * output_meta.Grid.Row.SS,
            (output_meta.ImageData.NumRows -
             output_meta.ImageData.SCPPixel.Row) * output_meta.Grid.Row.SS,
            grid_samples)
        [coords_az_m_2d, coords_rg_m_2d] = np.meshgrid(coords_az_m,
                                                       coords_rg_m)
        timeca_sampled = poly.polyval2d(
            coords_rg_m_2d, coords_az_m_2d,
            np.atleast_2d(output_meta.RMA.INCA.TimeCAPoly))
        dopcentroid_sampled = poly.polyval2d(
            coords_rg_m_2d, coords_az_m_2d,
            output_meta.RMA.INCA.DopCentroidPoly)
        doprate_sampled = poly.polyval2d(
            coords_rg_m_2d, coords_az_m_2d,
            dop_rate_poly_rg_scaled[:, np.newaxis])
        timecoa_sampled = timeca_sampled + (
            dopcentroid_sampled / doprate_sampled)
        # Least squares fit for 2D polynomial
        a = np.zeros(((POLY_ORDER + 1)**2, (POLY_ORDER + 1)**2))
        for k in range(POLY_ORDER + 1):
            for j in range(POLY_ORDER + 1):
                a[:, k * (POLY_ORDER + 1) + j] = np.multiply(
                    np.power(coords_az_m_2d.flatten(), j),
                    np.power(coords_rg_m_2d.flatten(), k))
        A = np.zeros(((POLY_ORDER + 1)**2, (POLY_ORDER + 1)**2))
        for k in range((POLY_ORDER + 1)**2):
            for j in range((POLY_ORDER + 1)**2):
                A[k, j] = np.multiply(a[:, k], a[:, j]).sum()
        b_coa = [
            np.multiply(timecoa_sampled.flatten(), a[:, k]).sum()
            for k in range((POLY_ORDER + 1)**2)
        ]
        x_coa = np.linalg.solve(A, b_coa)
        output_meta.Grid.TimeCOAPoly = np.reshape(
            x_coa, (POLY_ORDER + 1, POLY_ORDER + 1))

        ## Radiometric
        output_meta.Radiometric = MetaNode()
        if h5meta['Range Spreading Loss Compensation Geometry'] != 'NONE':
            fact = h5meta['Reference Slant Range']**(
                2 * h5meta['Reference Slant Range Exponent'])
            if h5meta['Calibration Constant Compensation Flag'] == 0:
                fact = fact * (1 / (h5meta['Rescaling Factor']**2))
                fact = fact / band_meta[i]['Calibration Constant']
                output_meta.Radiometric.BetaZeroSFPoly = np.array([[fact]])
        ## GeoData
        # Now that sensor model fields have been populated, we can populate
        # GeoData.SCP more precisely.
        ecf = point.image_to_ground([
            output_meta.ImageData.SCPPixel.Row,
            output_meta.ImageData.SCPPixel.Col
        ], output_meta)[0]

        output_meta.GeoData.SCP.ECF.X = ecf[0]
        output_meta.GeoData.SCP.ECF.Y = ecf[1]
        output_meta.GeoData.SCP.ECF.Z = ecf[2]
        llh = gc.ecf_to_geodetic(ecf)[0]
        output_meta.GeoData.SCP.LLH.Lat = llh[0]
        output_meta.GeoData.SCP.LLH.Lon = llh[1]
        output_meta.GeoData.SCP.LLH.HAE = llh[2]

        ## SCPCOA
        sicd.derived_fields(output_meta)

        grouped_meta.append(output_meta)

    return grouped_meta
