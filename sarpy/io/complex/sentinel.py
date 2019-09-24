"""Module for reading Sentinel-1 data into a SICD model."""

# SarPy imports
from .sicd import MetaNode
from .utils import chipper
from . import Reader as ReaderSuper  # Reader superclass
from . import sicd
from . import tiff
from ...geometry import geocoords as gc
from ...geometry import point_projection as point
# Python standard library imports
import copy
import os
import datetime
import xml.etree.ElementTree as ET
# External dependencies
import numpy as np
# We prefer numpy.polynomial.polynomial over numpy.polyval/polyfit since its coefficient
# ordering is consistent with SICD, and because it supports 2D polynomials.
from numpy.polynomial import polynomial as poly
from scipy.interpolate import griddata
# try to import comb from scipy.special.
# If an old version of scipy is being used then import from scipy.misc
from scipy import __version__ as scipy_version
dot_locs = []
for i, version_char in enumerate(scipy_version):
    if version_char == '.':
        dot_locs.append(i)
major_version = int(scipy_version[0:dot_locs[0]])
if major_version >= 1:
    from scipy.special import comb
else:
    from scipy.misc import comb

_classification__ = "UNCLASSIFIED"
__author__ = "Daniel Haverporth"
__email__ = "Daniel.L.Haverporth@nga.mil"

DATE_FMT = '%Y-%m-%dT%H:%M:%S.%f'  # The datetime format Sentinel1 always uses


def isa(filename):
    # Test to see if file is a manifest.safe file
    try:
        ns = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
        # Parse everything else
        root_node = ET.parse(filename).getroot()
        if ((root_node.find('./metadataSection/metadataObject[@ID="platform"]/' +
                            'metadataWrap/xmlData/safe:platform/safe:familyName', ns).text ==
             'SENTINEL-1') and
            (root_node.find('./metadataSection/metadataObject[@ID="generalProductInformation"]/' +
                            'metadataWrap/xmlData/s1sarl1:standAloneProductInformation/' +
                            's1sarl1:productType', ns).text ==
             'SLC')):
            return Reader
    except Exception:
        pass


class Reader(ReaderSuper):
    """Creates a file reader object for an Sentinel Data."""
    def __init__(self, manifest_filename):
        # print('Opening Sentinel reader object.')
        # Read Sentinel Metadata from XML file first
        filesets = manifest_files(manifest_filename)
        meta_manifest = meta2sicd_manifest(manifest_filename)

        self.sicdmeta = []
        self.read_chip = []
        for current_fs in filesets:
            # There will be a set of files (product, data/tiff, noise, and
            # calibration) for each swath and polarization.  Within each of
            # these file set, there may be multiple bursts, and thus SICDs.
            basepathname = os.path.dirname(manifest_filename)
            tiff_filename = os.path.join(basepathname, current_fs['data'])
            meta_tiff = tiff.read_meta(tiff_filename)
            if (('product' in current_fs) and
               os.path.isfile(os.path.join(basepathname, current_fs['product']))):
                product_filename = os.path.join(basepathname, current_fs['product'])
                meta_product = meta2sicd_annot(product_filename)
                # Extra calibration files
                if (('calibration' in current_fs) and
                   os.path.isfile(os.path.join(basepathname, current_fs['calibration']))):
                    cal_filename = os.path.join(basepathname, current_fs['calibration'])
                    meta2sicd_cal(cal_filename, meta_product, meta_manifest)
                # Noise metadata computation
                if (('noise' in current_fs) and
                   os.path.isfile(os.path.join(basepathname, current_fs['noise']))):
                    noise_filename = os.path.join(basepathname, current_fs['noise'])
                    meta2sicd_noise(noise_filename, meta_product, meta_manifest)

                # Image data
                symmetry = (False, False, True)  # True for all Sentinel-1 data
                if len(meta_product) == 1:  # Stripmap, single burst, open entire file
                    self.read_chip.append(tiff.chipper(tiff_filename, symmetry, meta_tiff))
                else:  # Multiple bursts within a single data file
                    base_chipper = tiff.chipper(tiff_filename, symmetry, meta_tiff)
                    num_lines_burst = int(ET.parse(product_filename).getroot().find(
                        './swathTiming/linesPerBurst').text)
                    for j in range(len(meta_product)):
                        self.read_chip.append(chipper.subset(
                            base_chipper, [0, meta_tiff['ImageWidth'][0]],
                            num_lines_burst*j + np.array([0, num_lines_burst])))

                for current_mp in meta_product:
                    # Populate derived SICD fields now that all data has been read in
                    sicd.derived_fields(current_mp)

                    # Handle dual-polarization case.  Label channel number
                    # appropriately for ordering in manifest file.
                    # Should be the same for all burst in a TIFF
                    current_mp.ImageFormation.RcvChanProc.ChanIndex = 1 + \
                        [cp.TxRcvPolarization for cp in
                         meta_manifest.RadarCollection.RcvChannels.ChanParameters].index(
                         current_mp.ImageFormation.TxRcvPolarizationProc)
                    current_mp.merge(meta_manifest)

                    self.sicdmeta.append(current_mp)

                    # meta should already be set to this from meta_product:
                    # self.sicdmeta[-1].ImageData.NumRows = meta_tiff['ImageWidth'][0]
                    self.sicdmeta[-1].native = MetaNode()
                    self.sicdmeta[-1].native.tiff = meta_tiff

            else:  # No annotation metadata could be found
                self.sicdmeta.append(meta_manifest)
                self.sicdmeta[-1].ImageData = MetaNode()
                self.sicdmeta[-1].ImageData.NumCols = meta_tiff['ImageLength'][0]
                self.sicdmeta[-1].ImageData.NumRows = meta_tiff['ImageWidth'][0]
                self.sicdmeta[-1].native = MetaNode()
                self.sicdmeta[-1].native.tiff = meta_tiff


def manifest_files(filename):
    """Extract relevant filenames and relative paths for measurement and metadata files
    from a Sentinel manifest.safe file and group them together appropriately."""
    def _get_file_location(root_node, schema_type, possible_ids):
        """We want the data object that matches both the desired schema type and
        the possible ids from the relevant measurment data unit."""
        return [dataobject.find('./byteStream/fileLocation').attrib['href']  # File location
                for dataobject in [
                    root_node.find('dataObjectSection/' +
                                   'dataObject[@repID="' + schema_type + '"]/' +
                                   '[@ID="' + ids + '"]', ns)
                    for ids in possible_ids]  # Attempt to find objects for all ids
                if dataobject is not None][0]  # ids not found will be None and discarded

    # Parse namespaces
    ns = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
    # Parse everything else
    root_node = ET.parse(filename).getroot()
    files = []
    # Iterate through all of the "Measurement Data Units".  This should provide each
    # data object (measurements), together with its metadata and noise and
    # calibration files.
    for mdu in root_node.iterfind('./informationPackageMap/xfdu:contentUnit/' +
                                  'xfdu:contentUnit/[@repID="s1Level1MeasurementSchema"]', ns):
        # The dmdID references for each measurement data unit are indirect.
        # They are equivalently pointers to pointers. Not clear why it was
        # done this way, but here we get the real IDs for all  files associated
        # with this data unit.
        associated_ids = [root_node.find('./metadataSection/metadataObject[@ID="' +
                                         dmd + '"]/dataObjectPointer').attrib['dataObjectID']
                          for dmd in mdu.attrib['dmdID'].split()]
        fnames = dict()
        # Find data ("measurement") file itself
        fnames['data'] = _get_file_location(
            root_node, 's1Level1MeasurementSchema',
            [mdu.find('./dataObjectPointer').attrib['dataObjectID']])
        # Find all metadata files
        fnames['product'] = _get_file_location(
            root_node, 's1Level1ProductSchema', associated_ids)
        fnames['noise'] = _get_file_location(
            root_node, 's1Level1NoiseSchema', associated_ids)
        fnames['calibration'] = _get_file_location(
            root_node, 's1Level1CalibrationSchema', associated_ids)
        files.append(fnames)
    return files


def meta2sicd_manifest(filename):
    # Parse namespaces
    ns = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
    # Parse everything else
    root_node = ET.parse(filename).getroot()

    manifest = MetaNode()
    # CollectionInfo
    platform = root_node.find('./metadataSection/' +
                              'metadataObject[@ID="platform"]/' +
                              'metadataWrap/' +
                              'xmlData/' +
                              'safe:platform', ns)
    manifest.CollectionInfo = MetaNode()
    manifest.CollectionInfo.CollectorName = (platform.find('./safe:familyName', ns).text +
                                             platform.find('./safe:number', ns).text)
    manifest.CollectionInfo.RadarMode = MetaNode()
    manifest.CollectionInfo.RadarMode.ModeID = platform.find(
        './safe:instrument/safe:extension/s1sarl1:instrumentMode/s1sarl1:mode', ns).text
    if manifest.CollectionInfo.RadarMode.ModeID == 'SM':
        manifest.CollectionInfo.RadarMode.ModeType = 'STRIPMAP'
    else:
        # Actually TOPSAR.  Not what we normally think of for DYNAMIC STRIPMAP,
        # but it is definitely not SPOTLIGHT, and doesn't seem to be regular
        # STRIPMAP either.
        manifest.CollectionInfo.RadarMode.ModeType = 'DYNAMIC STRIPMAP'
    # Image Creation
    processing = root_node.find('./metadataSection/' +
                                'metadataObject[@ID="processing"]/' +
                                'metadataWrap/' +
                                'xmlData/' +
                                'safe:processing', ns)
    facility = processing.find('safe:facility', ns)
    software = facility.find('./safe:software', ns)
    manifest.ImageCreation = MetaNode()
    manifest.ImageCreation.Application = software.attrib['name'] + ' ' + software.attrib['version']
    manifest.ImageCreation.DateTime = datetime.datetime.strptime(
        processing.attrib['stop'], DATE_FMT)
    manifest.ImageCreation.Site = (facility.attrib['name'] + ', ' +
                                   facility.attrib['site'] + ', ' +
                                   facility.attrib['country'])
    manifest.ImageCreation.Profile = 'Prototype'
    # RadarCollection
    manifest.RadarCollection = MetaNode()
    manifest.RadarCollection.RcvChannels = MetaNode()
    manifest.RadarCollection.RcvChannels.ChanParameters = []
    for current_pol in root_node.findall('./metadataSection/' +
                                         'metadataObject[@ID="generalProductInformation"]/' +
                                         'metadataWrap/' +
                                         'xmlData/' +
                                         's1sarl1:standAloneProductInformation/' +
                                         's1sarl1:transmitterReceiverPolarisation', ns):
        manifest.RadarCollection.RcvChannels.ChanParameters.append(MetaNode())
        manifest.RadarCollection.RcvChannels.ChanParameters[-1].TxRcvPolarization = \
            current_pol.text[0] + ':' + current_pol.text[1]
    return(manifest)


def meta2sicd_annot(filename):
    def _polyshift(a, shift):
        b = np.zeros(a.size)
        for j in range(1, len(a)+1):
            for k in range(j, len(a)+1):
                b[j-1] = b[j-1] + (a[k-1]*comb(k-1, j-1)*np.power(shift, (k-j)))
        return b

    # Setup constants
    C = 299792458.
    # Parse annotation XML (no namespace to worry about)
    root_node = ET.parse(filename).getroot()

    common_meta = MetaNode()
    # CollectionInfo
    common_meta.CollectionInfo = MetaNode()
    common_meta.CollectionInfo.CollectorName = root_node.find('./adsHeader/missionId').text
    common_meta.CollectionInfo.CollectType = 'MONOSTATIC'
    common_meta.CollectionInfo.RadarMode = MetaNode()
    common_meta.CollectionInfo.RadarMode.ModeID = root_node.find('./adsHeader/mode').text
    if common_meta.CollectionInfo.RadarMode.ModeID[0] == 'S':
        common_meta.CollectionInfo.RadarMode.ModeType = 'STRIPMAP'
    else:
        # Actually TOPSAR.  Not what we normally think of for DYNAMIC STRIPMAP,
        # but it is definitely not SPOTLIGHT (actually counter to the spotlight
        # beam motion), and it isn't STRIPMAP with a constant angle between the
        # beam and direction of travel either, so we use DYNAMIC STRIPMAP as a
        # catchall.
        common_meta.CollectionInfo.RadarMode.ModeType = 'DYNAMIC STRIPMAP'
    common_meta.CollectionInfo.Classification = 'UNCLASSIFIED'

    # ImageData
    common_meta.ImageData = MetaNode()
    # For SLC, the following test should always hold true:
    if root_node.find('./imageAnnotation/imageInformation/pixelValue').text == 'Complex':
        common_meta.ImageData.PixelType = 'RE16I_IM16I'
    else:  # This code only handles SLC
        raise(ValueError('SLC data should be 16-bit complex.'))
    burst_list = root_node.findall('./swathTiming/burstList/burst')
    if burst_list:
        numbursts = len(burst_list)
    else:
        numbursts = 0
    # These two definitions of NumRows should always be the same for
    # non-STRIPMAP data (For STRIPMAP, samplesPerBurst is set to zero.)  Number
    # of rows in burst should be the same as the full image.  Both of these
    # numbers also should match the ImageWidth field of the measurement TIFF.
    # The NumCols definition will be different for TOPSAR/STRIPMAP.  Since each
    # burst is its own coherent data period, and thus SICD, we set the SICD
    # metadata to describe each individual burst.
    if numbursts > 0:  # TOPSAR
        common_meta.ImageData.NumRows = int(root_node.find('./swathTiming/samplesPerBurst').text)
        # Ths in the number of columns in a single burst.
        common_meta.ImageData.NumCols = int(root_node.find('./swathTiming/linesPerBurst').text)
    else:  # STRIPMAP
        common_meta.ImageData.NumRows = int(root_node.find(
            './imageAnnotation/imageInformation/numberOfSamples').text)
        # This in the number of columns in the full TIFF measurement file, even
        # if it contains multiple bursts.
        common_meta.ImageData.NumCols = int(root_node.find(
            './imageAnnotation/imageInformation/numberOfLines').text)
    common_meta.ImageData.FirstRow = 0
    common_meta.ImageData.FirstCol = 0
    common_meta.ImageData.FullImage = MetaNode()
    common_meta.ImageData.FullImage.NumRows = common_meta.ImageData.NumRows
    common_meta.ImageData.FullImage.NumCols = common_meta.ImageData.NumCols
    # SCP pixel within entire TIFF
    # Note: numpy round behaves differently than python round and MATLAB round,
    # so we avoid it here.
    center_cols = np.ceil((0.5 + np.arange(max(numbursts, 1))) *
                          float(common_meta.ImageData.NumCols))-1
    center_rows = round(float(common_meta.ImageData.NumRows)/2.-1.) * \
        np.ones_like(center_cols)
    # SCP pixel within single burst image is the same for all burst, since east
    # burst is the same size
    common_meta.ImageData.SCPPixel = MetaNode()
    common_meta.ImageData.SCPPixel.Col = int(center_cols[0])
    common_meta.ImageData.SCPPixel.Row = int(center_rows[0])

    # GeoData
    common_meta.GeoData = MetaNode()
    common_meta.GeoData.EarthModel = 'WGS_84'
    # Initially, we just seed the SCP coordinate with a rough value.  Later
    # we will put in something more precise.
    geo_grid_point_list = root_node.findall(
        './geolocationGrid/geolocationGridPointList/geolocationGridPoint')
    scp_col, scp_row, x, y, z = [], [], [], [], []
    for grid_point in geo_grid_point_list:
        scp_col.append(float(grid_point.find('./line').text))
        scp_row.append(float(grid_point.find('./pixel').text))
        lat = float(grid_point.find('./latitude').text)
        lon = float(grid_point.find('./longitude').text)
        hgt = float(grid_point.find('./height').text)
        # Can't interpolate across international date line -180/180 longitude,
        # so move to ECF space from griddata interpolation
        ecf = gc.geodetic_to_ecf((lat, lon, hgt))
        x.append(ecf[0, 0])
        y.append(ecf[0, 1])
        z.append(ecf[0, 2])
    row_col = np.vstack((scp_col, scp_row)).transpose()
    center_row_col = np.vstack((center_cols, center_rows)).transpose()
    scp_x = griddata(row_col, x, center_row_col)
    scp_y = griddata(row_col, y, center_row_col)
    scp_z = griddata(row_col, z, center_row_col)

    # Grid
    common_meta.Grid = MetaNode()
    if root_node.find('./generalAnnotation/productInformation/projection').text == 'Slant Range':
        common_meta.Grid.ImagePlane = 'SLANT'
    common_meta.Grid.Type = 'RGZERO'
    delta_tau_s = 1./float(root_node.find(
        './generalAnnotation/productInformation/rangeSamplingRate').text)
    common_meta.Grid.Row = MetaNode()
    common_meta.Grid.Col = MetaNode()
    # Range Processing
    range_proc = root_node.find('./imageAnnotation/processingInformation' +
                                '/swathProcParamsList/swathProcParams/rangeProcessing')
    common_meta.Grid.Row.SS = (C/2.) * delta_tau_s
    common_meta.Grid.Row.Sgn = -1
    # Justification for Sgn:
    # 1) "Sentinel-1 Level 1 Detailed Algorithm Definition" shows last step in
    # image formation as IFFT, which would mean a forward FFT (-1 Sgn) would be
    # required to transform back.
    # 2) The forward FFT of a sliding window shows the Doppler centroid
    # increasing as you move right in the image, which must be the case for the
    # TOPSAR collection mode which starts in a rear squint and transitions to a
    # forward squint (and are always right looking).
    fc = float(root_node.find('./generalAnnotation/productInformation/radarFrequency').text)
    common_meta.Grid.Row.KCtr = 2.*fc / C
    common_meta.Grid.Row.DeltaKCOAPoly = np.atleast_2d(0)
    common_meta.Grid.Row.ImpRespBW = 2.*float(range_proc.find('./processingBandwidth').text)/C
    common_meta.Grid.Row.WgtType = MetaNode()
    common_meta.Grid.Row.WgtType.WindowName = range_proc.find('./windowType').text.upper()
    if (common_meta.Grid.Row.WgtType.WindowName == 'NONE'):
        common_meta.Grid.Row.WgtType.WindowName = 'UNIFORM'
    elif (common_meta.Grid.Row.WgtType.WindowName == 'HAMMING'):
        # The usual Sentinel weighting
        common_meta.Grid.Row.WgtType.Parameter = MetaNode()
        # Generalized Hamming window parameter
        common_meta.Grid.Row.WgtType.Parameter.name = 'COEFFICIENT'
        common_meta.Grid.Row.WgtType.Parameter.value = range_proc.find('./windowCoefficient').text
    # Azimuth Processing
    az_proc = root_node.find('./imageAnnotation/processingInformation' +
                             '/swathProcParamsList/swathProcParams/azimuthProcessing')
    common_meta.Grid.Col.SS = float(root_node.find(
        './imageAnnotation/imageInformation/azimuthPixelSpacing').text)
    common_meta.Grid.Col.Sgn = -1  # Must be the same as Row.Sgn
    common_meta.Grid.Col.KCtr = 0
    dop_bw = float(az_proc.find('./processingBandwidth').text)  # Doppler bandwidth
    # Image column spacing in zero doppler time (seconds)
    # Sentinel-1 is always right-looking, so should always be positive
    ss_zd_s = float(root_node.find('./imageAnnotation/imageInformation/azimuthTimeInterval').text)
    # Convert to azimuth spatial bandwidth (cycles per meter)
    common_meta.Grid.Col.ImpRespBW = dop_bw*ss_zd_s/common_meta.Grid.Col.SS
    common_meta.Grid.Col.WgtType = MetaNode()
    common_meta.Grid.Col.WgtType.WindowName = az_proc.find('./windowType').text.upper()
    if (common_meta.Grid.Col.WgtType.WindowName == 'NONE'):
        common_meta.Grid.Col.WgtType.WindowName = 'UNIFORM'
    elif (common_meta.Grid.Row.WgtType.WindowName == 'HAMMING'):
        # The usual Sentinel weighting
        common_meta.Grid.Col.WgtType.Parameter = MetaNode()
        # Generalized Hamming window parameter
        common_meta.Grid.Col.WgtType.Parameter.name = 'COEFFICIENT'
        common_meta.Grid.Col.WgtType.Parameter.value = az_proc.find('./windowCoefficient').text
    # We will compute Grid.Col.DeltaKCOAPoly separately per burst later.
    # Grid.Row/Col.DeltaK1/2, WgtFunct, ImpRespWid will be computed later in sicd.derived_fields

    # Timeline
    prf = float(root_node.find(
        './generalAnnotation/downlinkInformationList/downlinkInformation/prf').text)
    common_meta.Timeline = MetaNode()
    common_meta.Timeline.IPP = MetaNode()
    common_meta.Timeline.IPP.Set = MetaNode()
    # Because of calibration pulses, it is unlikely this PRF was maintained
    # through this entire period, but we don't currently include that detail.
    common_meta.Timeline.IPP.Set.IPPPoly = np.array([0, prf])
    # Always the left-most SICD column (of first bursts or entire STRIPMAP dataset),
    # since Sentinel-1 is always right-looking.
    azimuth_time_first_line = datetime.datetime.strptime(root_node.find(
        './imageAnnotation/imageInformation/productFirstLineUtcTime').text, DATE_FMT)
    # Offset in zero Doppler time from first column to SCP column
    eta_mid = ss_zd_s * float(common_meta.ImageData.SCPPixel.Col)

    # Position
    orbit_list = root_node.findall('./generalAnnotation/orbitList/orbit')
    # For ARP vector calculation later on
    state_vector_T, state_vector_X, state_vector_Y, state_vector_Z = [], [], [], []
    for orbit in orbit_list:
        state_vector_T.append(datetime.datetime.strptime(orbit.find('./time').text, DATE_FMT))
        state_vector_X.append(float(orbit.find('./position/x').text))
        state_vector_Y.append(float(orbit.find('./position/y').text))
        state_vector_Z.append(float(orbit.find('./position/z').text))
    # We could also have used external orbit file here, instead of orbit state fields
    # in SLC annotation file.

    # RadarCollection
    pol = root_node.find('./adsHeader/polarisation').text
    common_meta.RadarCollection = MetaNode()
    common_meta.RadarCollection.TxPolarization = pol[0]
    common_meta.RadarCollection.TxFrequency = MetaNode()
    common_meta.RadarCollection.Waveform = MetaNode()
    common_meta.RadarCollection.TxFrequency.Min = fc + float(root_node.find(
        './generalAnnotation/downlinkInformationList/downlinkInformation/' +
        'downlinkValues/txPulseStartFrequency').text)
    wfp_common = MetaNode()
    wfp_common.TxFreqStart = common_meta.RadarCollection.TxFrequency.Min
    wfp_common.TxPulseLength = float(root_node.find(
        './generalAnnotation/downlinkInformationList/downlinkInformation/' +
        'downlinkValues/txPulseLength').text)
    wfp_common.TxFMRate = float(root_node.find(
        './generalAnnotation/downlinkInformationList/downlinkInformation/' +
        'downlinkValues/txPulseRampRate').text)
    bw = wfp_common.TxPulseLength * wfp_common.TxFMRate
    common_meta.RadarCollection.TxFrequency.Max = \
        common_meta.RadarCollection.TxFrequency.Min + bw
    wfp_common.TxRFBandwidth = bw
    wfp_common.RcvDemodType = 'CHIRP'
    # RcvFMRate = 0 for RcvDemodType='CHIRP'
    wfp_common.RcvFMRate = 0
    wfp_common.ADCSampleRate = float(root_node.find(
        './generalAnnotation/productInformation/rangeSamplingRate').text)  # Raw not decimated
    # After decimation would be:
    # wfp_common.ADCSampleRate = \
    #     product/generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/rangeDecimation/samplingFrequencyAfterDecimation
    # We could have multiple receive window lengths across the collect
    swl_list = root_node.findall('./generalAnnotation/downlinkInformationList/' +
                                 'downlinkInformation/downlinkValues/swlList/swl')
    common_meta.RadarCollection.Waveform.WFParameters = []
    for swl in swl_list:
        common_meta.RadarCollection.Waveform.WFParameters.append(copy.deepcopy(wfp_common))
        common_meta.RadarCollection.Waveform.WFParameters[-1].RcvWindowLength = \
            float(swl.find('./value').text)

    # ImageFormation
    common_meta.ImageFormation = MetaNode()
    common_meta.ImageFormation.RcvChanProc = MetaNode()
    common_meta.ImageFormation.RcvChanProc.NumChanProc = 1
    common_meta.ImageFormation.RcvChanProc.PRFScaleFactor = 1
    # RcvChanProc.ChanIndex must be populated external to this since it depends
    # on how the polarization were ordered in manifest file.
    common_meta.ImageFormation.TxRcvPolarizationProc = pol[0] + ':' + pol[1]
    # Assume image formation uses all data
    common_meta.ImageFormation.TStartProc = 0
    common_meta.ImageFormation.TxFrequencyProc = MetaNode()
    common_meta.ImageFormation.TxFrequencyProc.MinProc = \
        common_meta.RadarCollection.TxFrequency.Min
    common_meta.ImageFormation.TxFrequencyProc.MaxProc = \
        common_meta.RadarCollection.TxFrequency.Max
    common_meta.ImageFormation.ImageFormAlgo = 'RMA'
    # From the Sentinel-1 Level 1 Detailed Algorithm Definition document
    if common_meta.CollectionInfo.RadarMode.ModeID[0] == 'S':  # stripmap mode
        common_meta.ImageFormation.STBeamComp = 'NO'
    else:
        common_meta.ImageFormation.STBeamComp = 'SV'  # TOPSAR Mode
    common_meta.ImageFormation.ImageBeamComp = 'NO'
    common_meta.ImageFormation.AzAutofocus = 'NO'
    common_meta.ImageFormation.RgAutofocus = 'NO'

    # RMA
    # "Sentinel-1 Level 1 Detailed Algorithm Definition" document seems to most
    # closely match the RangeDoppler algorithm (with accurate secondary range
    # compression or "option 2" as described in the Cumming and Wong book).
    common_meta.RMA = MetaNode()
    common_meta.RMA.RMAlgoType = 'RG_DOP'
    common_meta.RMA.ImageType = 'INCA'
    # tau_0 is notation from ESA deramping paper
    tau_0 = float(root_node.find('./imageAnnotation/imageInformation/slantRangeTime').text)
    common_meta.RMA.INCA = MetaNode()
    common_meta.RMA.INCA.R_CA_SCP = ((C/2.) *
                                     (tau_0 +
                                      (float(common_meta.ImageData.SCPPixel.Row) *
                                       delta_tau_s)))
    common_meta.RMA.INCA.FreqZero = fc
    # If we use the Doppler Centroid as defined directly in the manifest.safe
    # metadata, then the center of frequency support Col.DeltaKCOAPoly does not
    # correspond to RMA.INCA.DopCentroidPoly.  However, we will compute
    # TimeCOAPoly later to match a newly computed Doppler Centroid based off of
    # DeltaKCOAPoly, assuming that the the COA is at the peak signal (fdop_COA
    # = fdop_DC).
    common_meta.RMA.INCA.DopCentroidCOA = True
    # Doppler Centroid
    # Get common (non-burst specific) parameters we will need for Doppler
    # centroid and rate computations later
    dc_estimate_list = root_node.findall('./dopplerCentroid/dcEstimateList/dcEstimate')
    dc_az_time, dc_t0, data_dc_poly = [], [], []
    for dc_estimate in dc_estimate_list:
        dc_az_time.append(datetime.datetime.strptime(
            dc_estimate.find('./azimuthTime').text, DATE_FMT))
        dc_t0.append(float(dc_estimate.find('./t0').text))
        data_dc_poly.append(np.fromstring(dc_estimate.find('./dataDcPolynomial').text, sep=' '))
    azimuth_fm_rate_list = root_node.findall(
        './generalAnnotation/azimuthFmRateList/azimuthFmRate')
    az_t, az_t0, k_a_poly = [], [], []
    for az_fm_rate in azimuth_fm_rate_list:
        az_t.append(datetime.datetime.strptime(
            az_fm_rate.find('./azimuthTime').text, DATE_FMT))
        az_t0.append(float(az_fm_rate.find('./t0').text))
        # Two different ways we have seen in XML for storing the FM Rate polynomial
        try:
            k_a_poly.append(np.fromstring(az_fm_rate.find(
                './azimuthFmRatePolynomial').text, sep=' '))
        except TypeError:  # old annotation xml file format
            k_a_poly.append(np.array([float(az_fm_rate.find('./c0').text),
                                      float(az_fm_rate.find('./c1').text),
                                      float(az_fm_rate.find('./c2').text)]))
    # Azimuth steering rate (constant, not dependent on burst or range)
    k_psi = float(root_node.find(
        './generalAnnotation/productInformation/azimuthSteeringRate').text)
    k_psi = k_psi*np.pi/180.  # Convert from degrees/sec into radians/sec

    # Compute per/burst metadata
    sicd_meta = []
    for count in range(max(numbursts, 1)):
        burst_meta = copy.deepcopy(common_meta)
        # Collection Info
        # Sensor specific portions of metadata
        sliceNumber = root_node.find('./imageAnnotation/imageInformation/sliceNumber')
        if sliceNumber is not None:
            sliceNumber = sliceNumber.text
        else:
            sliceNumber = 0
        swath = root_node.find('./adsHeader/swath').text
        burst_meta.CollectionInfo.Parameter = [MetaNode()]
        burst_meta.CollectionInfo.Parameter[0].name = 'SLICE'
        burst_meta.CollectionInfo.Parameter[0].value = sliceNumber
        burst_meta.CollectionInfo.Parameter.append(MetaNode())
        burst_meta.CollectionInfo.Parameter[1].name = 'SWATH'
        burst_meta.CollectionInfo.Parameter[1].value = swath
        burst_meta.CollectionInfo.Parameter.append(MetaNode())
        burst_meta.CollectionInfo.Parameter[2].name = 'BURST'
        burst_meta.CollectionInfo.Parameter[2].value = str(count+1)
        burst_meta.CollectionInfo.Parameter.append(MetaNode())
        burst_meta.CollectionInfo.Parameter[3].name = 'ORBIT_SOURCE'
        burst_meta.CollectionInfo.Parameter[3].value = 'SLC_INTERNAL'  # No external orbit file
        # Image Data.ValidData
        # Get valid bounds of burst from metadata.  Assume a rectangular valid
        # area-- not totally true, but all that seems to be defined by the
        # product XML metadata.
        if numbursts > 0:  # Valid data does not seem to be defined for STRIPMAP data
            burst = root_node.find('./swathTiming/burstList/burst[' + str(count+1) + ']')
            xml_first_cols = np.fromstring(burst.find('./firstValidSample').text, sep=' ')
            xml_last_cols = np.fromstring(burst.find('./lastValidSample').text, sep=' ')
            valid_cols = np.where((xml_first_cols >= 0) & (xml_last_cols >= 0))[0]
            first_row = int(min(xml_first_cols[valid_cols]))
            last_row = int(max(xml_last_cols[valid_cols]))
            # From SICD spec: Vertices ordered clockwise with vertex 1
            # determined by: (1) minimum row index, (2) minimum column index if
            # 2 vertices exist with minimum row index.
            burst_meta.ImageData.ValidData = MetaNode()
            burst_meta.ImageData.ValidData.Vertex = [MetaNode()]
            burst_meta.ImageData.ValidData.Vertex[0].Row = first_row
            burst_meta.ImageData.ValidData.Vertex[0].Col = int(valid_cols[0])
            burst_meta.ImageData.ValidData.Vertex.append(MetaNode())
            burst_meta.ImageData.ValidData.Vertex[1].Row = first_row
            burst_meta.ImageData.ValidData.Vertex[1].Col = int(valid_cols[-1])
            burst_meta.ImageData.ValidData.Vertex.append(MetaNode())
            burst_meta.ImageData.ValidData.Vertex[2].Row = last_row
            burst_meta.ImageData.ValidData.Vertex[2].Col = int(valid_cols[-1])
            burst_meta.ImageData.ValidData.Vertex.append(MetaNode())
            burst_meta.ImageData.ValidData.Vertex[3].Row = last_row
            burst_meta.ImageData.ValidData.Vertex[3].Col = int(valid_cols[0])
        else:
            burst_meta.ImageData.ValidData = MetaNode()
            burst_meta.ImageData.ValidData.Vertex = [MetaNode()]
            burst_meta.ImageData.ValidData.Vertex[0].Row = 0
            burst_meta.ImageData.ValidData.Vertex[0].Col = 0
            burst_meta.ImageData.ValidData.Vertex.append(MetaNode())
            burst_meta.ImageData.ValidData.Vertex[1].Row = 0
            burst_meta.ImageData.ValidData.Vertex[1].Col = int(common_meta.ImageData.NumCols)
            burst_meta.ImageData.ValidData.Vertex.append(MetaNode())
            burst_meta.ImageData.ValidData.Vertex[2].Row = int(common_meta.ImageData.NumRows)
            burst_meta.ImageData.ValidData.Vertex[2].Col = int(common_meta.ImageData.NumCols)
            burst_meta.ImageData.ValidData.Vertex.append(MetaNode())
            burst_meta.ImageData.ValidData.Vertex[3].Row = int(common_meta.ImageData.NumRows)
            burst_meta.ImageData.ValidData.Vertex[3].Col = 0
        # Timeline
        if numbursts > 0:
            # This is the first and last zero doppler times of the columns in
            # the burst.  This isn't really what we mean by CollectStart and
            # CollectDuration in SICD (really we want first and last pulse
            # times), but its all we have.
            start = datetime.datetime.strptime(burst.find('./azimuthTime').text, DATE_FMT)
            first_line_relative_start = 0  # CollectStart is zero Doppler time of first column
        else:
            start = datetime.datetime.strptime(root_node.find(
                './generalAnnotation/downlinkInformationList/downlinkInformation/' +
                'firstLineSensingTime').text, DATE_FMT)
            stop = datetime.datetime.strptime(root_node.find(
                './generalAnnotation/downlinkInformationList/downlinkInformation/' +
                'lastLineSensingTime').text, DATE_FMT)
            # Maybe CollectStart/CollectDuration should be set by
            # product/imageAnnotation/imageInformation/productFirstLineUtcTime
            # and productLastLineUtcTime.  This would make it consistent with
            # non-stripmap which just defines first and last zero doppler
            # times, but is not really consistent with what SICD generally
            # means by CollectStart/CollectDuration.
            burst_meta.Timeline.CollectStart = start
            burst_meta.Timeline.CollectDuration = (stop-start).total_seconds()
            first_line_relative_start = (azimuth_time_first_line-start).total_seconds()
        # After we have start_s, we can generate CoreName
        burst_meta.CollectionInfo.CoreName = (
            # Prefix with the NGA CoreName standard format
            start.strftime('%d%b%y') + common_meta.CollectionInfo.CollectorName +
            # The following core name is unique within all Sentinel-1 coherent data periods:
            root_node.find('./adsHeader/missionDataTakeId').text + '_' +
            ('%02d' % int(sliceNumber)) + '_' + swath + '_' + ('%02d' % (count+1)))
        # Position
        # Polynomial is computed with respect to time from start of burst
        state_vector_T_burst = np.array([(t-start).total_seconds() for t in state_vector_T])
        # Some datasets don't include enough state vectors for 5th order fit
        # One could find the order of polynomial that most accurately describes
        # this position, but use velocity as cross-validation so that the data
        # is not being overfit.  Orders over 5 often become badly conditioned
        polyorder = 5
        burst_meta.Position = MetaNode()
        burst_meta.Position.ARPPoly = MetaNode()
        burst_meta.Position.ARPPoly.X = poly.polyfit(state_vector_T_burst,
                                                     state_vector_X, polyorder)
        burst_meta.Position.ARPPoly.Y = poly.polyfit(state_vector_T_burst,
                                                     state_vector_Y, polyorder)
        burst_meta.Position.ARPPoly.Z = poly.polyfit(state_vector_T_burst,
                                                     state_vector_Z, polyorder)
        # RMA (still in for statement for each burst)
        # Sentinel-1 is always right-looking, so TimeCAPoly should never have
        # to be "flipped" for left-looking cases.
        burst_meta.RMA.INCA.TimeCAPoly = np.array([first_line_relative_start + eta_mid,
                                                   ss_zd_s/float(common_meta.Grid.Col.SS)])
        # Doppler Centroid
        # We choose the single Doppler centroid polynomial closest to the
        # center of the current burst.
        dc_est_times = np.array([(t - start).total_seconds() for t in dc_az_time])
        dc_poly_ind = np.argmin(abs(dc_est_times - burst_meta.RMA.INCA.TimeCAPoly[0]))
        # Shift polynomial from origin at dc_t0 (reference time for Sentinel
        # polynomial) to SCP time (reference time for SICD polynomial)
        range_time_scp = (common_meta.RMA.INCA.R_CA_SCP * 2)/C
        # The Doppler centroid field in the Sentinel-1 metadata is not
        # complete, so we cannot use it directly.  That description of Doppler
        # centroid by itself does not vary by azimuth although the
        # Col.DeltaKCOAPoly we see in the data definitely does. We will define
        # DopCentroidPoly differently later down in the code.
        # Doppler rate
        # Total Doppler rate is a combination of the Doppler FM rate and the
        # Doppler rate introduced by the scanning of the antenna.
        # We pick a single velocity magnitude at closest approach to represent
        # the entire burst.  This is valid, since the magnitude of the velocity
        # changes very little.
        vm_ca = np.linalg.norm([  # Magnitude of the velocity at SCP closest approach
            poly.polyval(burst_meta.RMA.INCA.TimeCAPoly[0],  # Velocity in X
                         poly.polyder(burst_meta.Position.ARPPoly.X)),
            poly.polyval(burst_meta.RMA.INCA.TimeCAPoly[0],  # Velocity in Y
                         poly.polyder(burst_meta.Position.ARPPoly.Y)),
            poly.polyval(burst_meta.RMA.INCA.TimeCAPoly[0],  # Velocity in Z
                         poly.polyder(burst_meta.Position.ARPPoly.Z))])
        # Compute FM Doppler Rate, k_a
        # We choose the single azimuth FM rate polynomial closest to the
        # center of the current burst.
        az_rate_times = np.array([(t - start).total_seconds() for t in az_t])
        az_rate_poly_ind = np.argmin(abs(az_rate_times - burst_meta.RMA.INCA.TimeCAPoly[0]))
        # SICD's Doppler rate seems to be FM Doppler rate, not total Doppler rate
        # Shift polynomial from origin at az_t0 (reference time for Sentinel
        # polynomial) to SCP time (reference time for SICD polynomial)
        DR_CA = _polyshift(k_a_poly[az_rate_poly_ind],
                           range_time_scp - az_t0[az_rate_poly_ind])
        # Scale 1D polynomial to from Hz/s^n to Hz/m^n
        DR_CA = DR_CA * ((2./C)**np.arange(len(DR_CA)))
        r_ca = np.array([common_meta.RMA.INCA.R_CA_SCP, 1])
        # RMA.INCA.DRateSFPoly is a function of Doppler rate.
        burst_meta.RMA.INCA.DRateSFPoly = (- np.convolve(DR_CA, r_ca) *  # Assumes a SGN of -1
                                           (C / (2 * fc * np.power(vm_ca, 2))))
        burst_meta.RMA.INCA.DRateSFPoly = burst_meta.RMA.INCA.DRateSFPoly[:, np.newaxis]
        # TimeCOAPoly
        # TimeCOAPoly = TimeCA + (DopCentroid/dop_rate);  # True if DopCentroidCOA = true
        # Since we don't know how to evaluate this equation analytically, we
        # could evaluate samples of it across our image and fit a 2D polynomial
        # to it later.
        POLY_ORDER = 2
        grid_samples = POLY_ORDER + 1
        cols = np.around(np.linspace(0, common_meta.ImageData.NumCols-1,
                                     num=grid_samples)).astype(int)
        rows = np.around(np.linspace(0, common_meta.ImageData.NumRows-1,
                                     num=grid_samples)).astype(int)
        coords_az_m = (cols - common_meta.ImageData.SCPPixel.Col).astype(float) *\
            common_meta.Grid.Col.SS
        coords_rg_m = (rows - common_meta.ImageData.SCPPixel.Row).astype(float) *\
            common_meta.Grid.Row.SS
        timeca_sampled = poly.polyval(coords_az_m, burst_meta.RMA.INCA.TimeCAPoly)
        doprate_sampled = poly.polyval(coords_rg_m, DR_CA)
        # Grid.Col.DeltaKCOAPoly
        # Reference: Definition of the TOPS SLC deramping function for products
        # generated by the S-1 IPF, COPE-GSEG-EOPG-TN-14-0025
        tau = tau_0 + delta_tau_s * np.arange(0, int(common_meta.ImageData.NumRows))
        # The vm_ca used here is slightly different than the ESA deramp
        # document, since the document interpolates the velocity values given
        # rather than the position values, which is what we do here.
        k_s = (2. * (vm_ca / C)) * fc * k_psi
        k_a = poly.polyval(tau - az_t0[az_rate_poly_ind], k_a_poly[az_rate_poly_ind])
        k_t = (k_a * k_s)/(k_a - k_s)
        f_eta_c = poly.polyval(tau - dc_t0[dc_poly_ind], data_dc_poly[dc_poly_ind])
        eta = ((-float(common_meta.ImageData.SCPPixel.Col) * ss_zd_s) +
               (np.arange(float(common_meta.ImageData.NumCols))*ss_zd_s))
        eta_c = -f_eta_c/k_a  # Beam center crossing time.  TimeCOA in SICD terminology
        eta_ref = eta_c - eta_c[0]
        eta_grid, eta_ref_grid = np.meshgrid(eta[cols], eta_ref[rows])
        eta_arg = eta_grid - eta_ref_grid
        deramp_phase = k_t[rows, np.newaxis] * np.power(eta_arg, 2) / 2
        demod_phase = eta_arg * f_eta_c[rows, np.newaxis]
        # Sampled phase correction for deramping and demodding
        total_phase = deramp_phase + demod_phase
        # Least squares fit for 2D polynomial
        # A*x = b
        [coords_az_m_2d, coords_rg_m_2d] = np.meshgrid(coords_az_m, coords_rg_m)
        a = np.zeros(((POLY_ORDER+1)**2, (POLY_ORDER+1)**2))
        for k in range(POLY_ORDER+1):
            for j in range(POLY_ORDER+1):
                a[:, k*(POLY_ORDER+1)+j] = np.multiply(
                    np.power(coords_az_m_2d.flatten(), j),
                    np.power(coords_rg_m_2d.flatten(), k))
        A = np.zeros(((POLY_ORDER+1)**2, (POLY_ORDER+1)**2))
        for k in range((POLY_ORDER+1)**2):
            for j in range((POLY_ORDER+1)**2):
                A[k, j] = np.multiply(a[:, k], a[:, j]).sum()
        b_phase = [np.multiply(total_phase.flatten(), a[:, k]).sum()
                   for k in range((POLY_ORDER+1)**2)]
        x_phase = np.linalg.solve(A, b_phase)
        phase = np.reshape(x_phase, (POLY_ORDER+1, POLY_ORDER+1))
        # DeltaKCOAPoly is derivative of phase in Col direction
        burst_meta.Grid.Col.DeltaKCOAPoly = poly.polyder(phase, axis=1)
        # DopCentroidPoly/TimeCOAPoly
        # Another way to derive the Doppler Centroid, which is back-calculated
        # from the ESA-documented azimuth deramp phase function.
        DopCentroidPoly = (burst_meta.Grid.Col.DeltaKCOAPoly *
                           (float(common_meta.Grid.Col.SS) / ss_zd_s))
        burst_meta.RMA.INCA.DopCentroidPoly = DopCentroidPoly
        dopcentroid2_sampled = poly.polyval2d(coords_rg_m_2d, coords_az_m_2d, DopCentroidPoly)
        timecoa_sampled = timeca_sampled + dopcentroid2_sampled/doprate_sampled[:, np.newaxis]
        # Convert sampled TimeCOA to polynomial
        b_coa = [np.multiply(timecoa_sampled.flatten(), a[:, k]).sum()
                 for k in range((POLY_ORDER+1)**2)]
        x_coa = np.linalg.solve(A, b_coa)
        burst_meta.Grid.TimeCOAPoly = np.reshape(x_coa, (POLY_ORDER+1, POLY_ORDER+1))
        # Timeline
        # We don't know the precise start and stop time of each burst (as in
        # the times of first and last pulses), so we use the min and max COA
        # time, which is a closer approximation than the min and max zero
        # Doppler times.  At least COA times will not overlap between bursts.
        if numbursts > 0:
            # STRIPMAP case uses another time origin different from first zero Doppler time
            time_offset = timecoa_sampled.min()
            burst_meta.Timeline.CollectStart = start + datetime.timedelta(seconds=time_offset)
            burst_meta.Timeline.CollectDuration = (timecoa_sampled.max() -
                                                   timecoa_sampled.min())
            # Adjust all SICD fields that were dependent on start time
            # Time is output of polynomial:
            burst_meta.Grid.TimeCOAPoly[0, 0] = \
                burst_meta.Grid.TimeCOAPoly[0, 0] - time_offset
            burst_meta.RMA.INCA.TimeCAPoly[0] = \
                burst_meta.RMA.INCA.TimeCAPoly[0] - time_offset
            # Time is input of polynomial:
            burst_meta.Position.ARPPoly.X = \
                _polyshift(burst_meta.Position.ARPPoly.X, time_offset)
            burst_meta.Position.ARPPoly.Y = \
                _polyshift(burst_meta.Position.ARPPoly.Y, time_offset)
            burst_meta.Position.ARPPoly.Z = \
                _polyshift(burst_meta.Position.ARPPoly.Z, time_offset)
        burst_meta.Timeline.IPP.Set.TStart = 0
        burst_meta.Timeline.IPP.Set.TEnd = burst_meta.Timeline.CollectDuration
        burst_meta.Timeline.IPP.Set.IPPStart = int(0)
        burst_meta.Timeline.IPP.Set.IPPEnd = \
            int(np.floor(burst_meta.Timeline.CollectDuration * prf))
        burst_meta.ImageFormation.TEndProc = burst_meta.Timeline.CollectDuration
        # GeoData
        # Rough estimate of SCP (interpolated from metadata geolocation grid)
        # to bootstrap (point_image_to_ground uses it only to find tangent to
        # ellipsoid.)  Then we will immediately replace it with a more precise
        # value from point_image_to_ground and the SICD sensor model.
        burst_meta.GeoData.SCP = MetaNode()
        burst_meta.GeoData.SCP.ECF = MetaNode()
        burst_meta.GeoData.SCP.ECF.X = scp_x[count]
        burst_meta.GeoData.SCP.ECF.Y = scp_y[count]
        burst_meta.GeoData.SCP.ECF.Z = scp_z[count]
        # Note that blindly using the heights in the geolocationGridPointList
        # can result in some confusing results.  Since the scenes can be
        # extremely large, you could easily be using a height in your
        # geolocationGridPointList that is very high, but still have ocean
        # shoreline in your scene. Blindly projecting the the plane tangent to
        # the inflated ellipsoid at SCP could result in some badly placed
        # geocoords in Google Earth.  Of course, one must always be careful
        # with ground projection and height variability, but probably even more
        # care is warranted in this data than even usual due to large scene
        # sizes and frequently steep graze angles.
        # Note also that some Sentinel-1 data we have see has different heights
        # in the geolocation grid for polarimetric channels from the same
        # swath/burst!?!
        llh = gc.ecf_to_geodetic((burst_meta.GeoData.SCP.ECF.X,
                                 burst_meta.GeoData.SCP.ECF.Y,
                                 burst_meta.GeoData.SCP.ECF.Z))
        burst_meta.GeoData.SCP.LLH = MetaNode()
        burst_meta.GeoData.SCP.LLH.Lat = llh[0, 0]
        burst_meta.GeoData.SCP.LLH.Lon = llh[0, 1]
        burst_meta.GeoData.SCP.LLH.HAE = llh[0, 2]
        # Now that SCP has been populated, populate GeoData.SCP more precisely.
        ecf = point.image_to_ground([burst_meta.ImageData.SCPPixel.Row,
                                     burst_meta.ImageData.SCPPixel.Col], burst_meta)[0]
        burst_meta.GeoData.SCP.ECF.X = ecf[0]
        burst_meta.GeoData.SCP.ECF.Y = ecf[1]
        burst_meta.GeoData.SCP.ECF.Z = ecf[2]
        llh = gc.ecf_to_geodetic(ecf)[0]
        burst_meta.GeoData.SCP.LLH.Lat = llh[0]
        burst_meta.GeoData.SCP.LLH.Lon = llh[1]
        burst_meta.GeoData.SCP.LLH.HAE = llh[2]

        sicd_meta.append(burst_meta)

    return sicd_meta


def meta2sicd_noise(filename, sicd_meta, manifest_meta):
    """This function parses the Sentinel Noise file and populates the NoisePoly field."""

    # Sentinel baseline processing calibration update on Nov 25 2015
    if manifest_meta.ImageCreation.DateTime < datetime.datetime(2015, 11, 25):
        return

    lines_per_burst = int(sicd_meta[0].ImageData.NumCols)
    range_size_pixels = int(sicd_meta[0].ImageData.NumRows)

    # Extract all relevant noise values from XML
    root_node = ET.parse(filename).getroot()

    def extract_noise(noise_string):
        noise_vector_list = root_node.findall('./' + noise_string + 'VectorList/' +
                                              noise_string + 'Vector')
        line = [None]*len(noise_vector_list)
        pixel = [None]*len(noise_vector_list)
        noise = [None]*len(noise_vector_list)
        nLUTind = 0
        for noise_vector in noise_vector_list:
            line[nLUTind] = np.fromstring(noise_vector.find('./line').text,
                                          dtype='int32', sep=' ')
            # Some datasets have noise vectors for negative lines.
            # Might mean that the data is included from a burst before the actual slice?
            # Ignore it for now, since line 0 always lines up with the first valid burst
            if np.all(line[nLUTind] < 0):
                continue
            pixel[nLUTind] = noise_vector.find('./pixel')
            if pixel[nLUTind] is not None:  # Doesn't exist for azimuth noise
                pixel[nLUTind] = np.fromstring(pixel[nLUTind].text, dtype='int32', sep=' ')
            noise[nLUTind] = np.fromstring(noise_vector.find('./'+noise_string+'Lut').text,
                                           dtype='float', sep=' ')
            # Some datasets do not have any noise data and are populated with 0s instead
            # In this case just don't populate the SICD noise metadata at all
            if not np.any(np.array(noise != 0.0) and np.array(noise) != np.NINF):
                return
            # Linear values given in XML. SICD uses dB.
            noise[nLUTind] = 10 * np.log10(noise[nLUTind])
            # Sanity checking
            if (sicd_meta[0].CollectionInfo.RadarMode.ModeID == 'IW' and  # SLC IW product
               np.any(line[nLUTind] % lines_per_burst != 0) and
               noise_vector != noise_vector_list[-1]):
                    # Last burst has different timing
                    raise(ValueError('Expect noise file to have one LUT per burst. ' +
                                     'More are present'))
            if (pixel[nLUTind] is not None and
               pixel[nLUTind][len(pixel[nLUTind])-1] > range_size_pixels):
                raise(ValueError('Noise file has more pixels in LUT than range size.'))
            nLUTind += 1
        # Remove empty list entries from negative lines
        line = [x for x in line if x is not None]
        pixel = [x for x in pixel if x is not None]
        noise = [x for x in noise if x is not None]
        return (line, pixel, noise)

    if root_node.find("./noiseVectorList") is not None:
        range_line, range_pixel, range_noise = extract_noise("noise")
        azi_noise = None
    else:  # noiseRange and noiseAzimuth fields began in March 2018
        range_line, range_pixel, range_noise = extract_noise("noiseRange")
        # This would pull azimuth noise, but it seems to only be about 1 dB, and is a
        # cycloid, which is hard to fit.
        azi_line, azi_pixel, azi_noise = extract_noise("noiseAzimuth")
    # Loop through each burst and fit a polynomial for SICD.
    # If data is stripmap, sicd_meta will be of length 1.
    for x in range(len(sicd_meta)):
        # Stripmaps have more than one noise LUT for the whole image
        if(sicd_meta[0].CollectionInfo.RadarMode.ModeID[0] == 'S'):
            coords_rg_m = (range_pixel[0] + sicd_meta[x].ImageData.FirstRow -
                           sicd_meta[x].ImageData.SCPPixel.Row) * sicd_meta[x].Grid.Row.SS
            coords_az_m = (np.concatenate(range_line) + sicd_meta[x].ImageData.FirstCol -
                           sicd_meta[x].ImageData.SCPPixel.Col) * sicd_meta[x].Grid.Col.SS

            # Fitting the two axis seperately then combining gives error than most 2d solvers
            # The noise is not monotonic across range
            rg_fit = np.polynomial.polynomial.polyfit(coords_rg_m, np.mean(range_noise, 0), 7)
            # Azimuth noise varies far less than range
            az_fit = np.polynomial.polynomial.polyfit(coords_az_m, np.mean(range_noise, 1), 7)
            noise_poly = np.outer(az_fit / np.max(az_fit), rg_fit)
        else:  # TOPSAR modes (IW/EW) have a single LUT per burst. Num of bursts varies.
            # Noise varies significantly in range, but also typically a little, ~1dB, in azimuth.
            coords_rg_m = (range_pixel[x]+sicd_meta[x].ImageData.FirstRow -
                           sicd_meta[x].ImageData.SCPPixel.Row) * sicd_meta[x].Grid.Row.SS
            rg_fit = np.polynomial.polynomial.polyfit(coords_rg_m, range_noise[x], 7)
            noise_poly = np.array(rg_fit).reshape(1, -1).T  # Make values along SICD range

            if azi_noise is not None:
                coords_az_m = ((azi_line[0] - (lines_per_burst*x) -
                                sicd_meta[x].ImageData.SCPPixel.Col) * sicd_meta[x].Grid.Col.SS)
                valid_lines = np.logical_and(np.array(azi_line[0]) >= lines_per_burst*x,
                                             np.array(azi_line[0]) < lines_per_burst*(x+1))
                az_fit = np.polynomial.polynomial.polyfit(coords_az_m[valid_lines],
                                                          azi_noise[0][valid_lines], 2)
                noise_poly = np.zeros((len(rg_fit), len(az_fit)))
                noise_poly[:, 0] = rg_fit
                noise_poly[0, :] = az_fit
                noise_poly[0, 0] = rg_fit[0]+az_fit[0]

        # should have Radiometric field already in metadata if cal file is present
        if not hasattr(sicd_meta[x], 'Radiometric'):
            sicd_meta[x].Radiometric = MetaNode()
        sicd_meta[x].Radiometric.NoiseLevel = MetaNode()
        sicd_meta[x].Radiometric.NoiseLevel.NoiseLevelType = 'ABSOLUTE'
        sicd_meta[x].Radiometric.NoiseLevel.NoisePoly = noise_poly


def meta2sicd_cal(filename, sicd_meta, manifest_meta):
    """This function parses the Sentinel calibration file and extends SICD metadata
    with the radiometric fields"""

    # Data before the Sentinel baseline processing calibration update on Nov 25 2015 is useless
    if manifest_meta.ImageCreation.DateTime < datetime.datetime(2015, 11, 25):
        return

    # Extract all calibration values form XML
    root_node = ET.parse(filename).getroot()
    calib_vec_list = root_node.findall('./calibrationVectorList/calibrationVector')

    line = np.empty((0,))
    pixel = [None]*len(calib_vec_list)
    sigma = [None]*len(calib_vec_list)
    beta = [None]*len(calib_vec_list)
    gamma = [None]*len(calib_vec_list)

    for i in range(0, len(calib_vec_list)):
        pixels_for_this_line = np.fromstring(calib_vec_list[i].find('./pixel').text, sep=' ')
        line = np.append(line, float(calib_vec_list[i].find('./line').text))
        pixel[i] = pixels_for_this_line
        sigma[i] = np.array(
                          np.fromstring(calib_vec_list[i].find('./sigmaNought').text,
                                        dtype='float', sep=' '))
        beta[i] = np.array(
                         np.fromstring(calib_vec_list[i].find('./betaNought').text,
                                       dtype='float', sep=' '))
        gamma[i] = np.array(
                          np.fromstring(calib_vec_list[i].find('./gamma').text,
                                        dtype='float', sep=' '))

    lines_per_burst = int(sicd_meta[0].ImageData.NumCols)

    pixel = np.array(pixel)
    beta = np.array(beta).flatten()
    gamma = np.array(gamma).flatten()
    sigma = np.array(sigma).flatten()

    # Sentinel values must be squared before SICD uses them as a scalar
    # Also Sentinel convention is to divide out scale factor. SICD convention is to multiply.
    beta = beta**-2
    gamma = gamma**-2
    sigma = sigma**-2

    # Compute noise polynomial for each burst (or one time for stripmap)
    for x in range(len(sicd_meta)):
        if all(beta[0] == beta):  # This should be most of the time
            if not hasattr(sicd_meta[x], 'Radiometric'):
                sicd_meta[x].Radiometric = MetaNode()
            sicd_meta[x].Radiometric.BetaZeroSFPoly = np.atleast_2d(beta[0])
            # sicd.derived_fields will populate other radiometric fields later
        else:  # In case we run into spatially variant radiometric data
            # Sentinel generates radiometric calibration info every 1 second.
            # For each burst's polynomials, use only the calibration vectors
            # that occurred during each burst so that the fitted polynomials have less error
            # We could also investigate results of possibly including
            # vectors on the outside of each burst edge
            valid_lines = ((line >= (x * lines_per_burst)) &
                           (line < ((x+1) * lines_per_burst)))

            valid_pixels = np.repeat(valid_lines, len(np.ones_like(pixel[x])))

            # If Burst has no calibration data
            if not np.any(valid_lines) or not np.any(valid_pixels):
                continue

            # Convert pixel coordinates from image indices to SICD image
            # coordinates (xrow and ycol)
            coords_rg_m = (pixel[valid_lines]+sicd_meta[x].ImageData.FirstRow -
                           sicd_meta[x].ImageData.SCPPixel.Row) * sicd_meta[x].Grid.Row.SS
            coords_az_m = np.repeat((line[valid_lines]+sicd_meta[x].ImageData.FirstCol -
                                    sicd_meta[x].ImageData.SCPPixel.Col) *
                                    sicd_meta[x].Grid.Col.SS,
                                    len(np.ones_like(pixel[x])))

            # Fitting the two axes seperately then combining gives lower error than most 2d solvers
            rg_fit = np.polynomial.polynomial.polyfit(coords_rg_m.flatten(),
                                                      sigma[valid_pixels], 2)
            az_fit = np.polynomial.polynomial.polyfit(coords_az_m.flatten(),
                                                      sigma[valid_pixels], 2)
            sigma_poly = np.outer(az_fit / np.max(az_fit), rg_fit)

            rg_fit = np.polynomial.polynomial.polyfit(coords_rg_m.flatten(),
                                                      beta[valid_pixels], 2)
            az_fit = np.polynomial.polynomial.polyfit(coords_az_m.flatten(),
                                                      beta[valid_pixels], 2)
            beta_poly = np.outer(az_fit / np.max(az_fit), rg_fit)

            rg_fit = np.polynomial.polynomial.polyfit(coords_rg_m.flatten(),
                                                      gamma[valid_pixels], 2)
            az_fit = np.polynomial.polynomial.polyfit(coords_az_m.flatten(),
                                                      gamma[valid_pixels], 2)
            gamma_poly = np.outer(az_fit / np.max(az_fit), rg_fit)

            if not hasattr(sicd_meta[x], 'Radiometric'):
                sicd_meta[x].Radiometric = MetaNode()
            sicd_meta[x].Radiometric.SigmaZeroSFPoly = sigma_poly
            sicd_meta[x].Radiometric.BetaZeroSFPoly = beta_poly
            sicd_meta[x].Radiometric.GammaZeroSFPoly = gamma_poly
