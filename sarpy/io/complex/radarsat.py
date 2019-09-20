"""Module for reading Radarsat (RS2 and RCM) data into a SICD model."""

# SarPy imports
from .sicd import MetaNode
from . import Reader as ReaderSuper  # Reader superclass
from . import sicd
from . import tiff
from ...geometry import geocoords as gc
from ...geometry import point_projection as point
# Python standard library imports
import copy
import datetime
import os
import re
import xml.etree.ElementTree as ET
# External dependencies
import numpy as np
# We prefer numpy.polynomial.polynomial over numpy.polyval/polyfit since its coefficient
# ordering is consistent with SICD, and because it supports 2D polynomials.
from numpy.polynomial import polynomial as poly
from scipy.constants import speed_of_light
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
__author__ = ["Khanh Ho", "Wade Schwartzkopf"]
__email__ = "Wade.C.Schwartzkopf.ctr@nga.mil"

DATE_FMT = '%Y-%m-%dT%H:%M:%S.%fZ'  # The datetime format Sentinel1 always uses


def isa(filename):
    """Test to see if file is a product.xml file."""
    try:
        ns = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
        ns['default'] = ns.get('')
        root_node = ET.parse(filename).getroot()
        satellite = root_node.find('./default:sourceAttributes/default:satellite', ns).text.upper()
        product_type = root_node.find('./default:imageGenerationParameters' +
                                      '/default:generalProcessingInformation' +
                                      '/default:productType', ns).text.upper()
        if ((satellite.upper() == 'RADARSAT-2' or satellite[:3].upper() == 'RCM') and
           product_type == 'SLC'):
            return Reader
    except Exception:
        pass


class Reader(ReaderSuper):
    def __init__(self, product_filename):
        basepathname = os.path.dirname(product_filename)
        ns = dict([node for _, node in ET.iterparse(product_filename,
                                                    events=['start-ns'])])
        ns['default'] = ns.get('')
        root_node = ET.parse(product_filename).getroot()
        # Read metadata and convert to SICD structure
        collector_name = root_node.find(
            './default:sourceAttributes/default:satellite', ns).text
        if collector_name.upper() == 'RADARSAT-2':
            ia_str = 'imageAttributes'
            ipdf_str = ('./default:imageAttributes' +
                        '/default:fullResolutionImageData')
            beta_lut_str = ('./default:imageAttributes' +
                            '/default:lookupTable' +
                            '[@incidenceAngleCorrection="Beta Nought"]')
            beta_lut_str = os.path.join(os.path.dirname(product_filename),
                                        root_node.find(beta_lut_str, ns).text)
            noise_str = None  # No separate noise file for RS2
        elif collector_name[:3].upper() == 'RCM':
            ia_str = 'imageReferenceAttributes'
            ipdf_str = ('./default:sceneAttributes' +
                        '/default:imageAttributes' +
                        '/default:ipdf')
            beta_lut_str = ('./default:imageReferenceAttributes' +
                            '/default:lookupTableFileName' +
                            '[@sarCalibrationType="Beta Nought"]')
            beta_lut_str = os.path.join(os.path.dirname(product_filename),
                                        'calibration',
                                        root_node.find(beta_lut_str, ns).text)
            noise_str = ('./default:imageReferenceAttributes' +
                         '/default:noiseLevelFileName')
            noise_str = os.path.join(os.path.dirname(product_filename),
                                     'calibration',
                                     root_node.find(noise_str, ns).text)
        self.sicdmeta = meta2sicd(product_filename, beta_lut_str, noise_str)
        # Setup pixel readers
        line_order = root_node.find('./default:' + ia_str +
                                    '/default:rasterAttributes' +
                                    '/default:lineTimeOrdering', ns).text
        lookdir = root_node.find('./default:sourceAttributes' +
                                 '/default:radarParameters' +
                                 '/default:antennaPointing', ns).text.upper()
        sample_order = root_node.find('./default:' + ia_str +
                                      '/default:rasterAttributes' +
                                      '/default:pixelTimeOrdering', ns).text
        symmetry = [(line_order.upper() == 'DECREASING') != (lookdir[0] == 'L'),
                    sample_order.upper() == 'DECREASING',
                    True]
        self.read_chip = []
        datafilenames = root_node.findall(ipdf_str, ns)
        for i in range(len(datafilenames)):
            tiff_filename = os.path.join(basepathname, datafilenames[i].text)
            meta_tiff = tiff.read_meta(tiff_filename)
            self.read_chip.append(tiff.chipper(tiff_filename, symmetry, meta_tiff))
            self.sicdmeta[i].native = MetaNode()
            self.sicdmeta[i].native.tiff = meta_tiff


def meta2sicd(filename, betafile, noisefile):
    """Converts Radarsat product.xml description into a SICD-style metadata structure.

    There is an outstanding question with regard to the meaning of the pulseRepetitionFrequency
    as provided.  See comments in Timeline section below."""

    def _polyshift(a, shift):
        b = np.zeros(a.size)
        for j in range(1, len(a)+1):
            for k in range(j, len(a)+1):
                b[j-1] = b[j-1] + (a[k-1]*comb(k-1, j-1)*np.power(shift, (k-j)))
        return b

    root = _xml_parse_wo_default_ns(filename)

    meta = MetaNode()

    # CollectionInfo
    meta.CollectionInfo = MetaNode()
    meta.CollectionInfo.CollectorName = root.find('./sourceAttributes/satellite').text
    if meta.CollectionInfo.CollectorName == 'RADARSAT-2':
        gen = 'RS2'
    else:
        gen = 'RCM'
    raw_start_time = datetime.datetime.strptime(root.find(
        './sourceAttributes/rawDataStartTime').text, DATE_FMT)
    date_str = raw_start_time.strftime('%d%B%y').upper()
    time_str = raw_start_time.strftime('%H%M%S')
    if gen == 'RS2':
        meta.CollectionInfo.CoreName = (date_str + 'RS2' +
                                        root.find('./sourceAttributes/imageId').text)
    elif gen == 'RCM':
        meta.CollectionInfo.CoreName = (date_str +
                                        meta.CollectionInfo.CollectorName.replace('-', '') +
                                        time_str)
    meta.CollectionInfo.CollectType = 'MONOSTATIC'
    meta.CollectionInfo.RadarMode = MetaNode()
    meta.CollectionInfo.RadarMode.ModeID = (
            root.find('./sourceAttributes/beamModeMnemonic').text)

    # First use beammode if it exists
    beamMode = root.find('./sourceAttributes/beamMode')
    acqType = root.find('./sourceAttributes/radarParameters/acquisitionType')
    if ((beamMode is not None and beamMode.text.upper().startswith("SPOTLIGHT")) or
       (acqType is not None and acqType.text.upper().startswith("SPOTLIGHT")) or
       "SL" in meta.CollectionInfo.RadarMode.ModeID):
        meta.CollectionInfo.RadarMode.ModeType = "SPOTLIGHT"
    elif (meta.CollectionInfo.RadarMode.ModeID[:2] == 'SC'):
        raise(ValueError('ScanSAR mode data is not currently handled.'))
    else:
        # Finally assume it's stripmap
        meta.CollectionInfo.RadarMode.ModeType = 'STRIPMAP'
    if gen == 'RS2':
        meta.CollectionInfo.Classification = 'UNCLASSIFIED'
    elif gen == 'RCM':
        classification_str = root.find('./securityAttributes/securityClassification').text
        if "UNCLASS" in classification_str.upper():
            meta.CollectionInfo.Classification = "UNCLASSIFIED"
        else:
            meta.CollectionInfo.Classification = classification_str
    # ImageCreation
    meta.ImageCreation = MetaNode()
    procinfo = root.find('./imageGenerationParameters/generalProcessingInformation')
    meta.ImageCreation.Application = procinfo.find('softwareVersion').text
    meta.ImageCreation.DateTime = datetime.datetime.strptime(
        procinfo.find('processingTime').text, DATE_FMT)
    meta.ImageCreation.Site = procinfo.find('processingFacility').text
    meta.ImageCreation.Profile = 'Prototype'

    # ImageData
    meta.ImageData = MetaNode()
    if gen == 'RS2':
        meta.ImageData.NumCols = int(root.find(
                './imageAttributes/rasterAttributes/numberOfLines').text)
        meta.ImageData.NumRows = int(root.find(
                './imageAttributes/rasterAttributes/numberOfSamplesPerLine').text)
    elif gen == 'RCM':
        meta.ImageData.NumCols = int(root.find(
                './sceneAttributes/imageAttributes/numLines').text)
        meta.ImageData.NumRows = int(root.find(
                './sceneAttributes/imageAttributes/samplesPerLine').text)

    meta.ImageData.FullImage = copy.deepcopy(meta.ImageData)
    meta.ImageData.FirstRow = int(0)
    meta.ImageData.FirstCol = int(0)
    meta.ImageData.PixelType = 'RE16I_IM16I'
    if (gen == 'RCM'):
        bpp = int(root.find(
                './imageReferenceAttributes/rasterAttributes/bitsPerSample').text)
        if (bpp == 32):
            meta.ImageData.PixelType = 'RE32F_IM32F'
    # Seems that all pixels are always valid
    meta.ImageData.ValidData = MetaNode()
    meta.ImageData.ValidData.Vertex = [MetaNode(), MetaNode(), MetaNode(), MetaNode()]
    meta.ImageData.ValidData.Vertex[0].Row = 0
    meta.ImageData.ValidData.Vertex[0].Col = 0
    meta.ImageData.ValidData.Vertex[1].Row = 0
    meta.ImageData.ValidData.Vertex[1].Col = meta.ImageData.NumCols-1
    meta.ImageData.ValidData.Vertex[2].Row = meta.ImageData.NumRows-1
    meta.ImageData.ValidData.Vertex[2].Col = meta.ImageData.NumCols-1
    meta.ImageData.ValidData.Vertex[3].Row = meta.ImageData.NumRows-1
    meta.ImageData.ValidData.Vertex[3].Col = 0
    # SCP
    if gen == 'RS2':
        im_at_str = './imageAttributes/'
    elif gen == 'RCM':
        im_at_str = './imageReferenceAttributes/'
    # There are many different equally valid options for picking the SCP point.
    # One way is to chose the tie point that is closest to the image center.
    tiepoints = root.findall(im_at_str + 'geographicInformation/geolocationGrid/imageTiePoint')
    pixels = [float(tp.find('imageCoordinate/pixel').text) for tp in tiepoints]
    lines = [float(tp.find('imageCoordinate/line').text) for tp in tiepoints]
    lats = [float(tp.find('geodeticCoordinate/latitude').text) for tp in tiepoints]
    longs = [float(tp.find('geodeticCoordinate/longitude').text) for tp in tiepoints]
    hgts = [float(tp.find('geodeticCoordinate/height').text) for tp in tiepoints]
    # Pick tie point closest to center for SCP
    center_point = [(meta.ImageData.NumRows-1)/2, (meta.ImageData.NumCols-1)/2]
    D = (np.vstack((pixels, lines)).transpose() - center_point)
    scp_index = np.argmin(np.sqrt(np.sum(np.power(D, 2), axis=1)))
    meta.ImageData.SCPPixel = MetaNode()
    meta.ImageData.SCPPixel.Row = int(round(pixels[scp_index]))
    meta.ImageData.SCPPixel.Col = int(round(lines[scp_index]))

    # GeoData
    meta.GeoData = MetaNode()
    # All RS2 and RCM data we know use the WGS84 model, although it is stated
    # in slightly different XML fields in the RS2 and RCM product.xml
    meta.GeoData.EarthModel = 'WGS_84'
    meta.GeoData.SCP = MetaNode()
    meta.GeoData.SCP.LLH = MetaNode()
    # Initially, we just seed this with a rough value.  Later we will put in
    # something more precise.
    meta.GeoData.SCP.LLH.Lat = lats[scp_index]
    meta.GeoData.SCP.LLH.Lon = longs[scp_index]
    meta.GeoData.SCP.LLH.HAE = hgts[scp_index]
    pos_ecf = gc.geodetic_to_ecf((meta.GeoData.SCP.LLH.Lat,
                                  meta.GeoData.SCP.LLH.Lon,
                                  meta.GeoData.SCP.LLH.HAE))
    meta.GeoData.SCP.ECF = MetaNode()
    meta.GeoData.SCP.ECF.X = pos_ecf[0, 0]
    meta.GeoData.SCP.ECF.Y = pos_ecf[0, 1]
    meta.GeoData.SCP.ECF.Z = pos_ecf[0, 2]
    # We could also use tie points for corner coordinates, but instead we will be compute them
    # later more precisely with sicd.derived_fields.

    # Position
    meta.Position = MetaNode()
    meta.Position.ARPPoly = MetaNode()
    state_vec_list = root.findall('./sourceAttributes/orbitAndAttitude/' +
                                  'orbitInformation/stateVector')
    state_vector_T, state_vector_X, state_vector_Y, state_vector_Z = [], [], [], []
    vel_X, vel_Y, vel_Z = [], [], []
    for state_vec in state_vec_list:
        state_vector_T.append(datetime.datetime.strptime(state_vec.find('timeStamp').text,
                                                         DATE_FMT))
        state_vector_X.append(float(state_vec.find('xPosition').text))
        state_vector_Y.append(float(state_vec.find('yPosition').text))
        state_vector_Z.append(float(state_vec.find('zPosition').text))
        vel_X.append(float(state_vec.find('xVelocity').text))
        vel_Y.append(float(state_vec.find('yVelocity').text))
        vel_Z.append(float(state_vec.find('zVelocity').text))
    state_vector_T = np.array([(t-raw_start_time).total_seconds() for t in state_vector_T])
    # Here we find the order of polynomial that most accurately describes
    # this position, but use velocity as cross-validation so that the data
    # is not being overfit.
    polyorder = 2
    current_vel_error = float('inf')
    last_vel_error = float('inf')
    P_x, P_y, P_z = [], [], []
    while ((current_vel_error <= last_vel_error) and (polyorder < len(state_vector_T))):
        last_vel_error = current_vel_error
        P_x = poly.polyfit(state_vector_T, state_vector_X, polyorder)
        P_y = poly.polyfit(state_vector_T, state_vector_Y, polyorder)
        P_z = poly.polyfit(state_vector_T, state_vector_Z, polyorder)
        V_x = poly.polyval(state_vector_T, poly.polyder(P_x))
        V_y = poly.polyval(state_vector_T, poly.polyder(P_y))
        V_z = poly.polyval(state_vector_T, poly.polyder(P_z))
        current_vel_error = np.sum(np.power(np.subtract([V_x, V_y, V_z],
                                                        [vel_X, vel_Y, vel_Z]), 2))
        polyorder = polyorder + 1
        if current_vel_error < last_vel_error:
            meta.Position.ARPPoly.X = P_x
            meta.Position.ARPPoly.Y = P_y
            meta.Position.ARPPoly.Z = P_z

    # Another option is just to fix the polynomial order to something reasonable
    polyorder = min((4, len(state_vector_T) - 1))
    meta.Position.ARPPoly.X = poly.polyfit(state_vector_T,
                                           state_vector_X, polyorder)
    meta.Position.ARPPoly.Y = poly.polyfit(state_vector_T,
                                           state_vector_Y, polyorder)
    meta.Position.ARPPoly.Z = poly.polyfit(state_vector_T,
                                           state_vector_Z, polyorder)

    # Grid
    meta.Grid = MetaNode()
    meta.Grid.ImagePlane = 'SLANT'
    meta.Grid.Type = 'RGZERO'
    meta.Grid.Row = MetaNode()
    meta.Grid.Col = MetaNode()
    meta.Grid.Row.SS = float(root.find(im_at_str + 'rasterAttributes/sampledPixelSpacing').text)
    # Col.SS is derived after DRateSFPoly below, rather than used from this
    # given field, so that SICD metadata can be internally consistent.
    # meta.Grid.Col.SS = float(root.find(im_at_str +
    #                                    'rasterAttributes/sampledLineSpacing').text)
    meta.Grid.Row.Sgn = -1  # Always true for RS2
    meta.Grid.Col.Sgn = -1  # Always true for RS2
    rp = root.find('./sourceAttributes/radarParameters')
    fc = float(rp.find('radarCenterFrequency').text)  # Center frequency
    meta.Grid.Row.ImpRespBW = (2 / speed_of_light) * float(root.find(
        './imageGenerationParameters/sarProcessingInformation/totalProcessedRangeBandwidth').text)
    dop_bw = float(root.find('./imageGenerationParameters/sarProcessingInformation/' +
                             'totalProcessedAzimuthBandwidth').text)  # Doppler bandwidth
    zd_last = datetime.datetime.strptime(root.find(
        './imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeLastLine').text,
        DATE_FMT)
    zd_first = datetime.datetime.strptime(root.find(
        './imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeFirstLine').text,
        DATE_FMT)
    # Image column spacing in zero doppler time (seconds)
    ss_zd_s = abs((zd_last - zd_first).total_seconds()) / (meta.ImageData.NumCols - 1)
    meta.Grid.Row.KCtr = 2*fc/speed_of_light
    meta.Grid.Col.KCtr = 0
    meta.Grid.Row.DeltaKCOAPoly = np.atleast_2d(0)
    # Constants used to compute weighting parameters
    meta.Grid.Row.WgtType = MetaNode()
    meta.Grid.Row.WgtType.WindowName = root.find('./imageGenerationParameters/' +
                                                 'sarProcessingInformation/' +
                                                 'rangeWindow/windowName').text.upper()
    if meta.Grid.Row.WgtType.WindowName == 'KAISER':  # The usual RS2 weigting
        meta.Grid.Row.WgtType.Parameter = MetaNode()
        meta.Grid.Row.WgtType.Parameter.name = 'BETA'
        meta.Grid.Row.WgtType.Parameter.value = root.find('./imageGenerationParameters/' +
                                                          'sarProcessingInformation/' +
                                                          'rangeWindow/windowCoefficient').text
    meta.Grid.Col.WgtType = MetaNode()
    meta.Grid.Col.WgtType.WindowName = root.find('./imageGenerationParameters/' +
                                                 'sarProcessingInformation/' +
                                                 'azimuthWindow/windowName').text.upper()
    if meta.Grid.Col.WgtType.WindowName == 'KAISER':  # The usual RS2 weigting
        meta.Grid.Col.WgtType.Parameter = MetaNode()
        meta.Grid.Col.WgtType.Parameter.name = 'BETA'
        meta.Grid.Col.WgtType.Parameter.value = root.find('./imageGenerationParameters/' +
                                                          'sarProcessingInformation/' +
                                                          'azimuthWindow/windowCoefficient').text
    # WgtFunct and ImpRespWid will be computed later in sicd.derived_fields

    # Radar Collection
    # Ultrafine and spotlight modes have "lower" and "upper" parts to the pulse.
    meta.RadarCollection = MetaNode()
    meta.RadarCollection.Waveform = MetaNode()
    meta.RadarCollection.Waveform.WFParameters = []
    bw_elements = rp.findall('pulseBandwidth')
    pulse_parts = sorted([node.get('pulse') for node in bw_elements])  # Sort so 'lower' is first
    bw = np.zeros(len(pulse_parts))
    for i in range(len(bw_elements)):
        wfpar = MetaNode()
        if gen == 'RS2':
            pulse_part_str = '[@pulse="' + pulse_parts[i] + '"]'
        elif gen == 'RCM':
            pulse_part_str = ''
        bw[i] = float(rp.find('pulseBandwidth' + pulse_part_str).text)
        wfpar.TxRFBandwidth = bw[i]
        wfpar.TxPulseLength = float(rp.find('pulseLength' + pulse_part_str).text)
        wfpar.RcvDemodType = 'CHIRP'
        sample_rate = float(rp.find('adcSamplingRate' + pulse_part_str).text)
        wfpar.RcvWindowLength = float(rp.find('samplesPerEchoLine').text)/sample_rate
        wfpar.ADCSampleRate = sample_rate
        wfpar.RcvFMRate = 0  # True for RcvDemodType='CHIRP'
        meta.RadarCollection.Waveform.WFParameters.append(wfpar)
    bw = np.sum(bw)
    meta.RadarCollection.TxFrequency = MetaNode()
    meta.RadarCollection.TxFrequency.Min = fc - (bw / 2)  # fc calculated in Grid section
    meta.RadarCollection.TxFrequency.Max = fc + (bw / 2)
    # Assumes pulse parts are exactly adjacent in bandwidth
    meta.RadarCollection.Waveform.WFParameters[0].TxFreqStart = \
        meta.RadarCollection.TxFrequency.Min
    for i in range(1, len(pulse_parts)):
        meta.RadarCollection.Waveform.WFParameters[i].TxFreqStart = \
            meta.RadarCollection.Waveform.WFParameters[i-1].TxFreqStart + \
            meta.RadarCollection.Waveform.WFParameters[i-1].TxRFBandwidth
    # Polarization
    pols = rp.find('polarizations').text.split()

    def convert_c_to_rhc(s):
        if s == "C":
            return "RHC"
        else:
            return s
    tx_pols = list({convert_c_to_rhc(p[0]) for p in pols})
    meta.RadarCollection.RcvChannels = MetaNode()
    meta.RadarCollection.RcvChannels.ChanParameters = [None] * len(pols)
    for i in range(len(pols)):
        meta.RadarCollection.RcvChannels.ChanParameters[i] = MetaNode()
        meta.RadarCollection.RcvChannels.ChanParameters[i].TxRcvPolarization = \
            convert_c_to_rhc(pols[i][0]) + ':' + convert_c_to_rhc(pols[i][1])
    if len(tx_pols) == 1:  # Only one transmit polarization
        meta.RadarCollection.TxPolarization = tx_pols[0]
    else:  # Multiple transmit polarizations
        meta.RadarCollection.TxPolarization = 'SEQUENCE'
        meta.RadarCollection.TxSequence = MetaNode()
        meta.RadarCollection.TxSequence.TxStep = [None] * len(tx_pols)
        for i in range(len(tx_pols)):
            meta.RadarCollection.TxSequence.TxStep[i] = MetaNode()
            meta.RadarCollection.TxSequence.TxStep[i].TxPolarization = tx_pols[i]

    # Timeline
    meta.Timeline = MetaNode()
    meta.Timeline.CollectStart = raw_start_time
    if gen == 'RS2':
        prf_xp_str = 'pulseRepetitionFrequency'
    elif gen == 'RCM':
        prf_xp_str = 'prfInformation/pulseRepetitionFrequency'
    prf = float(rp.find(prf_xp_str).text)
    num_lines_processed = [float(element.text) for element in
                           root.findall('imageGenerationParameters/sarProcessingInformation/' +
                                        'numberOfLinesProcessed')]
    if (len(num_lines_processed) == len(pols) and
       all(x == num_lines_processed[0] for x in num_lines_processed)):
        # If the above cases don't hold, we don't know what to do
        num_lines_processed = num_lines_processed[0] * len(tx_pols)
        prf = prf * len(pulse_parts)
        if len(pulse_parts) == 2 and meta.CollectionInfo.RadarMode.ModeType == 'STRIPMAP':
            # Why????
            # This seems to be necessary to make CollectDuration match CA ranges
            # and to make 1/prf roughly equal to ss_zd_s (which is generally true
            # for STRIPMAP).  But we already doubled the prf above to account for
            # the pulse parts (to make real vs effective prf), so why do we have to
            # do it again? And why don't we have to do it for SPOTLIGHT?
            prf = 2*prf
        meta.Timeline.CollectDuration = num_lines_processed/prf
        meta.Timeline.IPP = MetaNode()
        meta.Timeline.IPP.Set = MetaNode()
        meta.Timeline.IPP.Set.TStart = 0
        meta.Timeline.IPP.Set.TEnd = num_lines_processed/prf
        meta.Timeline.IPP.Set.IPPStart = 0
        meta.Timeline.IPP.Set.IPPEnd = int(num_lines_processed)
        meta.Timeline.IPP.Set.IPPPoly = np.array([0, prf])

    # Image Formation
    meta.ImageFormation = MetaNode()
    meta.ImageFormation.RcvChanProc = MetaNode()
    meta.ImageFormation.RcvChanProc.NumChanProc = 1
    # PRFScaleFactor for either polarimetric or multi-step, but not both.
    meta.ImageFormation.RcvChanProc.PRFScaleFactor = 1/max((len(pulse_parts), len(tx_pols)))
    meta.ImageFormation.ImageFormAlgo = 'RMA'
    meta.ImageFormation.TStartProc = 0
    meta.ImageFormation.TEndProc = meta.Timeline.CollectDuration
    meta.ImageFormation.TxFrequencyProc = MetaNode()
    meta.ImageFormation.TxFrequencyProc.MinProc = meta.RadarCollection.TxFrequency.Min
    meta.ImageFormation.TxFrequencyProc.MaxProc = meta.RadarCollection.TxFrequency.Max
    meta.ImageFormation.STBeamComp = 'NO'
    meta.ImageFormation.ImageBeamComp = 'NO'
    meta.ImageFormation.AzAutofocus = 'NO'
    meta.ImageFormation.RgAutofocus = 'NO'

    # RMA.INCA
    meta.RMA = MetaNode()
    meta.RMA.RMAlgoType = 'OMEGA_K'
    meta.RMA.ImageType = 'INCA'
    meta.SCPCOA = MetaNode()
    # We could derive SideOfTrack, but its easier to assume metadata is correct and just use it.
    meta.SCPCOA.SideOfTrack = root.find('./sourceAttributes/radarParameters/antennaPointing').text
    meta.SCPCOA.SideOfTrack = meta.SCPCOA.SideOfTrack[0].upper()
    if meta.SCPCOA.SideOfTrack == 'L':
        ss_zd_s = -ss_zd_s
        # In addition to left/right, RS2 data can independently be in
        # increasing/decreasing line order.
        if (zd_first - zd_last).total_seconds() < 0:  # zd_last occurred after zd_first
            zd_first = zd_last
        look = 1
    else:
        if (zd_first - zd_last).total_seconds() > 0:  # zd_last occurred before zd_first
            zd_first = zd_last
        look = -1
    # Zero doppler time of SCP relative to collect start
    zd_t_scp = (zd_first-raw_start_time).total_seconds() + (meta.ImageData.SCPPixel.Col * ss_zd_s)
    if gen == 'RS2':
        near_range = float(root.find('./imageGenerationParameters/sarProcessingInformation/' +
                                     'slantRangeNearEdge').text)  # in meters
    elif gen == 'RCM':
        near_range = float(root.find('./sceneAttributes/imageAttributes/' +
                                     'slantRangeNearEdge').text)  # in meters
    meta.RMA.INCA = MetaNode()
    meta.RMA.INCA.R_CA_SCP = near_range + (meta.ImageData.SCPPixel.Row * meta.Grid.Row.SS)
    meta.RMA.INCA.FreqZero = fc
    # Doppler Rate
    # We do this first since some other things are dependent on it.
    # For the purposes of the DRateSFPoly computation, we ignore any
    # changes in velocity over the azimuth dimension.
    vel = [poly.polyval(zd_t_scp, poly.polyder(meta.Position.ARPPoly.X)),
           poly.polyval(zd_t_scp, poly.polyder(meta.Position.ARPPoly.Y)),
           poly.polyval(zd_t_scp, poly.polyder(meta.Position.ARPPoly.Z))]
    vm_ca_sq = np.sum(np.power(vel, 2))  # Magnitude of the velocity squared
    # Polynomial representing range as a function of range distance from SCP
    r_ca = [meta.RMA.INCA.R_CA_SCP, 1]
    if gen == 'RS2':
        drc_xp_str = './imageGenerationParameters/dopplerRateValues/dopplerRateValuesCoefficients'
    elif gen == 'RCM':
        drc_xp_str = './dopplerRate/dopplerRateEstimate/dopplerRateCoefficients'
    # Shifted (origin at dop_rate_ref_t, not SCP) and scaled (sec, not m) version of
    # SICD DopCentroidPoly
    dop_rate_coefs = [float(x) for x in root.find(drc_xp_str).text.split()]
    # Reference time of Doppler rate polynomial
    dop_rate_ref_t = float(root.find(drc_xp_str + '/../dopplerRateReferenceTime').text)
    dop_rate_coefs_shifted = _polyshift(  # Shift so SCP is reference
        np.array(dop_rate_coefs),
        (meta.RMA.INCA.R_CA_SCP*2/speed_of_light) -  # SICD reference time (SCP)
        dop_rate_ref_t)  # Reference time of native Doppler Centroid polynomial
    dop_rate_coefs_scaled = (dop_rate_coefs_shifted *   # Scale from seconds to meters
                             np.power((2/speed_of_light), np.arange(len(dop_rate_coefs))))
    # Multiplication of two polynomials is just a convolution of their coefficients
    # # Assumes a SGN of -1
    meta.RMA.INCA.DRateSFPoly = (- np.convolve(dop_rate_coefs_scaled, r_ca) *
                                 speed_of_light / (2 * fc * vm_ca_sq))[:, np.newaxis]

    # Fields dependent on Doppler rate
    # This computation of SS is actually better than the claimed SS
    # (sampledLineSpacing) in many ways, because this makes all of the metadata
    # internally consistent.  This must be the sample spacing exactly at SCP
    # (which is the definition for SS in SICD), if the other metadata from
    # which is it computed is correct and consistent. Since column SS can vary
    # slightly over a RGZERO image, we don't know if the claimed sample spacing
    # in the native metadata is at our chosen SCP, or another point, or an
    # average across image or something else.
    meta.Grid.Col.SS = np.sqrt(vm_ca_sq) * ss_zd_s * meta.RMA.INCA.DRateSFPoly[0, 0]
    # Convert to azimuth spatial bandwidth (cycles per meter)
    meta.Grid.Col.ImpRespBW = dop_bw * abs(ss_zd_s) / meta.Grid.Col.SS
    meta.RMA.INCA.TimeCAPoly = np.array([zd_t_scp, ss_zd_s / meta.Grid.Col.SS])

    # Doppler Centroid
    if gen == 'RS2':
        dc_xp_str = './imageGenerationParameters/dopplerCentroid/'
    elif gen == 'RCM':
        dc_xp_str = './dopplerCentroid/dopplerCentroidEstimate/'
    # Shifted (origin at dop_cent_ref_t, not SCP) and scaled (sec, not m)
    # version of SICD DopCentroidPoly
    dop_cent_coefs = np.array([float(x) for x in
                               root.find(dc_xp_str + 'dopplerCentroidCoefficients').text.split()])
    # Reference time of Doppler Centroid polynomial
    dop_cent_ref_t = float(root.find(dc_xp_str + 'dopplerCentroidReferenceTime').text)
    dop_cent_coefs_shifted = _polyshift(
        dop_cent_coefs,  # Shift so SCP is reference
        (meta.RMA.INCA.R_CA_SCP*2/speed_of_light) -  # SICD reference time (SCP)
        dop_cent_ref_t)  # Reference time of native Doppler Centroid polynomial
    dop_cent_coefs_scaled = (dop_cent_coefs_shifted *  # Scale from seconds to meters
                             np.power((2/speed_of_light), np.arange(len(dop_cent_coefs))))
    meta.RMA.INCA.DopCentroidPoly = dop_cent_coefs_scaled[:, np.newaxis]
    # Adjust Doppler Centroid for spotlight
    if meta.CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT':
        # Doppler estimate time
        dop_est = datetime.datetime.strptime(root.find(
            dc_xp_str + 'timeOfDopplerCentroidEstimate').text, DATE_FMT)
        dop_est_t = (dop_est-raw_start_time).total_seconds()
        # This is the column (offset from the SCP) where the doppler centroid was computed.
        dop_est_col = (dop_est_t - zd_t_scp)/ss_zd_s
        # Column-dependent variation in DopCentroidPoly due to spotlight
        meta.RMA.INCA.DopCentroidPoly = np.hstack((meta.RMA.INCA.DopCentroidPoly,
                                                   np.zeros((dop_cent_coefs_scaled.size, 1))))
        meta.RMA.INCA.DopCentroidPoly[0, 1] = (
            -look * fc * (2 / speed_of_light) * np.sqrt(vm_ca_sq) / meta.RMA.INCA.R_CA_SCP)
        # dopplerCentroid in native metadata was defined at specific column,
        # which might not be our SCP column.  Adjust so that SCP column is
        # correct.
        meta.RMA.INCA.DopCentroidPoly[0, 0] = (meta.RMA.INCA.DopCentroidPoly[0, 0] -
                                               (meta.RMA.INCA.DopCentroidPoly[0, 1] *
                                                dop_est_col * meta.Grid.Col.SS))
    meta.Grid.Col.DeltaKCOAPoly = meta.RMA.INCA.DopCentroidPoly * ss_zd_s / meta.Grid.Col.SS
    # Compute Col.DeltaK1/K2 from DeltaKCOAPoly
    # This is not always straightforward to do generically for any possible
    # DeltaKCOAPoly 2D polynomial, since you would have to compute all 2D roots
    # and edge cases.  However, for the RS case, this can be solved exactly,
    # since its usually a 1D polynomial.  Even the spotlight case only brings
    # in a linear variation in the second dimensions, so its still easily
    # solved.
    # Min/max in row/range must exist at edges or internal local min/max
    minmax = poly.polyroots(poly.polyder(meta.Grid.Col.DeltaKCOAPoly[:, 0]))
    rg_bounds_m = (np.array([0, (meta.ImageData.NumRows-1)]) -
                   meta.ImageData.SCPPixel.Row) * meta.Grid.Row.SS
    possible_bounds_rg = np.concatenate((rg_bounds_m, minmax[np.logical_and(
        minmax > np.min(rg_bounds_m), minmax < np.max(rg_bounds_m))]))
    # Constant or (linearly increasing\decreasing for spotlight) in column, so
    # edges must contain max/min.
    possible_bounds_az = (np.array([0, (meta.ImageData.NumCols-1)]) -
                          meta.ImageData.SCPPixel.Col) * meta.Grid.Col.SS
    [coords_az_m_2d, coords_rg_m_2d] = np.meshgrid(possible_bounds_az, possible_bounds_rg)
    possible_bounds_deltak = poly.polyval2d(
        coords_rg_m_2d, coords_az_m_2d, meta.Grid.Col.DeltaKCOAPoly)
    meta.Grid.Col.DeltaK1 = np.min(possible_bounds_deltak) - (meta.Grid.Col.ImpRespBW/2)
    meta.Grid.Col.DeltaK2 = np.max(possible_bounds_deltak) + (meta.Grid.Col.ImpRespBW/2)
    # Wrapped spectrum
    if ((meta.Grid.Col.DeltaK1 < -(1/meta.Grid.Col.SS)/2) or
       (meta.Grid.Col.DeltaK2 > (1/meta.Grid.Col.SS)/2)):
        meta.Grid.Col.DeltaK1 = -(1/meta.Grid.Col.SS)/2
        meta.Grid.Col.DeltaK2 = -meta.Grid.Col.DeltaK1
    # TimeCOAPoly
    # TimeCOAPoly = TimeCA + (DopCentroid / dop_rate)
    # Since we can't evaluate this equation analytically, we will evaluate
    # samples of it across our image and fit a 2D polynomial to it.
    POLY_ORDER = 2  # Order of polynomial which we want to compute in each dimension
    grid_samples = POLY_ORDER + 1
    coords_az_m = np.linspace(-meta.ImageData.SCPPixel.Col * meta.Grid.Col.SS,
                              (meta.ImageData.NumCols - meta.ImageData.SCPPixel.Col) *
                              meta.Grid.Col.SS, grid_samples)
    coords_rg_m = np.linspace(-meta.ImageData.SCPPixel.Row * meta.Grid.Row.SS,
                              (meta.ImageData.NumRows - meta.ImageData.SCPPixel.Row) *
                              meta.Grid.Row.SS, grid_samples)
    [coords_az_m_2d, coords_rg_m_2d] = np.meshgrid(coords_az_m, coords_rg_m)
    timeca_sampled = poly.polyval2d(coords_rg_m_2d, coords_az_m_2d,
                                    np.atleast_2d(meta.RMA.INCA.TimeCAPoly))
    dopcentroid_sampled = poly.polyval2d(coords_rg_m_2d, coords_az_m_2d,
                                         meta.RMA.INCA.DopCentroidPoly)
    doprate_sampled = poly.polyval2d(coords_rg_m_2d, coords_az_m_2d,
                                     dop_rate_coefs_scaled[:, np.newaxis])
    timecoa_sampled = timeca_sampled + (dopcentroid_sampled / doprate_sampled)
    # Least squares fit for 2D polynomial
    # A*x = b
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
    b_coa = [np.multiply(timecoa_sampled.flatten(), a[:, k]).sum()
             for k in range((POLY_ORDER+1)**2)]
    x_coa = np.linalg.solve(A, b_coa)
    meta.Grid.TimeCOAPoly = np.reshape(x_coa, (POLY_ORDER+1, POLY_ORDER+1))
    if meta.CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT':
        meta.Grid.TimeCOAPoly = np.atleast_2d(meta.Grid.TimeCOAPoly[0, 0])
        # This field required to compute TimeCOAPoly, but not allowed for
        # spotlight in SICD.
        del meta.RMA.INCA.DopCentroidPoly
    else:  # This field also not allowed for spotlight in SICD.
        meta.RMA.INCA.DopCentroidCOA = True

    # GeoData
    # Now that sensor model fields have been populated, we can populate
    # GeoData.SCP more precisely.
    ecf = point.image_to_ground([meta.ImageData.SCPPixel.Row,
                                 meta.ImageData.SCPPixel.Col], meta)[0]
    meta.GeoData.SCP.ECF.X = ecf[0]
    meta.GeoData.SCP.ECF.Y = ecf[1]
    meta.GeoData.SCP.ECF.Z = ecf[2]
    llh = gc.ecf_to_geodetic(ecf)[0]
    meta.GeoData.SCP.LLH.Lat = llh[0]
    meta.GeoData.SCP.LLH.Lon = llh[1]
    meta.GeoData.SCP.LLH.HAE = llh[2]

    if betafile is not None and os.path.isfile(betafile):
        if gen == 'RS2':
            root_beta = ET.parse(betafile).getroot()
        elif gen == 'RCM':
            root_beta = _xml_parse_wo_default_ns(betafile)
        betas = np.array([float(x) for x in
                          root_beta.find('./gains').text.split()])
        meta.Radiometric = MetaNode()
        # Of the provided LUTs, we really only work with beta here, since it is
        # the radiometric term most commonly kept constant, and the others
        # (sigma/gamma) can always be derived from it.
        # The following modes are known to have constant beta:
        #    'Constant-beta', 'Point Target', 'Point Target-1', 'Calibration-1',
        #    'Calibration-2', 'Ship-1', 'Ship-2', 'Ship-3', 'Unity'
        # We could check 'product/imageGenerationParameters/sarProcessingInformation/lutApplied'
        # but this might be more reliable if we missed some modes:
        if np.all(betas == betas[0]):
            meta.Radiometric.BetaZeroSFPoly = np.atleast_2d(1/(betas[0]**2))
        else:  # Otherwise fit a 1D polynomial in range
            # RS2 has value for every row
            coords_rg_m = (np.arange(meta.ImageData.NumRows) -
                           meta.ImageData.SCPPixel.Row) * meta.Grid.Row.SS
            if gen == 'RCM':  # RCM subsamples the rows
                rng_indices = np.arange(int(root_beta.find('./pixelFirstAnglesValue').text),
                                        # Spec says 'pixelFirstLutValue', but simulated data says
                                        # 'pixelFirstAnglesValue'
                                        # float(root_beta.find('./pixelFirstLutValue').text)
                                        int(root_beta.find('./numberOfValues').text)) * \
                                        int(root_beta.find('./stepSize').text)
                coords_rg_m = coords_rg_m[rng_indices]
            # For the datasets we have seen, this function is very close to
            # linear.  For the "Mixed" LUT, there is a tiny linear piecewise
            # deviation from a single overall linear.
            meta.Radiometric.BetaZeroSFPoly = poly.polyfit(coords_rg_m, betas, 2)[:, np.newaxis]
        # RCS, Sigma, and Gamma will be computed below in sicd.derived_fields()
        if gen == 'RS2' or (noisefile is not None and os.path.isfile(noisefile)):
            meta.Radiometric.NoiseLevel = MetaNode()
            meta.Radiometric.NoiseLevel.NoiseLevelType = 'ABSOLUTE'
            if gen == 'RS2':
                # RS2 noise is in main product.xml
                beta0_element = root.find('./sourceAttributes/radarParameters/' +
                                          'referenceNoiseLevel/' +
                                          '[@incidenceAngleCorrection="Beta Nought"]')
            elif gen == 'RCM':  # RCM noise is in separate file
                root_noise = _xml_parse_wo_default_ns(noisefile)
                noise_levels = root_noise.findall('./referenceNoiseLevel')
                # Is there a cleaner way to find which noise description is beta?
                pos = next((index for index, elem in enumerate(noise_levels) if
                            elem.find('sarCalibrationType').text[:4] == 'Beta'), None)
                beta0_element = noise_levels[pos]
            pfv = float(beta0_element.find('pixelFirstNoiseValue').text)
            step = float(beta0_element.find('stepSize').text)
            beta0s = np.array([float(x) for x in
                               beta0_element.find('noiseLevelValues').text.split()])
            range_coords = meta.Grid.Row.SS * \
                ((np.arange(len(beta0s)) * step) + pfv - meta.ImageData.SCPPixel.Row)
            meta.Radiometric.NoiseLevel.NoisePoly = poly.polyfit(
                range_coords, beta0s - 10*np.log10(poly.polyval(
                    range_coords, meta.Radiometric.BetaZeroSFPoly[:, 0])), 2)[:, np.newaxis]

    # SCPCOA
    # All of these fields (and others) are derivable for more fundamental fields.
    sicd.derived_fields(meta)

    # Process fields specific to each polarimetric band
    meta_list = []
    for i in range(len(pols)):
        band_meta = copy.deepcopy(meta)  # Values that are consistent across all bands
        band_meta.ImageFormation.RcvChanProc.ChanIndex = i + 1
        band_meta.ImageFormation.TxRcvPolarizationProc = \
            band_meta.RadarCollection.RcvChannels.ChanParameters[i].TxRcvPolarization
        meta_list.append(band_meta)
    meta = meta_list  # List with metadata struct for each band

    return(meta)


def _xml_parse_wo_default_ns(filename):
    """Parse XML with ElementTree, but remove namespace."""
    # RCM uses a default namespace we will remove
    with open(filename) as f:
        xmlstring = f.read()
    # Remove the default namespace definition (xmlns="http://some/namespace") and parse
    return(ET.fromstring(re.sub('\\sxmlns="[^"]+"', '', xmlstring, count=1)))
