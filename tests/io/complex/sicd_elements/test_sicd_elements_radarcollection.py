#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import RadarCollection


def test_radarcollection_getbandname():
    assert RadarCollection.get_band_name(None) == 'UN'
    assert RadarCollection.get_band_name(3.5e6) == 'HF'
    assert RadarCollection.get_band_name(3.5e7) == 'VHF'
    assert RadarCollection.get_band_name(3.5e8) == 'UHF'
    assert RadarCollection.get_band_name(1.5e9) == 'L'
    assert RadarCollection.get_band_name(2.5e9) == 'S'
    assert RadarCollection.get_band_name(4.5e9) == 'C'
    assert RadarCollection.get_band_name(8.5e9) == 'X'
    assert RadarCollection.get_band_name(1.5e10) == 'KU'
    assert RadarCollection.get_band_name(2.5e10) == 'K'
    assert RadarCollection.get_band_name(3.0e10) == 'KA'
    assert RadarCollection.get_band_name(6.0e10) == 'V'
    assert RadarCollection.get_band_name(1.0e11) == 'W'
    assert RadarCollection.get_band_name(2.0e11) == 'MM'
    assert RadarCollection.get_band_name(5.0e11) == 'UN'


def test_radarcollection_txfreqtype(sicd, kwargs, caplog):

    bad_tx_freq = RadarCollection.TxFrequencyType(None,
                                                  sicd.RadarCollection.TxFrequency.Max,
                                                  **kwargs)
    assert bad_tx_freq.center_frequency is None

    bad_tx_freq = RadarCollection.TxFrequencyType(sicd.RadarCollection.TxFrequency.Max,
                                                  sicd.RadarCollection.TxFrequency.Min,
                                                  **kwargs)
    assert not bad_tx_freq._basic_validity_check()
    assert 'Invalid frequency bounds Min ({}) > Max ({})'.format(bad_tx_freq.Min, bad_tx_freq.Max) in caplog.text

    tx_freq = RadarCollection.TxFrequencyType(sicd.RadarCollection.TxFrequency.Min,
                                              sicd.RadarCollection.TxFrequency.Max,
                                              **kwargs)
    assert tx_freq._xml_ns == kwargs['_xml_ns']
    assert tx_freq._xml_ns_key == kwargs['_xml_ns_key']
    assert tx_freq.Min == sicd.RadarCollection.TxFrequency.Min
    assert tx_freq.Max == sicd.RadarCollection.TxFrequency.Max
    assert tx_freq.center_frequency == 0.5 * (tx_freq.Min + tx_freq.Max)

    tx_freq._apply_reference_frequency(1000)
    assert tx_freq.Min == sicd.RadarCollection.TxFrequency.Min + 1000
    assert tx_freq.Max == sicd.RadarCollection.TxFrequency.Max + 1000
    assert tx_freq._basic_validity_check()
    assert tx_freq.get_band_abbreviation() == 'X__'

    tx_freq_arr = tx_freq.get_array()
    assert np.all(tx_freq_arr == np.array([tx_freq.Min, tx_freq.Max]))

    assert tx_freq.from_array(None) is None
    tx_freq1 = tx_freq.from_array(tx_freq_arr)
    assert tx_freq1.Min == tx_freq.Min
    assert tx_freq1.Max == tx_freq.Max

    with pytest.raises(ValueError, match='Expected array to be of length 2, and received 1'):
        tx_freq.from_array([tx_freq_arr[0]])
    with pytest.raises(ValueError, match='Expected array to be numpy.ndarray, list, or tuple'):
        tx_freq.from_array(tx_freq)


wf_input = [(1.0, 2.0, None), (1.0, None, 2.0), (None, 1.0, 2.0)]


@pytest.mark.parametrize("tx_pulse_len, tx_rf_bw, tx_fm_rate", wf_input)
def test_radarcollection_waveformparamtype(tx_pulse_len, tx_rf_bw, tx_fm_rate):
    wf_params = RadarCollection.WaveformParametersType(RcvDemodType='STRETCH', RcvFMRate=1.0, index=0)
    assert wf_params.RcvFMRate == 1.0
    assert wf_params.RcvDemodType == 'STRETCH'
    wf_params.RcvFMRate = None
    assert wf_params.RcvFMRate is None
    wf_params.RcvFMRate = 12.0
    assert wf_params.RcvFMRate == 12.0
    assert wf_params._basic_validity_check()

    wf_params = RadarCollection.WaveformParametersType(RcvDemodType='CHIRP', RcvFMRate=None, index=1)
    assert wf_params.RcvFMRate == 0.0
    assert wf_params.RcvDemodType == 'CHIRP'

    wf_params = RadarCollection.WaveformParametersType(RcvDemodType='cHiRp', RcvFMRate=None, index=2)
    assert wf_params.RcvFMRate is None

    tx_freq_start = 3.0
    rcv_freq_start = 2.0
    wf_params = RadarCollection.WaveformParametersType(TxPulseLength=tx_pulse_len,
                                                       TxRFBandwidth=tx_rf_bw,
                                                       TxFreqStart=tx_freq_start,
                                                       TxFMRate=tx_fm_rate,
                                                       RcvFreqStart=rcv_freq_start)
    wf_params.derive()
    assert np.all([wf_params.TxRFBandwidth, wf_params.TxFMRate, wf_params.TxPulseLength] is not None)

    ref_freq = 10.5
    wf_params._apply_reference_frequency(ref_freq)
    assert wf_params.TxFreqStart == tx_freq_start + ref_freq
    assert wf_params.RcvFreqStart == rcv_freq_start + ref_freq


def test_radarcollection_txsteptype(kwargs):
    tx_step = RadarCollection.TxStepType(1, 'V', 1, **kwargs)
    assert tx_step._xml_ns == kwargs['_xml_ns']
    assert tx_step._xml_ns_key == kwargs['_xml_ns_key']


def test_radarcollection_chanparameterstype(kwargs):
    chan_params = RadarCollection.ChanParametersType(None, 1, 1)
    assert chan_params.get_transmit_polarization() is None

    chan_params = RadarCollection.ChanParametersType('OTHER', 1, 1)
    assert chan_params.get_transmit_polarization() == 'OTHER'

    chan_params = RadarCollection.ChanParametersType('V:H', 1, 1, **kwargs)
    assert chan_params._xml_ns == kwargs['_xml_ns']
    assert chan_params._xml_ns_key == kwargs['_xml_ns_key']
    assert chan_params.get_transmit_polarization() == 'V'
    assert chan_params.version_required() == (1, 1, 0)


def test_radarcollection_segmentarrayelement(kwargs):
    seg_arr_elem = RadarCollection.SegmentArrayElement(0, 0, 2000, 5000, 'AA', 1, **kwargs)
    assert seg_arr_elem._xml_ns == kwargs['_xml_ns']
    assert seg_arr_elem._xml_ns_key == kwargs['_xml_ns_key']


def test_radarcollection_referenceplanetype(sicd, kwargs):
    seg_arr_elem1 = RadarCollection.SegmentArrayElement(0, 0, 500, 1501, 'XY', 1)
    seg_arr_elem2 = RadarCollection.SegmentArrayElement(501, 0, 1301, 1501, 'XZ', 2)

    ref_plane = RadarCollection.ReferencePlaneType(sicd.RadarCollection.Area.Plane.RefPt,
                                                   sicd.RadarCollection.Area.Plane.XDir,
                                                   sicd.RadarCollection.Area.Plane.YDir,
                                                   [seg_arr_elem1, seg_arr_elem2],
                                                   'D',
                                                   **kwargs)
    assert ref_plane._xml_ns == kwargs['_xml_ns']
    assert ref_plane._xml_ns_key == kwargs['_xml_ns_key']
    corners = ref_plane.get_ecf_corner_array()
    assert np.all(corners is not None)

    area = RadarCollection.AreaType(Corner=None, Plane=ref_plane)
    assert area.Corner is not None


def test_radarcollection_getpolabbr():
    chan_params1 = RadarCollection.ChanParametersType('V:V', 1, 1)
    chan_params2 = RadarCollection.ChanParametersType('S:H', 2, 2)

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=None)
    assert radar_collection.get_polarization_abbreviation() == 'U'

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=chan_params1)
    assert radar_collection.get_polarization_abbreviation() == 'S'

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=[chan_params1, chan_params2])
    assert radar_collection.get_polarization_abbreviation() == 'D'

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=[chan_params1, chan_params2, chan_params1])
    assert radar_collection.get_polarization_abbreviation() == 'T'

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=[chan_params1, chan_params2, chan_params1, chan_params2])
    assert radar_collection.get_polarization_abbreviation() == 'Q'


def test_radarcollection_derive(sicd):
    wf_params = RadarCollection.WaveformParametersType(TxPulseLength=1.0,
                                                       TxRFBandwidth=2.0,
                                                       TxFreqStart=3.0,
                                                       TxFMRate=4.0,
                                                       RcvDemodType='STRETCH',
                                                       RcvWindowLength=5.0,
                                                       ADCSampleRate=6.0,
                                                       RcvIFBandwidth=7.0,
                                                       RcvFreqStart=0.0,
                                                       RcvFMRate=1.0,
                                                       index=1)
    area = RadarCollection.AreaType(Plane=sicd.RadarCollection.Area.Plane)
    tx_step1 = RadarCollection.TxStepType(1, 'V', 1)
    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=sicd.RadarCollection.RcvChannels,
                                                           Area=area,
                                                           Waveform=wf_params,
                                                           TxSequence=[tx_step1])
    radar_collection.derive()
    assert radar_collection.TxPolarization == 'V'

    tx_step2 = RadarCollection.TxStepType(2, 'H', 2)
    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=sicd.RadarCollection.RcvChannels,
                                                           Area=area,
                                                           Waveform=wf_params,
                                                           TxSequence=[tx_step1, tx_step2])
    radar_collection.derive()
    assert radar_collection.TxPolarization == 'SEQUENCE'

    chan_params1 = RadarCollection.ChanParametersType('V:V', 1, 1)
    chan_params2 = RadarCollection.ChanParametersType('S:H', 2, 2)
    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=[chan_params1, chan_params2],
                                                           Area=area,
                                                           Waveform=wf_params)
    radar_collection.derive()
    assert radar_collection.TxPolarization == 'SEQUENCE'

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=chan_params1,
                                                           Area=area,
                                                           Waveform=wf_params)
    radar_collection.derive()
    assert radar_collection.TxPolarization == 'V'


def test_radarcollection_version(sicd):
    chan_params1 = RadarCollection.ChanParametersType('V:V', 1, 1)
    chan_params2 = RadarCollection.ChanParametersType('S:H', 2, 2)

    # Check SICD version requirements based on RcvChannels
    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=None)
    assert radar_collection.version_required() == (1, 1, 0)
    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=[chan_params1, chan_params2])
    assert radar_collection.version_required() == (1, 3, 0)


def test_radarcollection_smoketest(sicd, kwargs):
    params = {'fake_params': 'this_fake_str',
              'fake_params1': 'another_fake_str'}
    ref_freq_idx = 0

    radar_collection = RadarCollection.RadarCollectionType(sicd.RadarCollection.TxFrequency,
                                                           ref_freq_idx,
                                                           sicd.RadarCollection.Waveform,
                                                           sicd.RadarCollection.TxPolarization,
                                                           RadarCollection.TxStepType(TxPolarization='V'),
                                                           sicd.RadarCollection.RcvChannels,
                                                           sicd.RadarCollection.Area,
                                                           params,
                                                           **kwargs)
    assert radar_collection._xml_ns == kwargs['_xml_ns']
    assert radar_collection._xml_ns_key == kwargs['_xml_ns_key']


def test_radarcollection_private(sicd, caplog):
    wf_params = RadarCollection.WaveformParametersType(TxPulseLength=1.0,
                                                       TxRFBandwidth=2.0,
                                                       TxFreqStart=3.0,
                                                       TxFMRate=4.0,
                                                       RcvDemodType='STRETCH',
                                                       RcvWindowLength=5.0,
                                                       ADCSampleRate=6.0,
                                                       RcvIFBandwidth=7.0,
                                                       RcvFreqStart=0.0,
                                                       RcvFMRate=1.0,
                                                       index=1)
    area = RadarCollection.AreaType(Plane=sicd.RadarCollection.Area.Plane)
    chan_params1 = RadarCollection.ChanParametersType('V:V', 1, 1)
    wf_params.TxFreqStart = None
    wf_params.TxRFBandwidth = None
    tx_freq = RadarCollection.TxFrequencyType(sicd.RadarCollection.TxFrequency.Min,
                                              sicd.RadarCollection.TxFrequency.Max)
    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=chan_params1,
                                                           TxFrequency=tx_freq,
                                                           Area=area,
                                                           Waveform=wf_params)
    radar_collection._derive_wf_params()
    assert radar_collection.Waveform[0].TxFreqStart == sicd.RadarCollection.TxFrequency.Min
    assert radar_collection.Waveform[0].TxRFBandwidth == sicd.RadarCollection.TxFrequency.Max - sicd.RadarCollection.TxFrequency.Min

    ref_freq = 10000
    radar_collection._apply_reference_frequency(ref_freq)
    assert radar_collection.TxFrequency.Min == sicd.RadarCollection.TxFrequency.Min + ref_freq
    assert radar_collection.TxFrequency.Max == sicd.RadarCollection.TxFrequency.Max + ref_freq
    assert radar_collection.Waveform[0].TxFreqStart == sicd.RadarCollection.TxFrequency.Min + ref_freq

    assert radar_collection._check_frequency()
    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=chan_params1,
                                                           TxFrequency=tx_freq,
                                                           Area=area,
                                                           Waveform=wf_params)
    assert radar_collection._check_frequency()

    radar_collection.TxFrequency.Min *= -1
    assert not radar_collection._check_frequency()
    assert "TxFrequency.Min is negative, but RefFreqIndex is not populated." in caplog.text
    caplog.clear()

    radar_collection.RefFreqIndex = 10
    assert radar_collection._check_frequency()

    assert radar_collection._check_tx_sequence()

    radar_collection.TxPolarization = 'SEQUENCE'
    radar_collection.TxSequence = None
    assert not radar_collection._check_tx_sequence()
    assert 'TxPolarization is populated as "SEQUENCE", but TxSequence is not populated.' in caplog.text
    caplog.clear()

    tx_step1 = RadarCollection.TxStepType(1, 'V', 1)
    tx_step2 = RadarCollection.TxStepType(2, 'H', 2)

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=chan_params1,
                                                           Area=area,
                                                           Waveform=wf_params,
                                                           TxSequence=[tx_step1, tx_step2])
    radar_collection.TxPolarization = 'V'
    assert not radar_collection._check_tx_sequence()
    assert 'TxSequence is populated, but TxPolarization is populated as {}'.format(radar_collection.TxPolarization) in caplog.text
    caplog.clear()

    radar_collection = RadarCollection.RadarCollectionType(RcvChannels=chan_params1,
                                                           Area=area,
                                                           Waveform=wf_params,
                                                           TxSequence=[tx_step1, tx_step1])
    radar_collection.TxPolarization = 'SEQUENCE'
    assert not radar_collection._check_tx_sequence()
    assert 'TxSequence is populated, but the only unique TxPolarization' in caplog.text
    caplog.clear()

    radar_collection._basic_validity_check()
