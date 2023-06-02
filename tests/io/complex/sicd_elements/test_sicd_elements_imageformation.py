#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import re

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import ImageFormation

MIN_FREQ = 9.0e9
MAX_FREQ = 10.0e9


def test_image_formation_rcvchanproc(kwargs):
    # Basic smoke test
    rcv_chan_proc = ImageFormation.RcvChanProcType(
        NumChanProc=1, PRFScaleFactor=1.2, ChanIndices=[1, 2], **kwargs
    )
    assert rcv_chan_proc._xml_ns == kwargs["_xml_ns"]
    assert rcv_chan_proc._xml_ns_key == kwargs["_xml_ns_key"]
    assert rcv_chan_proc.NumChanProc == 1
    assert rcv_chan_proc.PRFScaleFactor == 1.2
    assert rcv_chan_proc.ChanIndices == [1, 2]


def test_image_formation_txfreq(caplog, tol, kwargs):
    tx_freq = ImageFormation.TxFrequencyProcType(
        MinProc=MIN_FREQ, MaxProc=MAX_FREQ, **kwargs
    )
    assert tx_freq.center_frequency == pytest.approx(
        np.mean([MIN_FREQ, MAX_FREQ]), abs=tol
    )
    assert tx_freq.bandwidth == pytest.approx(MAX_FREQ - MIN_FREQ, abs=tol)

    assert tx_freq._basic_validity_check()
    assert tx_freq.get_band_name() == "X"
    tx_freq_array = tx_freq.get_array()
    assert np.all(tx_freq_array == np.array([MIN_FREQ, MAX_FREQ]))

    # Test from_array paths
    assert tx_freq.from_array(None) is None
    tx_freq1 = tx_freq.from_array(tx_freq_array)
    assert tx_freq1.MinProc == tx_freq.MinProc
    assert tx_freq1.MaxProc == tx_freq.MaxProc

    with pytest.raises(
        ValueError,
        match=re.escape("Expected array to be of length 2, and received [1]"),
    ):
        tx_freq.from_array([1])

    with pytest.raises(
        ValueError,
        match="Expected array to be numpy.ndarray, list, or tuple, got <class 'dict'>",
    ):
        tx_freq.from_array({"1": 1})

    tx_freq._apply_reference_frequency(100000)
    assert tx_freq.MinProc == MIN_FREQ + 100000
    assert tx_freq.MaxProc == MAX_FREQ + 100000

    tx_freq = ImageFormation.TxFrequencyProcType(MinProc=None, MaxProc=MAX_FREQ)
    assert tx_freq.center_frequency is None
    assert tx_freq.bandwidth is None

    # Test invalid inputs path
    tx_freq = ImageFormation.TxFrequencyProcType(MinProc=MAX_FREQ, MaxProc=MIN_FREQ)
    assert not tx_freq._basic_validity_check()
    assert (
        f"Invalid frequency bounds MinProc ({tx_freq.MinProc}) > MaxProc ({tx_freq.MaxProc})"
        in caplog.text
    )


def test_image_formation_processing(kwargs):
    # Basic smoke test
    proc_type = ImageFormation.ProcessingType(Type="PFA", Applied="True", **kwargs)
    assert proc_type._xml_ns == kwargs["_xml_ns"]
    assert proc_type._xml_ns_key == kwargs["_xml_ns_key"]
    assert proc_type.Type == "PFA"
    assert proc_type.Applied is True


def test_image_formation_distortion(kwargs):
    # Basic smoke test
    distortion = ImageFormation.DistortionType(
        A=1.0,
        F1=complex(1, 2),
        Q1=complex(3, 4),
        Q2=complex(5, 6),
        F2=complex(7, 8),
        Q3=complex(9, 1),
        Q4=complex(1, 3),
        **kwargs,
    )
    assert distortion._xml_ns == kwargs["_xml_ns"]
    assert distortion._xml_ns_key == kwargs["_xml_ns_key"]


def test_image_formation_polcal(kwargs):
    # Basic smoke test
    distortion = ImageFormation.DistortionType(
        A=1.0,
        F1=complex(1, 2),
        Q1=complex(3, 4),
        Q2=complex(5, 6),
        F2=complex(7, 8),
        Q3=complex(9, 1),
        Q4=complex(1, 3),
    )

    pol_cal_type = ImageFormation.PolarizationCalibrationType(
        DistortCorrectApplied=True, Distortion=distortion, **kwargs
    )
    assert pol_cal_type._xml_ns == kwargs["_xml_ns"]
    assert pol_cal_type._xml_ns_key == kwargs["_xml_ns_key"]


def test_image_formation(sicd, caplog, kwargs):
    # Test basic validity paths
    image_form_type = ImageFormation.ImageFormationType(
        RcvChanProc=sicd.ImageFormation.RcvChanProc,
        TxRcvPolarizationProc=sicd.ImageFormation.TxRcvPolarizationProc,
        TStartProc=sicd.ImageFormation.TStartProc,
        TEndProc=sicd.ImageFormation.TEndProc,
        TxFrequencyProc=sicd.ImageFormation.TxFrequencyProc,
        ImageFormAlgo=sicd.ImageFormation.ImageFormAlgo,
        STBeamComp=sicd.ImageFormation.STBeamComp,
        ImageBeamComp=sicd.ImageFormation.ImageBeamComp,
        AzAutofocus=sicd.ImageFormation.AzAutofocus,
        RgAutofocus=sicd.ImageFormation.RgAutofocus,
        **kwargs,
    )
    assert image_form_type._xml_ns == kwargs["_xml_ns"]
    assert image_form_type._xml_ns_key == kwargs["_xml_ns_key"]

    assert image_form_type._basic_validity_check()

    image_form_type.TStartProc = image_form_type.TEndProc + 1
    image_form_type._basic_validity_check()
    assert (
        f"Invalid time processing bounds TStartProc ({image_form_type.TStartProc}) > TEndProc ({image_form_type.TEndProc})"
        in caplog.text
    )

    # Test derive TxFrequencyProc paths
    image_form_type = ImageFormation.ImageFormationType(
        RcvChanProc=sicd.ImageFormation.RcvChanProc,
        TxRcvPolarizationProc=sicd.ImageFormation.TxRcvPolarizationProc,
        TStartProc=sicd.ImageFormation.TStartProc,
        TEndProc=sicd.ImageFormation.TEndProc,
        TxFrequencyProc=None,
        ImageFormAlgo=sicd.ImageFormation.ImageFormAlgo,
        STBeamComp=sicd.ImageFormation.STBeamComp,
        ImageBeamComp=sicd.ImageFormation.ImageBeamComp,
        AzAutofocus=sicd.ImageFormation.AzAutofocus,
        RgAutofocus=sicd.ImageFormation.RgAutofocus,
    )
    image_form_type._derive_tx_frequency_proc(sicd.RadarCollection)
    assert image_form_type.TxFrequencyProc is not None

    image_form_type = ImageFormation.ImageFormationType(
        RcvChanProc=sicd.ImageFormation.RcvChanProc,
        TxRcvPolarizationProc=sicd.ImageFormation.TxRcvPolarizationProc,
        TStartProc=sicd.ImageFormation.TStartProc,
        TEndProc=sicd.ImageFormation.TEndProc,
        TxFrequencyProc=sicd.ImageFormation.TxFrequencyProc,
        ImageFormAlgo=sicd.ImageFormation.ImageFormAlgo,
        STBeamComp=sicd.ImageFormation.STBeamComp,
        ImageBeamComp=sicd.ImageFormation.ImageBeamComp,
        AzAutofocus=sicd.ImageFormation.AzAutofocus,
        RgAutofocus=sicd.ImageFormation.RgAutofocus,
    )
    image_form_type.TxFrequencyProc.MinProc = None
    image_form_type._derive_tx_frequency_proc(sicd.RadarCollection)
    assert (
        image_form_type.TxFrequencyProc.MinProc == sicd.RadarCollection.TxFrequency.Min
    )
    image_form_type.TxFrequencyProc.MaxProc = None
    image_form_type._derive_tx_frequency_proc(sicd.RadarCollection)
    assert (
        image_form_type.TxFrequencyProc.MaxProc == sicd.RadarCollection.TxFrequency.Max
    )

    image_form_type._apply_reference_frequency(100000)
    assert (
        image_form_type.TxFrequencyProc.MinProc
        == sicd.RadarCollection.TxFrequency.Min + 100000
    )
    assert (
        image_form_type.TxFrequencyProc.MaxProc
        == sicd.RadarCollection.TxFrequency.Max + 100000
    )

    assert (
        image_form_type.get_polarization() == sicd.ImageFormation.TxRcvPolarizationProc
    )

    # Test paths through get_polarization_abbreviation
    assert image_form_type.get_polarization_abbreviation() == "VV"
    image_form_type.TxRcvPolarizationProc = "OTHER"
    assert image_form_type.get_polarization_abbreviation() == "UN"

    # Test paths through get_transmit_band_name
    assert image_form_type.get_transmit_band_name() == "X"
    image_form_type.TxFrequencyProc = None
    assert image_form_type.get_transmit_band_name() == "UN"

    assert image_form_type.version_required() == (1, 1, 0)

    # Force the TxFrequencyProc from list path
    image_form_type = ImageFormation.ImageFormationType(
        RcvChanProc=sicd.ImageFormation.RcvChanProc,
        TxRcvPolarizationProc=sicd.ImageFormation.TxRcvPolarizationProc,
        TStartProc=sicd.ImageFormation.TStartProc,
        TEndProc=sicd.ImageFormation.TEndProc,
        TxFrequencyProc=[MIN_FREQ, MAX_FREQ],
        ImageFormAlgo=sicd.ImageFormation.ImageFormAlgo,
        STBeamComp=sicd.ImageFormation.STBeamComp,
        ImageBeamComp=sicd.ImageFormation.ImageBeamComp,
        AzAutofocus=sicd.ImageFormation.AzAutofocus,
        RgAutofocus=sicd.ImageFormation.RgAutofocus,
    )
    assert image_form_type.TxFrequencyProc.MinProc == MIN_FREQ
    assert image_form_type.TxFrequencyProc.MaxProc == MAX_FREQ
