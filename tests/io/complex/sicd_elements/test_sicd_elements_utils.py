#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import copy

from sarpy.io.complex.sicd_elements import utils
from sarpy.io.complex.sicd_elements import validation_checks


def test_validation_checks(sicd):
    """Smoke test with PFA SICD"""
    validation_checks.detailed_validation_checks(sicd)


def test_validation_checks_with_rma(rma_sicd):
    """Smoke test with RMA SICD"""
    validation_checks.detailed_validation_checks(rma_sicd)


def test_utils(sicd, rma_sicd, caplog):
    """Check sicd_elements utility functions"""
    assert utils.is_same_sensor(sicd, rma_sicd)
    assert utils.is_same_sensor(sicd, sicd)

    assert utils.is_same_start_time(sicd, rma_sicd)
    assert utils.is_same_start_time(sicd, sicd)

    assert not utils.is_same_size(sicd, rma_sicd)
    assert utils.is_same_size(sicd, sicd)

    assert utils.is_same_duration(sicd, rma_sicd)
    assert utils.is_same_duration(sicd, sicd)

    assert utils.is_same_scp(sicd, rma_sicd)
    assert utils.is_same_scp(sicd, sicd)

    assert not utils.is_same_band(sicd, rma_sicd)
    assert utils.is_same_band(sicd, sicd)

    assert not utils.is_general_match(sicd, rma_sicd)
    assert utils.is_general_match(sicd, sicd)

    pol = utils.polstring_version_required(None)
    assert pol == (1, 1, 0)
    pol = utils.polstring_version_required('V:V:H')
    assert 'Expected polarization string of length 2, but populated as `3`' in caplog.text
    assert pol is None
    pol = utils.polstring_version_required('V')
    assert 'Expected polarization string of length 2, but populated as `1`' in caplog.text
    assert pol is None
    pol = utils.polstring_version_required('S:V')
    assert pol == (1, 3, 0)
    pol = utils.polstring_version_required('H:X')
    assert pol == (1, 3, 0)
    pol = utils.polstring_version_required('V:RHC')
    assert pol == (1, 2, 1)
    pol = utils.polstring_version_required('LHC:H')
    assert pol == (1, 2, 1)
    pol = utils.polstring_version_required('V:H')
    assert pol == (1, 1, 0)
    pol = utils.polstring_version_required('OTHER:H')
    assert pol == (1, 3, 0)
    pol = utils.polstring_version_required('H:OTHERpol')
    assert pol == (1, 3, 0)

    # Must have both ImageFormation and RadarCollection
    freq = utils._get_center_frequency(None, sicd.ImageFormation)
    assert freq is None
    freq = utils._get_center_frequency(sicd.RadarCollection, None)
    assert freq is None

    # Use a copy to change RefFreqIndex value
    radar_collection = copy.copy(sicd.RadarCollection)
    radar_collection.RefFreqIndex = 10
    freq = utils._get_center_frequency(radar_collection, sicd.ImageFormation)
    assert freq is None

    radar_collection.RefFreqIndex = None
    freq = utils._get_center_frequency(radar_collection, sicd.ImageFormation)
    assert freq == sicd.ImageFormation.TxFrequencyProc.center_frequency
