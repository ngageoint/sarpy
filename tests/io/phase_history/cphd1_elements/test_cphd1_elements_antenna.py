#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.phase_history.cphd1_elements import Antenna


def test_antenna_freqsftype(cphd):
    """Test FreqSFType class"""
    sf_list = [cphd.Antenna.AntPattern[0].EBFreqShiftSF.DCXSF,
               cphd.Antenna.AntPattern[0].EBFreqShiftSF.DCYSF]
    freq_sf_type = Antenna.FreqSFType(
        DCXSF=sf_list[0],
        DCYSF=sf_list[1],
    )
    assert np.all(freq_sf_type.get_array() == np.array(sf_list))

    assert freq_sf_type.from_array(None) is None

    new_sf = [1, 1]
    new_freq_sf_type = freq_sf_type.from_array(new_sf)
    assert np.all(new_freq_sf_type.get_array() == np.array(new_sf))

    with pytest.raises(ValueError, match="Expected array to be of length 2"):
        baf_sf = [1]
        freq_sf_type.from_array(baf_sf)

    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        baf_sf = {1}
        freq_sf_type.from_array(baf_sf)


def test_antenna_antpolreftype():
    """Test AntPolRefType class"""
    ant_pol_ref_type = Antenna.AntPolRefType(
        AmpX=0.0,
        AmpY=1.0,
        PhaseY=2.0,
    )
    assert np.all(ant_pol_ref_type.get_array() == np.array([0.0, 1.0, 2.0]))

    assert ant_pol_ref_type.from_array(None) is None

    new_ant_pol_ref = [1, 2, 3]
    new_ant_pol_ref_type = ant_pol_ref_type.from_array(new_ant_pol_ref)
    assert np.all(new_ant_pol_ref_type.get_array() == np.array(new_ant_pol_ref))

    with pytest.raises(ValueError, match="Expected array to be of length 3"):
        baf_ant_pol_ref = [1, 2]
        ant_pol_ref_type.from_array(baf_ant_pol_ref)

    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        baf_ant_pol_ref = {1}
        ant_pol_ref_type.from_array(baf_ant_pol_ref)


def test_antenna_ebtype():
    """Test EBType class"""
    eb_type = Antenna.EBType(
        DCXPoly=None,
        DCYPoly=np.zeros(4),
        UseEBPVP=True,
    )
    assert eb_type(0) is None

    eb_type = Antenna.EBType(
        DCXPoly=np.ones(4),
        DCYPoly=np.zeros(4),
        UseEBPVP=True,
    )
    assert np.all(eb_type(0) == np.array([1.0, 0.0]))


def test_antenna_gainphasepolytype():
    """Test GainPhasePolyType class"""
    gain_phase_poly_type = Antenna.GainPhasePolyType(
        GainPoly=None,
        PhasePoly=np.zeros((3, 4)),
        AntGPId="xmit",
    )
    assert gain_phase_poly_type(0, 0) is None

    gain_phase_poly_type = Antenna.GainPhasePolyType(
        GainPoly=np.ones((3, 4)),
        PhasePoly=np.zeros((3, 4)),
        AntGPId="xmit",
    )
    assert np.all(gain_phase_poly_type(0, 0) == np.array([1.0, 0.0]))

    assert len(gain_phase_poly_type.PhasePoly.Coefs) == 3
    gain_phase_poly_type.minimize_order()
    assert len(gain_phase_poly_type.PhasePoly.Coefs) == 1


def test_antenna_antennatype():
    """Test AntennaType class"""
    antenna_type = Antenna.AntennaType()
    assert antenna_type.NumACFs == 0
    assert antenna_type.NumAPCs == 0
    assert antenna_type.NumAntPats == 0

    expected_num_acfs = 2
    antenna_type.AntCoordFrame = [
        Antenna.AntCoordFrameType(Identifier=str(x), XAxisPoly=np.zeros((3, 4)), YAxisPoly=np.zeros((3, 4)))
        for x in range(expected_num_acfs)
    ]
    assert antenna_type.NumACFs == expected_num_acfs

    expected_num_apcs = 2
    antenna_type.AntPhaseCenter = [
        Antenna.AntPhaseCenterType(Identifier=str(x), ACFId="ACF"+str(x), APCXYZ=np.zeros(3))
        for x in range(expected_num_apcs)
    ]
    assert antenna_type.NumAPCs == expected_num_apcs

    expected_num_arrays = 2
    arrays = [
        Antenna.GainPhasePolyType(GainPoly=np.zeros((3, 4)),
                                  PhasePoly=np.ones((3, 4))),
        Antenna.GainPhasePolyType(GainPoly=np.ones((3, 4)),
                                  PhasePoly=np.zeros((3, 4)))
    ]
    antenna_type.AntPattern = [
        Antenna.AntPatternType(Identifier=str(x), Array=arrays[x])
        for x in range(len(arrays))
    ]
    assert antenna_type.NumAntPats == expected_num_arrays

