#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.complex.sicd_elements import Radiometric


def test_radiometric(sicd, rma_sicd, kwargs):
    noise_level = Radiometric.NoiseLevelType_()
    assert noise_level.NoiseLevelType is None
    assert noise_level.NoisePoly is None

    noise_level = Radiometric.NoiseLevelType_('ABSOLUTE')
    assert noise_level.NoiseLevelType == 'ABSOLUTE'
    assert noise_level.NoisePoly is None

    noise_level = Radiometric.NoiseLevelType_(None, rma_sicd.Radiometric.NoiseLevel.NoisePoly, **kwargs)
    assert noise_level.NoiseLevelType == 'RELATIVE'

    noise_level = Radiometric.NoiseLevelType_(None, sicd.Radiometric.NoiseLevel.NoisePoly, **kwargs)
    assert noise_level._xml_ns == kwargs['_xml_ns']
    assert noise_level._xml_ns_key == kwargs['_xml_ns_key']
    assert noise_level.NoiseLevelType == 'ABSOLUTE'
    assert noise_level.NoisePoly == sicd.Radiometric.NoiseLevel.NoisePoly

    radio_type = Radiometric.RadiometricType()
    assert radio_type.NoiseLevel is None
    assert radio_type.RCSSFPoly is None
    assert radio_type.SigmaZeroSFPoly is None
    assert radio_type.BetaZeroSFPoly is None
    assert radio_type.GammaZeroSFPoly is None

    radio_type = Radiometric.RadiometricType(noise_level, sicd.Radiometric.RCSSFPoly)
    assert radio_type.NoiseLevel == noise_level
    assert radio_type.RCSSFPoly == sicd.Radiometric.RCSSFPoly

    radio_type._derive_parameters(sicd.Grid, sicd.SCPCOA)
    assert radio_type.SigmaZeroSFPoly is not None
    assert radio_type.BetaZeroSFPoly is not None
    assert radio_type.GammaZeroSFPoly is not None

    radio_type = Radiometric.RadiometricType(noise_level)
    assert radio_type.NoiseLevel == noise_level
    assert radio_type.RCSSFPoly is None
    assert radio_type.SigmaZeroSFPoly is None
    assert radio_type.BetaZeroSFPoly is None
    assert radio_type.GammaZeroSFPoly is None

    radio_type._derive_parameters(sicd.Grid, sicd.SCPCOA)
    assert radio_type.NoiseLevel == noise_level
    assert radio_type.RCSSFPoly is None
    assert radio_type.SigmaZeroSFPoly is None
    assert radio_type.BetaZeroSFPoly is None
    assert radio_type.GammaZeroSFPoly is None
