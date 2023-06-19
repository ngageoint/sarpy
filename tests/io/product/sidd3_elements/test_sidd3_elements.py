#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import logging

import numpy as np
import pytest

import sarpy.io.complex.sicd_elements.Radiometric as SicdRadiometric
from sarpy.io.product.sidd3_elements import blocks


def test_anglezeroto360_nominal(caplog):
    with caplog.at_level(logging.INFO, 'sarpy.io.xml.descriptors'):
        angmag = blocks.AngleZeroToExclusive360MagnitudeType(12.34, 56.78)
        np.testing.assert_array_equal(angmag.get_array(), [12.34, 56.78])
        assert angmag.Angle == 12.34
        assert angmag.Magnitude == 56.78

        angmag2 = blocks.AngleZeroToExclusive360MagnitudeType.from_array(angmag.get_array())
        np.testing.assert_array_equal(angmag.get_array(), angmag2.get_array())

        assert not caplog.records


@pytest.mark.parametrize('angle', (-1, 361))
def test_anglezeroto360_bad_angle(angle, caplog):
    with caplog.at_level(logging.INFO, 'sarpy.io.xml.descriptors'):
        angmag = blocks.AngleZeroToExclusive360MagnitudeType(angle, 56.78)
        assert len(caplog.records) == 1
        assert 'required by standard to take value between (0.0, 360).' in caplog.text


def test_radiometric(caplog):
    with caplog.at_level(logging.INFO, 'sarpy.io.xml.descriptors'):
        new_fields = set(blocks.RadiometricType._fields) - set(SicdRadiometric.RadiometricType._fields)
        assert new_fields == {'SigmaZeroSFIncidenceMap'}

        rad = blocks.RadiometricType()
        assert rad.SigmaZeroSFIncidenceMap is None

        rad = blocks.RadiometricType(SigmaZeroSFIncidenceMap='APPLIED')
        assert rad.SigmaZeroSFIncidenceMap == 'APPLIED'

        rad = blocks.RadiometricType(SigmaZeroSFIncidenceMap='NOT_APPLIED')
        assert rad.SigmaZeroSFIncidenceMap == 'NOT_APPLIED'

        assert not caplog.records


def test_radiometric_invalid(caplog):
    with caplog.at_level(logging.INFO, 'sarpy.io.xml.descriptors'):
        rad = blocks.RadiometricType(SigmaZeroSFIncidenceMap='invalid')
        assert rad.SigmaZeroSFIncidenceMap == 'invalid'
        assert len(caplog.records) == 1
        assert "values ARE REQUIRED to be one of ('APPLIED', 'NOT_APPLIED')" in caplog.text
