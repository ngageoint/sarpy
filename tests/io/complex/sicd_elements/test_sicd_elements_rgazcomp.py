#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.complex.sicd_elements import RgAzComp


def test_rgazcomp(rma_sicd, kwargs):
    rg_az = RgAzComp.RgAzCompType(None, None, **kwargs)
    rg_az._derive_parameters(rma_sicd.Grid, rma_sicd.Timeline, rma_sicd.SCPCOA)
    assert rg_az._xml_ns == kwargs['_xml_ns']
    assert rg_az._xml_ns_key == kwargs['_xml_ns_key']
    assert rg_az.AzSF is not None
    assert rg_az.KazPoly is not None
