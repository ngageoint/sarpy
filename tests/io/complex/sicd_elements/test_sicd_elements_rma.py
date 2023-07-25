#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import pytest

from sarpy.io.complex.sicd_elements import blocks
from sarpy.io.complex.sicd_elements import RMA


TIME_CA_POLY = blocks.Poly1DType([1, 2, 3, 4])
DRATE_SF_POLY = blocks.Poly2DType([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
DOP_CENT_POLY = blocks.Poly2DType([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
R_CA_SCP = 10000.0
FREQ_ZERO = 100000.0


@pytest.fixture()
def rm_ref_type(rma_sicd):
    return RMA.RMRefType(
        PosRef=rma_sicd.RMA.RMCR.PosRef,
        VelRef=rma_sicd.RMA.RMCR.VelRef,
        DopConeAngRef=rma_sicd.RMA.RMCR.DopConeAngRef,
    )


@pytest.fixture()
def inca_type():
    return RMA.INCAType(
        TimeCAPoly=TIME_CA_POLY,
        R_CA_SCP=R_CA_SCP,
        FreqZero=FREQ_ZERO,
        DRateSFPoly=DRATE_SF_POLY,
        DopCentroidPoly=DOP_CENT_POLY,
        DopCentroidCOA=True,
    )


def test_sicd_elements_rmreftype(rma_sicd, rm_ref_type, kwargs):
    assert rm_ref_type.PosRef == rma_sicd.RMA.RMCR.PosRef
    assert rm_ref_type.VelRef == rma_sicd.RMA.RMCR.VelRef
    assert rm_ref_type.DopConeAngRef == rma_sicd.RMA.RMCR.DopConeAngRef
    assert not hasattr(rm_ref_type, "_xml_ns")
    assert not hasattr(rm_ref_type, "_xml_ns_key")

    # Init with kwargs
    rm_ref_type = RMA.RMRefType(
        PosRef=rma_sicd.RMA.RMCR.PosRef,
        VelRef=rma_sicd.RMA.RMCR.VelRef,
        DopConeAngRef=rma_sicd.RMA.RMCR.DopConeAngRef,
        **kwargs
    )
    assert rm_ref_type._xml_ns == kwargs["_xml_ns"]
    assert rm_ref_type._xml_ns_key == kwargs["_xml_ns_key"]


def test_sicd_elements_incatype(inca_type, kwargs):
    assert inca_type.TimeCAPoly == TIME_CA_POLY
    assert inca_type.R_CA_SCP == R_CA_SCP
    assert inca_type.FreqZero == FREQ_ZERO
    assert inca_type.DRateSFPoly == DRATE_SF_POLY
    assert inca_type.DopCentroidPoly == DOP_CENT_POLY
    assert inca_type.DopCentroidCOA == True

    inca_type = RMA.INCAType(
        TimeCAPoly=TIME_CA_POLY,
        R_CA_SCP=R_CA_SCP,
        FreqZero=FREQ_ZERO,
        DRateSFPoly=DRATE_SF_POLY,
        DopCentroidPoly=DOP_CENT_POLY,
        DopCentroidCOA=True,
        **kwargs
    )
    assert inca_type._xml_ns == kwargs["_xml_ns"]
    assert inca_type._xml_ns_key == kwargs["_xml_ns_key"]

    inca_type._apply_reference_frequency(500.0)
    assert inca_type.FreqZero == FREQ_ZERO + 500.0


def test_sicd_elements_rmatype(rma_sicd, inca_type, rm_ref_type, kwargs):
    # No image type
    rma_type = RMA.RMAType(RMAlgoType="OMEGA_K")
    assert rma_type.ImageType is None

    # Define RMAT image type
    rma_type = RMA.RMAType(RMAlgoType="OMEGA_K", RMAT=rm_ref_type, **kwargs)
    assert rma_type._xml_ns == kwargs["_xml_ns"]
    assert rma_type._xml_ns_key == kwargs["_xml_ns_key"]
    assert rma_type.ImageType == "RMAT"

    # Nothing to do test
    rma_type._derive_parameters(
        SCPCOA=None,
        Position=rma_sicd.Position,
        RadarCollection=rma_sicd.RadarCollection,
        ImageFormation=rma_sicd.ImageFormation,
    )

    # Derive parameters smoke test
    orig_doppler_cone_angle = rma_type.RMAT.DopConeAngRef
    rma_type._derive_parameters(
        SCPCOA=rma_sicd.SCPCOA,
        Position=rma_sicd.Position,
        RadarCollection=rma_sicd.RadarCollection,
        ImageFormation=rma_sicd.ImageFormation,
    )
    assert rma_type.RMAT.DopConeAngRef == orig_doppler_cone_angle

    # Force SCPCOA to be used for reference position and velocity
    empty_rf_ref_type = RMA.RMRefType(
        PosRef=None, VelRef=None, DopConeAngRef=rma_sicd.RMA.RMCR.DopConeAngRef
    )
    rma_type = RMA.RMAType(RMAlgoType="OMEGA_K", RMAT=empty_rf_ref_type)
    rma_type._derive_parameters(
        SCPCOA=rma_sicd.SCPCOA,
        Position=rma_sicd.Position,
        RadarCollection=rma_sicd.RadarCollection,
        ImageFormation=rma_sicd.ImageFormation,
    )
    assert rma_type.RMAT.PosRef.X == rma_sicd.SCPCOA.ARPPos.X
    assert rma_type.RMAT.PosRef.Y == rma_sicd.SCPCOA.ARPPos.Y
    assert rma_type.RMAT.PosRef.Z == rma_sicd.SCPCOA.ARPPos.Z
    assert rma_type.RMAT.VelRef.X == rma_sicd.SCPCOA.ARPVel.X
    assert rma_type.RMAT.VelRef.Y == rma_sicd.SCPCOA.ARPVel.Y
    assert rma_type.RMAT.VelRef.Z == rma_sicd.SCPCOA.ARPVel.Z

    # No reference doppler cone angle
    no_dop_cone_ang_ref_type = RMA.RMRefType(
        PosRef=rma_sicd.RMA.RMCR.PosRef,
        VelRef=rma_sicd.RMA.RMCR.VelRef,
        DopConeAngRef=None,
    )
    rma_type = RMA.RMAType(RMAlgoType="OMEGA_K", RMAT=no_dop_cone_ang_ref_type)
    rma_type._derive_parameters(
        SCPCOA=rma_sicd.SCPCOA,
        Position=rma_sicd.Position,
        RadarCollection=rma_sicd.RadarCollection,
        ImageFormation=rma_sicd.ImageFormation,
    )
    assert rma_type.RMAT.DopConeAngRef is not None

    # Test the INCA path
    inca_type = RMA.INCAType(
        TimeCAPoly=TIME_CA_POLY,
        R_CA_SCP=None,
        FreqZero=None,
        DRateSFPoly=DRATE_SF_POLY,
        DopCentroidPoly=DOP_CENT_POLY,
        DopCentroidCOA=True,
    )

    rma_type = RMA.RMAType(RMAlgoType="OMEGA_K", INCA=inca_type)
    rma_type._derive_parameters(
        SCPCOA=rma_sicd.SCPCOA,
        Position=rma_sicd.Position,
        RadarCollection=rma_sicd.RadarCollection,
        ImageFormation=rma_sicd.ImageFormation,
    )
    assert rma_type.INCA.R_CA_SCP is not None
    assert rma_type.INCA.FreqZero is not None

    rma_type._apply_reference_frequency(500.0)
    assert rma_type.INCA.FreqZero == rma_sicd.Antenna.Tx.FreqZero + 500.0
