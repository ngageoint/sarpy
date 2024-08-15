#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import pathlib
import tempfile

import numpy as np
import pytest

import sarpy.io.product
import sarpy.utils.create_product

import tests


TOLERANCE = 1e-8

product_file_types = tests.find_test_data_files(pathlib.Path(__file__).parent / 'product_file_types.json')
sicd_files = product_file_types.get('SICD', [])


@pytest.fixture(scope="module", params=[1, 2, 3])
def sidd_nitf(request):
    if not sicd_files:
        pytest.skip("SICD file required; check SARPY_TEST_PATH")
    with tempfile.TemporaryDirectory() as tmpdir:
        sarpy.utils.create_product.main([
            str(sicd_files[0]),
            tmpdir,
            f'--version={request.param}',
        ])
        contents = list(pathlib.Path(tmpdir).iterdir())
        assert len(contents) == 1
        yield contents[0]


def test_sidd_projection(sidd_nitf):
    sidd_reader = sarpy.io.product.open(str(sidd_nitf))
    for sidd_obj in sidd_reader.sidd_meta:
        assert sidd_obj.Measurement.ProjectionType != "PolynomialProjection"  # would not support projections
        ref_pt_rowcol = sidd_obj.Measurement.ReferencePoint.Point.get_array()
        ref_pt_ecef = sidd_obj.Measurement.ReferencePoint.ECEF.get_array()
        ref_pt_ecef_proj = sidd_obj.project_image_to_ground(ref_pt_rowcol)
        ref_pt_rowcol_proj, _, _ = sidd_obj.project_ground_to_image(ref_pt_ecef)
        assert np.linalg.norm(ref_pt_rowcol - ref_pt_rowcol_proj) == pytest.approx(0, abs=1e-2)
        assert np.linalg.norm(ref_pt_ecef - ref_pt_ecef_proj) == pytest.approx(0, abs=1e-3)
