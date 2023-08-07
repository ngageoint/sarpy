#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import zipfile

import lxml.etree
import numpy as np

import sarpy.io.kml


def test_add_polygon_condition_coords(tmp_path):
    test_kmz = tmp_path / "test.kmz"
    with sarpy.io.kml.Document(str(test_kmz)) as kmz_doc:
        kmz_doc.add_default_style()
        input_coords_by_name = {}
        for val in (True, False):
            name = f"condition_coords={val}"
            coords = f"-179,0,0 179,0,0 -180,{(-1)**val},15000"
            kmz_doc.add_polygon(
                coords,
                styleUrl="#defaultStyle",
                altitudeMode="absolute",
                condition_coords=val,
                name=name,
            )
            input_coords_by_name[name] = coords

    with zipfile.ZipFile(test_kmz, "r") as kmz:
        assert set(kmz.namelist()) == {"doc.kml"}
        with kmz.open("doc.kml") as kml_fd:
            tree = lxml.etree.parse(kml_fd)
            assert tree.getroot().tag == "{http://www.opengis.net/kml/2.2}kml"

            actual_coords_by_name = {
                node.findtext("../../../../{*}name"): node.text
                for node in tree.findall(
                    ".//{*}Polygon/{*}outerBoundaryIs/{*}LinearRing/{*}coordinates"
                )
            }
            assert (
                actual_coords_by_name["condition_coords=False"]
                == input_coords_by_name["condition_coords=False"]
            )

            parsed_coords = np.array(
                [
                    [float(v) for v in coord.split(",")]
                    for coord in actual_coords_by_name["condition_coords=True"].split(
                        " "
                    )
                ]
            )
            assert np.array_equal(
                parsed_coords[:, 0], [-179, -181, -180, -179]
            )  # vertex 2 was unwrapped
            assert np.array_equal(parsed_coords[0], parsed_coords[-1])
