#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np

from sarpy.io.complex.sicd_elements import ImageCreation


def test_imagecreationtype(kwargs):
    # Smoke test
    image_creation_type = ImageCreation.ImageCreationType(
        Application="Fake IFP",
        DateTime=np.datetime64("2023-06-23"),
        Site="Fake site",
        Profile="Fake profile",
    )
    assert image_creation_type.Application == "Fake IFP"
    assert image_creation_type.DateTime == np.datetime64("2023-06-23")
    assert image_creation_type.Site == "Fake site"
    assert image_creation_type.Profile == "Fake profile"
    assert not hasattr(image_creation_type, "_xml_ns")
    assert not hasattr(image_creation_type, "_xml_ns_key")

    # Init with kwargs
    image_creation_type = ImageCreation.ImageCreationType(
        Application="Fake IFP",
        DateTime=np.datetime64("today"),
        Site="Fake site",
        Profile="Fake profile",
        **kwargs
    )
    assert image_creation_type._xml_ns == kwargs["_xml_ns"]
    assert image_creation_type._xml_ns_key == kwargs["_xml_ns_key"]
