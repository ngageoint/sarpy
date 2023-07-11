#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.complex.sicd_elements import CollectionInfo


def test_collectioninfo_radarmodetype(kwargs):
    radar_mode_type = CollectionInfo.RadarModeType(ModeType="SPOTLIGHT", ModeID="SL")
    assert radar_mode_type.ModeType == "SPOTLIGHT"
    assert radar_mode_type.ModeID == "SL"
    assert not hasattr(radar_mode_type, "_xml_ns")
    assert not hasattr(radar_mode_type, "_xml_ns_key")

    # Init with kwargs
    radar_mode_type = CollectionInfo.RadarModeType(
        ModeType="SPOTLIGHT", ModeID="SL", **kwargs
    )
    assert radar_mode_type._xml_ns == kwargs["_xml_ns"]
    assert radar_mode_type._xml_ns_key == kwargs["_xml_ns_key"]

    radar_mode_type = CollectionInfo.RadarModeType(ModeType="SPOTLIGHT")
    assert radar_mode_type.get_mode_abbreviation() == "SL"
    radar_mode_type = CollectionInfo.RadarModeType(ModeType="STRIPMAP")
    assert radar_mode_type.get_mode_abbreviation() == "ST"
    radar_mode_type = CollectionInfo.RadarModeType(ModeType="DYNAMIC STRIPMAP")
    assert radar_mode_type.get_mode_abbreviation() == "DS"


def test_collectioninfo_collinfotype(sicd, kwargs):
    collection_info_type = CollectionInfo.CollectionInfoType(
        CollectorName=sicd.CollectionInfo.CollectorName,
        IlluminatorName="FAKE_ILLUMINATOR",
        CoreName=sicd.CollectionInfo.CoreName,
        CollectType=sicd.CollectionInfo.CollectType,
        RadarMode=sicd.CollectionInfo.RadarMode,
        Classification=sicd.CollectionInfo.Classification,
        CountryCodes=["FAKE_CC"],
        Parameters={"FAKE": "PARAMETERS"},
    )
    assert collection_info_type.CollectorName == sicd.CollectionInfo.CollectorName
    assert collection_info_type.IlluminatorName == "FAKE_ILLUMINATOR"
    assert collection_info_type.CoreName == sicd.CollectionInfo.CoreName
    assert collection_info_type.CollectType == sicd.CollectionInfo.CollectType
    assert collection_info_type.RadarMode == sicd.CollectionInfo.RadarMode
    assert collection_info_type.Classification == sicd.CollectionInfo.Classification
    assert collection_info_type.CountryCodes == ["FAKE_CC"]
    assert collection_info_type.Parameters["FAKE"] == "PARAMETERS"
    assert not hasattr(collection_info_type, "_xml_ns")
    assert not hasattr(collection_info_type, "_xml_ns_key")

    # Init with kwargs
    collection_info_type = CollectionInfo.CollectionInfoType(
        CollectorName=sicd.CollectionInfo.CollectorName,
        CoreName=sicd.CollectionInfo.CoreName,
        RadarMode=sicd.CollectionInfo.RadarMode,
        Classification=sicd.CollectionInfo.Classification,
        **kwargs
    )
    assert collection_info_type._xml_ns == kwargs["_xml_ns"]
    assert collection_info_type._xml_ns_key == kwargs["_xml_ns_key"]
