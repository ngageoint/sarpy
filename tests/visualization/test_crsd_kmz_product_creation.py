import os
import pathlib
import xml.etree.ElementTree
import zipfile

import pytest

import sarpy.io.received.converter
import sarpy.io.received.crsd as sarpy_crsd
import sarpy.visualization.crsd_kmz_product_creation as crsd_kmz


@pytest.fixture
def crsd_reader():
    TEST_FILE_ROOT = os.environ.get("SARPY_TEST_PATH", None)
    if TEST_FILE_ROOT is not None:
        for item in (pathlib.Path(TEST_FILE_ROOT) / "crsd").rglob("*"):
            if item.is_file():
                reader = sarpy_crsd.is_a(str(item))
                if reader is not None:
                    return reader
    pytest.skip("crsd test file not found")


@pytest.fixture
def receive_only_crsd_reader(tmp_path, crsd_reader):
    modified_crsd_reader = sarpy.io.received.converter.open_received(
        crsd_reader.file_name
    )
    read_support = modified_crsd_reader.read_support_block()
    read_pvp = modified_crsd_reader.read_pvp_block()
    read_signal = modified_crsd_reader.read_signal_block()

    modified_crsd_reader.crsd_meta.CollectionID.CollectType = "RECEIVE_ONLY"
    modified_crsd_reader.crsd_meta.CollectionID.RadarMode = None
    modified_crsd_reader.crsd_meta.Global.FxBand = None
    modified_crsd_reader.crsd_meta.SceneCoordinates = None
    for chan in modified_crsd_reader.crsd_meta.Channel.Parameters:
        chan.SARImaging = None
    modified_crsd_reader.crsd_meta.PVP.TxPulse = None
    modified_crsd_reader.crsd_meta.Dwell = None

    names_to_remove = {
        "TxTime",
        "TxPos",
        "TxVel",
        "FX1",
        "FX2",
        "TXmt",
        "TxLFM",
        "TxACX",
        "TxACY",
        "TxEB",
    }
    new_pvps = {}
    for chan_id, pvps in read_pvp.items():
        new_pvps[chan_id] = pvps[list(set(pvps.dtype.names) - names_to_remove)].copy()

    # write the crsd file
    receive_only_crsd = tmp_path / "receive_only.crsd"
    with sarpy_crsd.CRSDWriter1(
        str(receive_only_crsd), modified_crsd_reader.crsd_meta
    ) as writer:
        writer.write_file(new_pvps, read_signal, read_support)

    return sarpy.io.received.converter.open_received(str(receive_only_crsd))


def _check_kmz(kmz_file):
    with zipfile.ZipFile(kmz_file, "r") as kmz:
        assert set(kmz.namelist()) == {"doc.kml"}
        with kmz.open("doc.kml") as kml_fd:
            tree = xml.etree.ElementTree.parse(kml_fd)
            assert tree.getroot().tag == "{http://www.opengis.net/kml/2.2}kml"


def test_create_kmz(crsd_reader, receive_only_crsd_reader, tmp_path):
    out_path = tmp_path / "crsd_kmz"
    out_path.mkdir()
    crsd_kmz.crsd_create_kmz_view(crsd_reader, out_path, file_stem="original")
    crsd_kmz.crsd_create_kmz_view(
        receive_only_crsd_reader, out_path, file_stem="receive_only"
    )

    assert len(list(out_path.glob("**/*"))) == 2
    original_kmz = next(out_path.glob("original*.kmz"))
    receive_only_kmz = next(out_path.glob("receive_only*.kmz"))

    _check_kmz(original_kmz)
    _check_kmz(receive_only_kmz)

    assert receive_only_kmz.stat().st_size <= original_kmz.stat().st_size
