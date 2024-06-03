import copy
import pathlib

import numpy as np
import numpy.testing
import pytest

import sarpy.consistency.cphd_consistency
import sarpy.io.phase_history.converter
from sarpy.io.phase_history.cphd import CPHDReader, CPHDReader0_3, CPHDWriter1

import tests


CPHD_FILE_TYPES = tests.find_test_data_files(pathlib.Path(__file__).parent / 'cphd_file_types.json')
for path in CPHD_FILE_TYPES.get('CPHD', []):
    if pathlib.Path(path).name == 'dynamic_stripmap_ci2.cphd':
        CI2_CPHD = path
        break
else:
    CI2_CPHD = None


@pytest.mark.parametrize('cphd_path', CPHD_FILE_TYPES.get('CPHD',[]))
def test_cphd_read(cphd_path):
    reader = sarpy.io.phase_history.converter.open_phase_history(cphd_path)
    assert isinstance(reader, CPHDReader)
    assert reader.reader_type == 'CPHD'
    _assert_read_sizes(reader)


def _assert_read_sizes(reader):
    data_sizes = reader.get_data_size_as_tuple()
    elements = reader.cphd_meta.Data.ArraySize if isinstance(reader, CPHDReader0_3) else reader.cphd_meta.Data.Channels
    for i, (data_size, element) in enumerate(zip(data_sizes, elements)):
        assert data_size == (element.NumVectors, element.NumSamples)

        assert reader[:2, :2, i].shape[:2] == (2, 2)
        assert reader[-2:, :2, i].shape[:2] == (2, 2)
        assert reader[-2:, -2:, i].shape[:2] == (2, 2)
        assert reader[:2, -2:, i].shape[:2] == (2, 2)
        assert reader[:, :2, i].shape[:2] == (data_size[0], 2)
        assert reader[:2, :, i].shape[:2] == (2, data_size[1])

        assert reader.read_pvp_variable('TxTime', i, the_range=None).shape == (data_size[0], )
        assert reader.read_pvp_variable('TxTime', i, the_range=(0, 10, 2)).shape == (5, )


@pytest.mark.parametrize('cphd_path', CPHD_FILE_TYPES.get('CPHD',[]))
def test_cphd_read_write(cphd_path, tmp_path):
    reader = sarpy.io.phase_history.converter.open_phase_history(cphd_path)
    if isinstance(reader, CPHDReader0_3):
        pytest.skip(reason='Writing CPHDs earlier than 1.0 is not supported')

    written_cphd_name = tmp_path / 'example_cphd.cphd'

    read_support = reader.read_support_block()
    read_pvp = reader.read_pvp_block()
    read_signal = reader.read_signal_block()

    # write the cphd file
    with CPHDWriter1(str(written_cphd_name), reader.cphd_meta, check_existence=False) as writer:
        writer.write_file(read_pvp, read_signal, read_support)

    # reread the newly written data
    rereader = CPHDReader(str(written_cphd_name))
    reread_support = rereader.read_support_block()
    reread_pvp = rereader.read_pvp_block()
    reread_signal = rereader.read_signal_block()

    # compare the original data and re-read data
    assert read_support.keys() == reread_support.keys(), 'Support keys are not identical'
    for support_key in reread_support:
        numpy.testing.assert_array_equal(read_support[support_key], reread_support[support_key])

    assert read_pvp.keys() == reread_pvp.keys(), 'PVP keys are not identical'
    for pvp_key in reread_pvp:
        numpy.testing.assert_array_equal(read_pvp[pvp_key], reread_pvp[pvp_key])

    assert read_signal.keys() == reread_signal.keys(), 'Signal keys are not identical'
    for signal_key in reread_signal:
        numpy.testing.assert_allclose(read_signal[signal_key], reread_signal[signal_key], rtol=1e-6)

    assert not sarpy.consistency.cphd_consistency.main([str(written_cphd_name), '--signal-data'])


@pytest.mark.skipif(CI2_CPHD is None, reason="dynamic_stripmap_ci2.cphd not found")
def test_cphd_read_write_cf8_ampsf(tmp_path):
    reader = sarpy.io.phase_history.converter.open_phase_history(CI2_CPHD)
    assert reader.cphd_meta.Data.SignalArrayFormat == 'CI2'
    assert reader.cphd_meta.PVP.AmpSF is not None

    written_cphd_name = tmp_path / 'example_cphd.cphd'

    read_support = reader.read_support_block()
    read_pvp = reader.read_pvp_block()
    read_signal = reader.read_signal_block()
    read_signal_raw = reader.read_signal_block_raw()

    modified_meta = copy.deepcopy(reader.cphd_meta)
    modified_meta.Data.SignalArrayFormat = 'CF8'
    modified_signal = {k: v.astype(np.float32) for k, v in read_signal_raw.items()}

    # write the cphd file
    with CPHDWriter1(str(written_cphd_name), modified_meta, check_existence=False) as writer:
        writer.write_file_raw(read_pvp, modified_signal, read_support)

    # reread the newly written data
    rereader = CPHDReader(str(written_cphd_name))
    reread_support = rereader.read_support_block()
    reread_pvp = rereader.read_pvp_block()
    reread_signal = rereader.read_signal_block()

    assert rereader.cphd_header.SIGNAL_BLOCK_SIZE == 4 * reader.cphd_header.SIGNAL_BLOCK_SIZE

    # compare the original data and re-read data
    assert read_support.keys() == reread_support.keys(), 'Support keys are not identical'
    for support_key in reread_support:
        numpy.testing.assert_array_equal(read_support[support_key], reread_support[support_key])

    assert read_pvp.keys() == reread_pvp.keys(), 'PVP keys are not identical'
    for pvp_key in reread_pvp:
        numpy.testing.assert_array_equal(read_pvp[pvp_key], reread_pvp[pvp_key])

    assert read_signal.keys() == reread_signal.keys(), 'Signal keys are not identical'
    for signal_key in reread_signal:
        numpy.testing.assert_array_equal(read_signal[signal_key], reread_signal[signal_key])

    assert not sarpy.consistency.cphd_consistency.main([str(written_cphd_name), '--signal-data'])


@pytest.mark.parametrize('cphd_path', CPHD_FILE_TYPES.get('CPHD',[]))
def test_cphd_read_write_compressed(cphd_path, tmp_path):
    reader = sarpy.io.phase_history.converter.open_phase_history(cphd_path)
    if isinstance(reader, CPHDReader0_3):
        pytest.skip(reason='Writing CPHDs earlier than 1.0 is not supported')

    read_support = reader.read_support_block()
    read_pvp = reader.read_pvp_block()
    read_signal = reader.read_signal_block()

    # Modify CPHD to use compressed signal blocks
    write_meta = reader.cphd_meta.copy()
    write_meta.Data.SignalCompressionID = "FauxCompression"
    write_signal = {}
    for entry in write_meta.Data.Channels:
        write_signal[entry.Identifier] = np.frombuffer(read_signal[entry.Identifier].tobytes(), dtype=np.uint8)
        entry.CompressedSignalSize = len(write_signal[entry.Identifier])

    written_cphd_name = tmp_path / 'compressed.cphd'
    with CPHDWriter1(str(written_cphd_name), write_meta, check_existence=False) as writer:
        writer.write_file(read_pvp, write_signal, read_support)

    # reread the newly written data
    rereader = CPHDReader(str(written_cphd_name))
    reread_support = rereader.read_support_block()
    reread_pvp = rereader.read_pvp_block()
    reread_signal = rereader.read_signal_block()

    # compare the original data and re-read data
    assert read_support.keys() == reread_support.keys(), 'Support keys are not identical'
    for support_key in reread_support:
        numpy.testing.assert_array_equal(read_support[support_key], reread_support[support_key])

    assert read_pvp.keys() == reread_pvp.keys(), 'PVP keys are not identical'
    for pvp_key in reread_pvp:
        numpy.testing.assert_array_equal(read_pvp[pvp_key], reread_pvp[pvp_key])

    assert write_signal.keys() == reread_signal.keys(), 'Signal keys are not identical'
    for signal_key in reread_signal:
        numpy.testing.assert_array_equal(write_signal[signal_key], reread_signal[signal_key])

    assert not sarpy.consistency.cphd_consistency.main([str(written_cphd_name), '--signal-data'])
