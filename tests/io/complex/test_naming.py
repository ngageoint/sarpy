import pathlib

import pytest

import sarpy.io.complex.naming.utils


@pytest.fixture
def load_plugin(monkeypatch):
    with monkeypatch.context() as mp:
        mock_dir = pathlib.Path(__file__).parents[2] / 'mock_site-packages'
        mp.syspath_prepend(mock_dir)
        sarpy.io.complex.naming.utils._name_functions = []
        sarpy.io.complex.naming.utils._parsed_name_functions = False
        yield
    sarpy.io.complex.naming.utils._name_functions = []
    sarpy.io.complex.naming.utils._parsed_name_functions = False


def test_plugin(load_plugin):
    import mock_plugin
    sarpy.io.complex.naming.utils.parse_name_functions()
    assert mock_plugin.plugin_commercial_id in sarpy.io.complex.naming.utils._name_functions
