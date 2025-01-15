import pathlib
import importlib.metadata

import pytest

import sarpy.io.general.base


@pytest.fixture
def load_plugin(monkeypatch):
    with monkeypatch.context() as mp:
        mock_dir = pathlib.Path(__file__).parents[2] / 'mock_site-packages'
        mp.syspath_prepend(mock_dir)
        assert importlib.metadata.entry_points().get('sarpy.io.complex', [])
        yield
    assert not importlib.metadata.entry_points().get('sarpy.io.complex', [])

@pytest.mark.parametrize(['sarpy_module', 'plugin_module'],
                         [('sarpy.io.complex', 'mock_plugin.mock_opener_complex'),
                          ('sarpy.io.product', 'mock_plugin.mock_opener_product'), ])
def test_plugin_openers(load_plugin, sarpy_module, plugin_module):
    openers = []
    sarpy.io.general.base.check_for_openers(sarpy_module, openers.append)
    loaded_modules = [opener.__module__ for opener in openers]
    assert plugin_module in loaded_modules
