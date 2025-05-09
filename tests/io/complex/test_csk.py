import pathlib

import sarpy.io.complex.csk

def test_load_addin(monkeypatch):
    assert sarpy.io.complex.csk.load_addin() is None
    with monkeypatch.context() as mp:
        mock_dir = pathlib.Path(__file__).parents[2] / 'mock_site-packages'
        mp.syspath_prepend(mock_dir)
        assert sarpy.io.complex.csk.load_addin()
    assert sarpy.io.complex.csk.load_addin() is None
