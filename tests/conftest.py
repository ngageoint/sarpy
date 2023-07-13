import pathlib

import pytest


@pytest.fixture
def tests_path():
    return pathlib.Path(__file__).parent
