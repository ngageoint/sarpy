#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#

import os

import nox

_PYTHON_VERSIONS = ['3.6', '3.11']
_LOCATIONS = ["tests"]


# Run only test session when no arguments are specified
nox.options.sessions = ["test"]


@nox.session(venv_backend="conda")
@nox.parametrize('version', _PYTHON_VERSIONS)
def test(session, version):
    assert 'SARPY_TEST_PATH' in os.environ
    args = session.posargs or _LOCATIONS
    session.conda_install(f'python={version}')
    session.install('.[all]')
    session.run("pytest", *args)
