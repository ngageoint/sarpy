# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import itertools

import pytest

import sarpy.consistency.consistency as con


class DummyConsistency(con.ConsistencyChecker):
    """A ConsistencyChecker used for unit testing and code coverage"""
    def __init__(self):
        super(DummyConsistency, self).__init__()

    def check_need_pass(self):
        with self.need('need pass'):
            assert True

    def check_need_fail(self):
        with self.need('need fail'):
            assert False

    def check_need_both(self):
        with self.need('need pass'):
            assert True
        with self.need('need fail'):
            assert False

    def check_need_fail_nodetails(self):
        with self.need():
            assert False

    def check_pre_need_pass(self):
        with self.precondition():
            assert True
            with self.need('need pass'):
                assert True

    def check_nopre_need_pass(self):
        with self.precondition():
            assert False
            with self.need('need pass'):
                assert True

    def check_want_pass(self):
        with self.want('want pass'):
            assert True

    def check_want_fail(self):
        with self.want('want fail'):
            assert False

    def check_pre_want_pass(self):
        with self.precondition():
            assert True
            with self.want('want pass'):
                assert True

    def check_nopre_want_pass(self):
        with self.precondition():
            assert False
            with self.want('want pass'):
                assert True

    def check_exception(self):
        raise ValueError


@pytest.fixture
def dummycon():
    """Fixture which initializes a DummyConsistency object

    Yields
    ------
    DummyConsistency object
    """
    import ast
    import os
    import _pytest.assertion.rewrite
    base, _ = os.path.splitext(__file__)  # python2 can return the '*.pyc' file
    with open(base + '.py', 'r') as fd:
        source = fd.read()
    tree = ast.parse(source)
    try:
        _pytest.assertion.rewrite.rewrite_asserts(tree)
    except TypeError as e:
        _pytest.assertion.rewrite.rewrite_asserts(tree, source)

    co = compile(tree, __file__, 'exec', dont_inherit=True)
    ns = {}
    exec(co, ns)
    cover_con = ns['DummyConsistency']()
    yield cover_con


def test_all(dummycon, capsys):
    dummycon.check()
    assert len(dummycon.all()) == 11
    assert len(dummycon.failures()) == 5

    failures = dummycon.failures()
    details = itertools.chain.from_iterable([value['details'] for value in failures.values()])
    passed = [item for item in details if item['passed']]
    assert passed

    failures = dummycon.failures(omit_passed_sub=True)
    details = itertools.chain.from_iterable([value['details'] for value in failures.values()])
    passed = [item for item in details if item['passed']]
    assert not passed

    dummycon.print_result()
    captured = capsys.readouterr()
    assert '\x1b' in captured.out
    dummycon.print_result(color=False)
    captured2 = capsys.readouterr()
    assert '\x1b' not in captured2.out

    dummycon.print_result(include_passed_checks=True, skip_detail=True, fail_detail=True, pass_detail=True)
    captured3 = capsys.readouterr()
    assert 'Skip' in captured3.out
    assert 'check_nopre_want_pass' in captured3.out
    assert 'check_want_pass' in captured3.out


def test_one(dummycon):
    dummycon.check('check_need_pass')


def test_multiple(dummycon):
    dummycon.check(['check_need_pass', 'check_need_fail'])


def test_invalid(dummycon):
    with pytest.raises(ValueError):
        dummycon.check('this_does_not_exist')


def test_approx():
    apx = con.Approx(10.0, atol=.1)
    assert apx == 10.0
    assert apx == 10.01
    assert not apx != 10.01
    assert apx > 10.01
    assert apx >= 10.01
    assert apx >= 0
    assert not apx <= 0
    assert apx < 10.01
    assert apx <= 10.01
    assert repr(apx) == "10.0 +/- 0.1"
