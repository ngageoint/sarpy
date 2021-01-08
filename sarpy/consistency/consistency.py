# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import collections
import contextlib
import linecache
import sys
import textwrap

import numpy as np


def _exception_stack():
    try:
        exctype, value, tb = sys.exc_info()

        stack = []
        tback = tb
        while tback is not None:
            frame = tback.tb_frame
            filename = frame.f_code.co_filename
            linecache.checkcache(filename)
            line = linecache.getline(filename, tback.tb_lineno, frame.f_globals)

            stack.append({'filename': filename,
                          'lineno': tback.tb_lineno,
                          'line': line.strip()})
            tback = tback.tb_next

    finally:
        exctype = value = tb = None

    return stack


class ConsistencyChecker(object):
    """Base class for implementing consistency checkers.

    This class can be used to perform and log comparisons. Each comparison can be logged as
    either an ``'Error'`` or a ``'Warning'``.
    """
    def __init__(self):
        self._all_check_results = collections.OrderedDict()
        self._active_check = None

        names = [name for name in dir(self) if name.startswith('check_')]
        attrs = [getattr(self, name) for name in sorted(names)]
        self.funcs = [attr for attr in attrs if hasattr(attr, '__call__')]

    def check(self, func_name=None):
        # run specified test(s) or all of them
        if func_name is None:
            funcs = self.funcs
        else:
            if isinstance(func_name, str):
                func_name = [func_name]

            not_found = set(func_name) - set([func.__name__ for func in self.funcs])
            if not_found:
                raise ValueError("Functions not found: {}".format(not_found))

            funcs = [func for func in self.funcs if func.__name__ in func_name]

        for func in funcs:
            self._run_check(func)

    def _run_check(self, func):
        """Runs a single 'check_' method."""

        self._active_check = {'doc': func.__doc__,
                              'details': [],
                              'passed': True}

        # func() will populate self._active_check
        try:
            func()
        except Exception as e:
            stack = _exception_stack()
            message = []
            for indent, frame in enumerate(stack[1:]):
                message.append(' '*indent*4 + "line#{lineno}: {line}".format(lineno=frame['lineno'], line=frame['line']))
            message.append(str(e))
            self._add_item_to_current('Error', False, '\n'.join(message), details="Exception Raised")

        self._all_check_results[func.__name__] = self._active_check
        self._active_check = None

    def _add_item_to_current(self, severity, passed, message, details=''):
        """Records the result of a test."""

        item = {'severity': severity,
                'passed': passed,
                'message': message,
                'details': str(details)}

        self._active_check['details'].append(item)
        self._active_check['passed'] &= passed

    def _format_assertion(self, e, depth=1):
        stack = _exception_stack()
        frame = stack[depth]
        return ("line#{lineno}: {line}".format(lineno=frame['lineno'], line=frame['line'])
                + '\n' + '\n'.join(str(x) for x in e.args))

    @contextlib.contextmanager
    def need(self, details=None):
        with self._crave('Error', details=details):
            yield

    @contextlib.contextmanager
    def want(self, details=None):
        with self._crave('Warning', details=details):
            yield

    @contextlib.contextmanager
    def _crave(self, level, details, depth=2):
        try:
            yield
            if self._active_check is not None:
                self._add_item_to_current(level, True, '', details=details)
        except AssertionError as e:
            if self._active_check is None:
                raise
            if not details:
                stack = _exception_stack()
                details = stack[depth]['line']
            self._add_item_to_current(level, False, self._format_assertion(e, depth=depth), details=details)

    @contextlib.contextmanager
    def precondition(self, details=None):
        try:
            yield
        except AssertionError as e:
            if self._active_check is None:
                return
            if not details:
                stack = _exception_stack()
                details = stack[1]['line']
            self._add_item_to_current('No-Op', True, self._format_assertion(e), details=details)

    def all(self):
        """Returns all results."""
        return self._all_check_results

    def failures(self, omit_passed_sub=False):
        """Returns failure results."""
        retval = collections.OrderedDict()
        for k, v in self._all_check_results.items():
            if not v['passed']:
                retval[k] = dict(v)
                if omit_passed_sub:
                    retval[k]['details'] = [d for d in v['details'] if not d['passed']]
        return retval

    def print_result(self, include_passed_asserts=True, color=True, include_passed_checks=False, width=120,
                     skip_detail=False, fail_detail=False, pass_detail=False):
        to_print = collections.OrderedDict()
        for k, v in self._all_check_results.items():
            if include_passed_checks or not v['passed']:
                to_print[k] = dict(v)
                to_print[k]['details'] = [d for d in v['details'] if include_passed_asserts or not d['passed']]

        if color:
            coloration = {('Error', True): ['[Pass]', 'green', 'bold'],
                          ('Error', False): ['[Error]', 'red', 'bold'],
                          ('Warning', True): ['[Pass]', 'cyan'],
                          ('Warning', False): ['[Warning]', 'yellow'],
                          ('No-Op', True): ['[Skip]', 'blue']}
        else:
            coloration = {('Error', True): ['[Need]'],
                          ('Error', False): ['[Error]'],
                          ('Warning', True): ['[Want]'],
                          ('Warning', False): ['[Warning]'],
                          ('No-Op', True): ['[Skip]']}

        indent = 4
        for case, details in to_print.items():
            print("{}: {}".format(case, details['doc']))
            if details['details']:
                for sub in details['details']:
                    lead = in_color(*coloration[sub['severity'], sub['passed']])
                    need_want = {'Error': 'Need', 'Warning': 'Want', 'No-Op': 'Unless'}[sub['severity']]
                    print("{indent}{lead} {need_want}: {details}".format(indent=' '*indent,
                                                                         lead=lead,
                                                                         need_want=need_want,
                                                                         details=sub['details']))
                    if (skip_detail and sub['severity'] == 'No-Op'
                            or (fail_detail and not sub['passed'])
                            or (pass_detail and sub['passed'])):
                        for line in sub['message'].splitlines():
                            message = '\n'.join(textwrap.wrap(line, width=width,
                                                              subsequent_indent=' '*(indent + 8),
                                                              initial_indent=' '*(indent+4)))
                            print(message)
            else:
                print("{}---: No test performed".format(' '*indent))


class Approx:
    # Tell numpy to use our comparison operators
    __array_ufunc__ = None
    __array_priority__ = 100

    def __init__(self, value, atol=1e-10, rtol=1e-6):
        self.value = value
        self.atol = atol
        self.rtol = rtol

    def __lt__(self, rhs):
        return self.__le__(rhs)

    def __le__(self, rhs):
        return np.all(np.logical_or(np.less(self.value, rhs),
                                    np.isclose(self.value, rhs, rtol=self.rtol, atol=self.atol)))

    def __eq__(self, rhs):
        return np.all(np.isclose(self.value, rhs, rtol=self.rtol, atol=self.atol))

    def __ne__(self, rhs):
        return not self.__eq__(rhs)

    def __ge__(self, rhs):
        return np.all(np.logical_or(np.greater(self.value, rhs),
                                    np.isclose(self.value, rhs, rtol=self.rtol, atol=self.atol)))

    def __gt__(self, rhs):
        return self.__ge__(rhs)

    def __repr__(self):
        tol = np.maximum(self.atol, np.asarray(self.value) * self.rtol)
        return "{} +/- {}".format(self.value, tol)


def in_color(string, *color):
    if color:
        start = ''.join(start_color(c) for c in color)
        return "{}{}{}".format(start, string, END_COLOR)
    else:
        return string


END_COLOR = "\x1b[0m"


def start_color(color):
    color_table = dict(
        black=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        purple=35,
        cyan=36,
        white=37,
        bold=1,
        light=2,
        invert=7,
    )
    return "\x1b[%sm" % color_table[color]
