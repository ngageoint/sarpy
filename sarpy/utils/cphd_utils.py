"""
Extract information from the CPHD header for review.

From the command-line

>>> python -m sarpy.utils.cphd_utils <path to cphd file>

For a basic help on the command-line, check

>>> python -m sarpy.utils.cphd_utils --help

"""

from __future__ import print_function
import argparse
import sys
import functools
from xml.dom import minidom
from typing import Union, TextIO, BinaryIO
import os
from io import StringIO

from sarpy.io.phase_history.cphd import CPHDDetails

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

# Custom print function
print_func = print


def _define_print_function(destination):
    """
    Define the print_func as necessary.

    Parameters
    ----------
    destination : TextIO
    """

    global print_func
    print_func = functools.partial(print, file=destination)


def _print_header(input_file):
    # type: (Union[str, BinaryIO]) -> None

    def _finalize():
        if close_after:
            file_object.close()
        if initial_location is not None:
            file_object.seek(initial_location)

    if isinstance(input_file, str):
        file_object = open(input_file, 'rb')
        initial_location = None
        close_after = True
    elif hasattr(input_file, 'readline'):
        file_object = input_file
        initial_location = file_object.tell()
        file_object.seek(0)
        close_after = False
    else:
        raise TypeError(
            'Input requires a file path or a binary mode file-like object')

    while True:
        lin = file_object.readline().strip()
        if not isinstance(lin, bytes):
            _finalize()
            raise ValueError('Requires an input opened in binary mode')

        if lin:
            print_func(lin.decode())
        else:
            break
    _finalize()


def _create_default_output_file(input_file):
    # type: (Union[str, BinaryIO]) -> str
    if isinstance(input_file, str):
        return os.path.splitext(input_file)[0] + '.meta_dump.txt'
    else:
        return os.path.expanduser('~/Desktop/phase_history.meta_dump.txt')


def _print_structure(input_file):
    # type: (Union[str, BinaryIO]) -> None
    details = CPHDDetails(input_file)
    data = details.get_cphd_bytes()
    xml_str = minidom.parseString(data.decode()).toprettyxml(indent='    ', newl='\n')
    # NB: this may or not exhibit platform dependent choices in which codec (i.e. latin-1 versus utf-8)
    for i, entry in enumerate(xml_str.splitlines()):
        if i == 0:
            # Remove xml that gets inserted by minidom, if it's not actually there
            if (not data.startswith(b'<?xml version')) and entry.startswith('<?xml version'):
                continue
            print_func(entry)
        elif entry.strip() != '':
            # Remove extra new lines if XML is already formatted
            print_func(entry)


def print_cphd_metadata(input_file, destination=sys.stdout):
    """
    Prints the full CPHD metadata (both header and CPHD structure) to the
    given destination.

    Parameters
    ----------
    input_file : str|BinaryIO
    destination : TextIO
    """

    _define_print_function(destination)

    if isinstance(input_file, str):
        print_func('Details for CPHD file {}'.format(input_file))

    print_func('---- CPHD Header Information ----')
    _print_header(input_file)
    print_func('')
    print_func('')

    print_func('---- CPHD Structure ----')
    _print_structure(input_file)
    print_func('')


def print_cphd_header(input_file, destination=sys.stdout):
    """
    Prints the full CPHD header to the given destination.

    Parameters
    ----------
    input_file : str|BinaryIO
    destination : TextIO
    """

    _define_print_function(destination)
    _print_header(input_file)


def print_cphd_xml(input_file, destination=sys.stdout):
    """
    Prints the full CPHD header to the given destination.

    Parameters
    ----------
    input_file : str|BinaryIO
    destination : TextIO
    """

    _define_print_function(destination)
    _print_structure(input_file)


def _dump_pattern(input_file, destination, call_method):
    # type: (Union[str, BinaryIO], str, Callable) -> Union[None, str]
    if destination == 'stdout':
        call_method(input_file, destination=sys.stdout)
    elif destination == 'string':
        out = StringIO()
        call_method(input_file, destination=out)
        value = out.getvalue()
        out.close()  # free the buffer
        return value
    else:
        the_out_file = _create_default_output_file(input_file) if destination == 'default' else destination
        with open(the_out_file, 'w') as fi:
            call_method(input_file, destination=fi)


def dump_cphd_metadata(input_file, destination):
    """
    Dump the CPHD metadata (both header and CPHD structure) to the given
    destination.

    Parameters
    ----------
    input_file : str|BinaryIO
        Path to or binary file-like object containing a CPHD file.
    destination : str
        'stdout', 'string', 'default' (will use `file_name+'.meta_dump.txt'`),
        or the path to an output file.

    Returns
    -------
    None|str
        There is only a return value if `destination=='string'`.
    """

    _dump_pattern(input_file, destination, print_cphd_metadata)


def dump_cphd_header(input_file, destination):
    """
    Dump the CPHD header to the given destination.

    Parameters
    ----------
    input_file : str|BinaryIO
        Path to or binary file-like object containing a CPHD file.
    destination : str
        'stdout', 'string', 'default' (will use `file_name+'.meta_dump.txt'`),
        or the path to an output file.

    Returns
    -------
    None|str
        There is only a return value if `destination=='string'`.
    """

    _dump_pattern(input_file, destination, print_cphd_header)


def dump_cphd_xml(input_file, destination):
    """
    Dump the CPHD structure to the given destination.

    Parameters
    ----------
    input_file : str|BinaryIO
        Path to or binary file-like object containing a CPHD file.
    destination : str
        'stdout', 'string', 'default' (will use `file_name+'.meta_dump.txt'`),
        or the path to an output file.

    Returns
    -------
    None|str
        There is only a return value if `destination=='string'`.
    """

    _dump_pattern(input_file, destination, print_cphd_xml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create extract metadata information from a CPHD file.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        'input_file', metavar='input_file', help='Path input CPHD file.')
    parser.add_argument(
        '-o', '--output', default='default',
        help="'default', 'stdout', or the path for an output file.\n"
             "* If not provided (`default`), the output will be at '<input path>.txt' \n"
             "* 'stdout' will print the information to standard out.\n"
             "* Otherwise, this is expected to be the path to a file \n"
             "       and output will be written there.\n"
             "  NOTE: existing output files will be overwritten.")
    parser.add_argument(
        '-d', '--data', default='both', choices=['both', 'header', 'xml'],
        help='Which information should be printed?')
    args = parser.parse_args()

    if args.data == 'both':
        dump_cphd_metadata(args.input_file, args.output)
    elif args.data == 'header':
        dump_cphd_header(args.input_file, args.output)
    elif args.data == 'xml':
        dump_cphd_xml(args.input_file, args.output)
    else:
        raise ValueError('Got unhandled data option {}'.format(args.data))
