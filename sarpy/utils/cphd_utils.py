"""
Script for extracting information from the CPHD header for review
"""

from __future__ import print_function
import argparse
import sys
import functools
from xml.dom import minidom
from typing import TextIO
import os

from sarpy.io.phase_history.cphd import CPHDDetails

if sys.version_info[0] < 3:
    import cStringIO as StringIO
else:
    from io import StringIO


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
    with open(input_file, 'rb') as fi:
        while True:
            lin = fi.readline().strip()
            if lin:
                print_func(lin.decode())
            else:
                break


def _create_default_output_file(input_file):
    return os.path.splitext(input_file)[0] + '.meta_dump.txt'


def _print_structure(input_file):
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
    input_file : str
    destination : TextIO
    """

    _define_print_function(destination)

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
    input_file : str
    destination : TextIO
    """

    _define_print_function(destination)
    _print_header(input_file)


def print_cphd_xml(input_file, destination=sys.stdout):
    """
    Prints the full CPHD header to the given destination.

    Parameters
    ----------
    input_file : str
    destination : TextIO
    """

    _define_print_function(destination)
    _print_structure(input_file)


def _dump_pattern(input_file, destination, call_method):
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
    input_file : str
        Path to a given input file.
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
    input_file : str
        Path to a given input file.
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
    input_file : str
        Path to a given input file.
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
