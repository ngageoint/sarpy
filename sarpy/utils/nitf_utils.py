"""
A utility for dumping a NITF header to the console. Contributed by Austin Lan of L3/Harris.

To dump NITF header information to a text file from the command-line

>>> python -m sarpy.utils.nitf_utils <path to nitf file>

For a basic help on the command-line, check

>>> python -m sarpy.utils.nitf_utils --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = "Austin Lan, L3/Harris"


import argparse
import functools
import sys
from xml.dom import minidom
import os
from typing import Union, BinaryIO, TextIO, List, Dict
from io import StringIO

from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.base import NITFElement, TRE, TREList, UserHeaderType
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, DataExtensionHeader0, \
    DESUserHeader
from sarpy.io.general.nitf_elements.graphics import GraphicsSegmentHeader
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageSegmentHeader0, MaskSubheader
from sarpy.io.general.nitf_elements.label import LabelSegmentHeader
from sarpy.io.general.nitf_elements.nitf_head import NITFHeader, NITFHeader0
from sarpy.io.general.nitf_elements.res import ReservedExtensionHeader, ReservedExtensionHeader0, \
    RESUserHeader
from sarpy.io.general.nitf_elements.symbol import SymbolSegmentHeader
from sarpy.io.general.nitf_elements.text import TextSegmentHeader, TextSegmentHeader0
from sarpy.io.general.nitf_elements.tres.tre_elements import TREElement


# Custom print function
print_func = print


############
# helper methods

def _filter_files(input_path):
    """
    Determine if a given input path corresponds to a NITF 2.1 or 2.0 file.

    Parameters
    ----------
    input_path : str

    Returns
    -------
    bool
    """

    if not os.path.isfile(input_path):
        return False
    _, fext = os.path.splitext(input_path)
    with open(input_path, 'rb') as fi:
        check = fi.read(9)
    return check in [b'NITF02.10', b'NITF02.00']


def _create_default_output_file(input_file, output_directory=None):
    if not isinstance(input_file, str):
        if output_directory is None:
            return os.path.expanduser('~/Desktop/header_dump.txt')
        else:
            return os.path.join(output_directory, 'header_dump.txt')

    if output_directory is None:
        return os.path.splitext(input_file)[0] + '.header_dump.txt'
    else:
        return os.path.join(output_directory, os.path.splitext(os.path.split(input_file)[1])[0] + '.header_dump.txt')


def _decode_effort(value):
    # type: (bytes) -> Union[bytes, str]

    # noinspection PyBroadException
    try:
        return value.decode()
    except Exception:
        return value


############
# printing methods

def _print_element_field(elem, field, prefix=''):
    # type: (Union[None, NITFElement], Union[None, str], str) -> None
    if elem is None or field is None:
        return

    value = getattr(elem, field, None)
    if value is None:
        value = ''
    print_func('{}{} = {}'.format(prefix, field, value))


def _print_element(elem, prefix=''):
    # type: (Union[None, NITFElement], str) -> None
    if elem is None:
        return

    # noinspection PyProtectedMember
    for field in elem._ordering:
        _print_element_field(elem, field, prefix=prefix)


def _print_element_list(elem_list, prefix=''):
    # type: (Union[None, List[NITFElement]], str) -> None
    if elem_list is None:
        return

    for i, elem in enumerate(elem_list):
        _print_element(elem, prefix='{}[{}].'.format(prefix, i))


def _print_tre_element(field, value, prefix=''):
    # type: (Union[None, str], Union[str, int, bytes], str) -> None
    if field is None:
        return

    if value is None:
        value = ''
    print_func('{}{} = {}'.format(prefix, field, value))


def _print_tre_list(elem_list, prefix=''):
    # type: (Union[None, List, TREList], str) -> None

    if elem_list is None:
        return

    for i, elem in enumerate(elem_list):
        _print_tre_dict(elem, '{}[{}].'.format(prefix, i))


def _print_tre_dict(elem_dict, prefix=''):
    # type: (Union[None, Dict], str) -> None
    if elem_dict is None:
        return

    for field, value in elem_dict.items():
        if isinstance(value, list):
            _print_tre_list(value, '{}{}'.format(prefix, field))
        else:
            _print_tre_element(field, value, prefix)


def _print_tres(tres):
    # type: (Union[TREList, List[TRE]]) -> None
    for tre in tres:
        print_func('')
        if isinstance(tre.DATA, TREElement):
            _print_tre_dict(tre.DATA.to_dict(), prefix='{}.'.format(tre.TAG))
        else:
            # Unknown TRE
            _print_tre_element('DATA', _decode_effort(tre.DATA), prefix='{}.'.format(tre.TAG))


def _print_file_header(hdr):
    # type: (Union[NITFHeader, NITFHeader0]) -> None

    # noinspection PyProtectedMember
    for field in hdr._ordering:
        if field == 'Security':
            _print_element(getattr(hdr, field, None), prefix='FS')
        elif field == 'FBKGC':
            value = getattr(hdr, field, None)
            print_func('FBKGC = {} {} {}'.format(value[0], value[1], value[2]))
        elif field in [
            'ImageSegments', 'GraphicsSegments', 'SymbolSegments', 'LabelSegments',
            'TextSegments', 'DataExtensions', 'ReservedExtensions']:
            pass
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                _print_tres(value.data.tres)
        else:
            _print_element_field(hdr, field)


def _print_mask_header(hdr):
    # type: (Union[None, MaskSubheader]) -> None
    if hdr is None:
        return

    print_func('----- Mask Subheader (part of image data segment) -----')
    # noinspection PyProtectedMember
    for field in hdr._ordering:
        if field in ['BMR', 'TMR']:
            value = getattr(hdr, field, None)
            if value is None:
                continue
            else:
                for the_band, subarray in enumerate(value):
                    print_func('{}BND{} = {}'.format(field, the_band, subarray))
        else:
            _print_element_field(hdr, field, prefix='')


def _print_image_header(hdr):
    # type: (Union[ImageSegmentHeader, ImageSegmentHeader0]) -> None

    # noinspection PyProtectedMember
    for field in hdr._ordering:
        if field == 'Security':
            _print_element(getattr(hdr, field, None), prefix='IS')
        elif field in ['Comments', 'Bands']:
            _print_element_list(getattr(hdr, field, None), prefix='{}'.format(field))
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                _print_tres(value.data.tres)
        else:
            _print_element_field(hdr, field)
    _print_mask_header(hdr.mask_subheader)


def _print_basic_header(hdr, prefix):
    # noinspection PyProtectedMember
    for field in hdr._ordering:
        if field == 'Security':
            _print_element(getattr(hdr, field, None), prefix=prefix)
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                _print_tres(value.data.tres)
        else:
            _print_element_field(hdr, field)


def _print_graphics_header(hdr):
    # type: (GraphicsSegmentHeader) -> None
    _print_basic_header(hdr, 'SS')


def _print_symbol_header(hdr):
    # type: (SymbolSegmentHeader) -> None
    _print_basic_header(hdr, 'SS')


def _print_label_header(hdr):
    # type: (LabelSegmentHeader) -> None
    _print_basic_header(hdr, 'LS')


def _print_text_header(hdr):
    # type: (Union[TextSegmentHeader, TextSegmentHeader0]) -> None
    _print_basic_header(hdr, 'TS')


def _print_extension_header(hdr, prefix):
    # noinspection PyProtectedMember
    for field in hdr._ordering:
        if field == 'Security':
            _print_element(getattr(hdr, field, None), prefix=prefix)
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            if isinstance(value, (DESUserHeader, RESUserHeader)):
                if value.data:
                    # Unknown user-defined subheader
                    print_func('{}SHF = {}'.format(prefix, _decode_effort(value.data)))
            else:
                # e.g., XMLDESSubheader
                _print_element(value, prefix='{}SHF.'.format(prefix))
        else:
            _print_element_field(hdr, field)


def _print_des_header(hdr):
    # type: (Union[DataExtensionHeader, DataExtensionHeader0]) -> None
    _print_extension_header(hdr, 'DES')


def _print_res_header(hdr):
    # type: (Union[ReservedExtensionHeader, ReservedExtensionHeader0]) -> None
    _print_extension_header(hdr, 'RES')


def print_nitf(file_name, dest=sys.stdout):
    """
    Worker function to dump the NITF header and various subheader details to the
    provided destination.

    Parameters
    ----------
    file_name : str|BinaryIO
    dest : TextIO
    """

    # Configure print function for desired destination
    #    - e.g., stdout, string buffer, file
    global print_func
    print_func = functools.partial(print, file=dest)

    details = NITFDetails(file_name)

    if isinstance(file_name, str):
        print_func('')
        print_func('Details for file {}'.format(file_name))
        print_func('')

    print_func('----- File Header -----')
    _print_file_header(details.nitf_header)
    print_func('')

    if details.img_subheader_offsets is not None:
        for img_subhead_num in range(details.img_subheader_offsets.size):
            print_func('----- Image {} -----'.format(img_subhead_num))
            hdr = details.parse_image_subheader(img_subhead_num)
            _print_image_header(hdr)
            print_func('')

    if details.graphics_subheader_offsets is not None:
        for graphics_subhead_num in range(details.graphics_subheader_offsets.size):
            print_func('----- Graphic {} -----'.format(graphics_subhead_num))
            hdr = details.parse_graphics_subheader(graphics_subhead_num)
            _print_graphics_header(hdr)
            data = details.get_graphics_bytes(graphics_subhead_num)
            print_func('GSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.symbol_subheader_offsets is not None:
        for symbol_subhead_num in range(details.symbol_subheader_offsets.size):
            print_func('----- Symbol {} -----'.format(symbol_subhead_num))
            hdr = details.parse_symbol_subheader(symbol_subhead_num)
            _print_symbol_header(hdr)
            data = details.get_symbol_bytes(symbol_subhead_num)
            print_func('SSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.label_subheader_offsets is not None:
        for label_subhead_num in range(details.label_subheader_offsets.size):
            print_func('----- Label {} -----'.format(label_subhead_num))
            hdr = details.parse_label_subheader(label_subhead_num)
            _print_label_header(hdr)
            data = details.get_label_bytes(label_subhead_num)
            print_func('LSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.text_subheader_offsets is not None:
        for text_subhead_num in range(details.text_subheader_offsets.size):
            print_func('----- Text {} -----'.format(text_subhead_num))
            hdr = details.parse_text_subheader(text_subhead_num)
            _print_text_header(hdr)
            data = details.get_text_bytes(text_subhead_num)
            print_func('TSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.des_subheader_offsets is not None:
        for des_subhead_num in range(details.des_subheader_offsets.size):
            print_func('----- DES {} -----'.format(des_subhead_num))
            hdr = details.parse_des_subheader(des_subhead_num)
            _print_des_header(hdr)
            data = details.get_des_bytes(des_subhead_num)

            des_id = hdr.DESID if details.nitf_version == '02.10' else hdr.DESTAG

            if des_id.strip() in ['XML_DATA_CONTENT', 'SICD_XML', 'SIDD_XML']:
                xml_str = minidom.parseString(
                    data.decode()).toprettyxml(indent='    ', newl='\n')
                # NB: this may or not exhibit platform dependent choices in which codec (i.e. latin-1 versus utf-8)
                print_func('DESDATA =')
                for line_num, xml_entry in enumerate(xml_str.splitlines()):
                    if line_num == 0:
                        # Remove xml that gets inserted by minidom, if it's not actually there
                        if (not data.startswith(b'<?xml version')) and xml_entry.startswith('<?xml version'):
                            continue
                        print_func(xml_entry)
                    elif xml_entry.strip() != '':
                        # Remove extra new lines if XML is already formatted
                        print_func(xml_entry)
            elif des_id.strip() in ['TRE_OVERFLOW', 'Registered Extensions', 'Controlled Extensions']:
                tres = TREList.from_bytes(data, 0)
                print_func('DESDATA = ')
                _print_tres(tres)
            else:
                # Unknown user-defined data
                print_func('DESDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.res_subheader_offsets is not None:
        for res_subhead_num in range(details.res_subheader_offsets.size):
            print_func('----- RES {} -----'.format(res_subhead_num))
            hdr = details.parse_res_subheader(res_subhead_num)
            _print_res_header(hdr)
            data = details.get_res_bytes(res_subhead_num)
            print_func('RESDATA = {}'.format(_decode_effort(data)))
            print_func('')


##########
# method for dumping file using the print method(s)

def dump_nitf_file(file_name, dest, over_write=True):
    """
    Utility to dump the NITF header and various subheader details to a configurable
    destination.

    Parameters
    ----------
    file_name : str|BinaryIO
        The path to or file-like object containing a NITF 2.1 or 2.0 file.
    dest : str
        'stdout', 'string', 'default' (will use `file_name+'.header_dump.txt'`),
        or the path to an output file.
    over_write : bool
        If `True`, then overwrite the destination file, otherwise append to the
        file.

    Returns
    -------
    None|str
        There is only a return value if `dest=='string'`.
    """

    if dest == 'stdout':
        print_nitf(file_name, dest=sys.stdout)
        return
    if dest == 'string':
        out = StringIO()
        print_nitf(file_name, dest=out)
        value = out.getvalue()
        out.close()  # free the buffer
        return value

    the_out_file = _create_default_output_file(file_name) if dest == 'default' else dest
    if not os.path.exists(the_out_file) or over_write:
        with open(the_out_file, 'w') as the_file:
            print_nitf(file_name, dest=the_file)
    else:
        with open(the_out_file, 'a') as the_file:
            print_nitf(file_name, dest=the_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility to dump NITF 2.1 or 2.0 headers.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'input_file',
        help='The path to a nitf file, or directory to search for NITF files.')
    parser.add_argument(
        '-o', '--output', default='default',
        help="'default', 'stdout', or the path for an output file.\n"
             "* 'default', the output will be at '<input path>.header_dump.txt' \n"
             "   This will be overwritten, if it exists.\n"
             "* 'stdout' will print the information to standard out.\n"
             "* Otherwise, "
             "     if `input_file` is a directory, this is expected to be the path to\n"
             "       an output directory for the output following the default naming scheme.\n"
             "*    if `input_file` a file path, this is expected to be the path to a file \n"
             "       and output will be written there.\n"
             "  In either case, existing output files will be overwritten.")
    args = parser.parse_args()

    if os.path.isdir(args.input_file):
        entries = [os.path.join(args.input_file, part) for part in os.listdir(args.input_file)]
        for file_number, entry in enumerate(filter(_filter_files, entries)):
            if args.output == 'stdout':
                output = args.output
            elif args.output == 'default':
                output = _create_default_output_file(entry, output_directory=None)
            else:
                if not os.path.isdir(args.output):
                    raise IOError(
                        'Provided input is a directory, so provided output must '
                        'be a directory, `stdout`, or `default`.')
                output = _create_default_output_file(entry, output_directory=args.output)
            dump_nitf_file(entry, output)
    else:
        dump_nitf_file(args.input_file, args.output)
