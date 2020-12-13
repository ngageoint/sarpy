"""
A utility for dumping a NITF header to the console.

Contributed by Austin Lan of L3/Harris.
"""

import functools
import os
import sys
import xml.dom.minidom

from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.base import TREList, UserHeaderType
from sarpy.io.general.nitf_elements.des import DESUserHeader
from sarpy.io.general.nitf_elements.res import RESUserHeader
from sarpy.io.general.nitf_elements.tres.tre_elements import TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Austin Lan"


# Custom print function
print_func = print


def _decode_effort(value):
    # type: (bytes) -> Union[bytes, str]
    try:
        return value.decode()
    except Exception:
        return value


def print_elem_field(elem, field, prefix=''):
    if elem is None or field is None:
        return
    value = getattr(elem, field, None)
    if value is None:
        value = ''
    print_func('{}{} = {}'.format(prefix, field, value))


def print_elem(elem, prefix=''):
    if elem is None:
        return
    for field in elem._ordering:
        print_elem_field(elem, field, prefix=prefix)


def print_elem_list(elem_list, prefix=''):
    if not elem_list:
        return
    for i, elem in enumerate(elem_list):
        print_elem(elem, prefix='{}[{}].'.format(prefix, i))


def print_tre_elem(field, value, prefix=''):
    if field is None:
        return
    if value is None:
        value = ''
    print_func('{}{} = {}'.format(prefix, field, value))


def print_tre_list(elem_list, prefix=''):
    if not elem_list:
        return
    for i, elem in enumerate(elem_list):
        print_tre_dict(elem, '{}[{}].'.format(prefix, i))


def print_tre_dict(elem_dict, prefix=''):
    if not elem_dict:
        return
    for field, value in elem_dict.items():
        if isinstance(value, list):
            print_tre_list(value, '{}{}'.format(prefix, field))
        else:
            print_tre_elem(field, value, prefix)


def print_tres(tres):
    for tre in tres:
        print_func('')
        if isinstance(tre.DATA, TREElement):
            print_tre_dict(tre.DATA.to_dict(), prefix='{}.'.format(tre.TAG))
        else:
            # Unknown TRE
            print_tre_elem('DATA', _decode_effort(tre.DATA), prefix='{}.'.format(tre.TAG))


def print_file_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='FS')
        elif field == 'FBKGC':
            value = getattr(hdr, field, None)
            print_func('FBKGC = {} {} {}'.format(value[0], value[1], value[2]))
        elif field in ['ImageSegments', 'GraphicsSegments', 'SymbolSegments', 'LabelSegments',
                       'TextSegments', 'DataExtensions', 'ReservedExtensions']:
            pass
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                print_tres(value.data.tres)
        else:
            print_elem_field(hdr, field)


def print_image_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='IS')
        elif field in ['Comments', 'Bands']:
            print_elem_list(getattr(hdr, field, None), prefix='{}'.format(field))
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                print_tres(value.data.tres)
        else:
            print_elem_field(hdr, field)


def print_graphics_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='SS')
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                print_tres(value.data.tres)
        else:
            print_elem_field(hdr, field)


def print_symbol_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='SS')
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                print_tres(value.data.tres)
        else:
            print_elem_field(hdr, field)


def print_label_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='LS')
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                print_tres(value.data.tres)
        else:
            print_elem_field(hdr, field)


def print_text_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='TS')
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            assert(isinstance(value, UserHeaderType))
            if value and value.data and value.data.tres:
                print_tres(value.data.tres)
        else:
            print_elem_field(hdr, field)


def print_des_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='DES')
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            if isinstance(value, DESUserHeader):
                if value.data:
                    # Unknown user-defined subheader
                    print_func('DESSHF = {}'.format(_decode_effort(value.data)))
            else:
                # e.g., XMLDESSubheader
                print_elem(value, prefix='DESSHF.')
        else:
            print_elem_field(hdr, field)


def print_res_header(hdr):
    for field in hdr._ordering:
        if field == 'Security':
            print_elem(getattr(hdr, field, None), prefix='RE')
        elif field in ['UserHeader', 'ExtendedHeader']:
            value = getattr(hdr, field, None)
            if isinstance(value, RESUserHeader):
                if value.data:
                    # Unknown user-defined subheader
                    print_func('RESSHF = {}'.format(_decode_effort(value.data)))
            else:
                print_elem(value, prefix='RESSHF.')
        else:
            print_elem_field(hdr, field)


def print_nitf(file_name, dest=sys.stdout):
    # Configure print function for desired destination
    #    - e.g., stdout, string buffer, file
    global print_func
    print_func = functools.partial(print, file=dest)

    details = NITFDetails(file_name)

    print_func('----- File Header -----')
    print_file_header(details.nitf_header)
    print_func('')

    if details.img_subheader_offsets is not None:
        for i in range(details.img_subheader_offsets.size):
            print_func('----- Image {} -----'.format(i))
            hdr = details.parse_image_subheader(i)
            print_image_header(hdr)
            print_func('')

    if details.graphics_subheader_offsets is not None:
        for i in range(details.graphics_subheader_offsets.size):
            print_func('----- Graphic {} -----'.format(i))
            hdr = details.parse_graphics_subheader(i)
            print_graphics_header(hdr)
            data = details.get_graphics_bytes(i)
            print_func('GSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.symbol_subheader_offsets is not None:
        for i in range(details.symbol_subheader_offsets.size):
            print_func('----- Symbol {} -----'.format(i))
            hdr = details.parse_symbol_subheader(i)
            print_symbol_header(hdr)
            data = details.get_symbol_bytes(i)
            print_func('SSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.label_subheader_offsets is not None:
        for i in range(details.label_subheader_offsets.size):
            print_func('----- Label {} -----'.format(i))
            hdr = details.parse_label_subheader(i)
            print_label_header(hdr)
            data = details.get_label_bytes(i)
            print_func('LSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.text_subheader_offsets is not None:
        for i in range(details.text_subheader_offsets.size):
            print_func('----- Text {} -----'.format(i))
            hdr = details.parse_text_subheader(i)
            print_text_header(hdr)
            data = details.get_text_bytes(i)
            print_func('TSDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.des_subheader_offsets is not None:
        for i in range(details.des_subheader_offsets.size):
            print_func('----- DES {} -----'.format(i))
            hdr = details.parse_des_subheader(i)
            print_des_header(hdr)
            data = details.get_des_bytes(i)

            des_id = hdr.DESID if details.nitf_version == '02.10' else hdr.DESTAG

            if des_id.strip() == 'XML_DATA_CONTENT':
                xml_str = xml.dom.minidom.parseString(
                    data.decode()).toprettyxml(indent='    ')
                # Remove extra new lines if XML is already formatted
                xml_str = os.linesep.join(
                    [s for s in xml_str.splitlines() if s.strip()])
                print_func('DESDATA = {}'.format(xml_str))
            elif des_id.strip() in ['TRE_OVERFLOW', 'Registered Extensions', 'Controlled Extensions']:
                tres = TREList.from_bytes(data, 0)
                print_func('DESDATA = ')
                print_tres(tres)
            else:
                # Unknown user-defined data
                print_func('DESDATA = {}'.format(_decode_effort(data)))
            print_func('')

    if details.res_subheader_offsets is not None:
        for i in range(details.res_subheader_offsets.size):
            print_func('----- RES {} -----'.format(i))
            hdr = details.parse_res_subheader(i)
            print_res_header(hdr)
            data = details.get_res_bytes(i)
            print_func('RESDATA = {}'.format(_decode_effort(data)))
            print_func('')


if __name__ == '__main__':
    import argparse

    def argparse_formatter_factory(prog):
        return argparse.ArgumentDefaultsHelpFormatter(prog, width=100)

    parser = argparse.ArgumentParser(
        description='Utility to dump NITF 2.1 or 2.0 headers',
        formatter_class=argparse_formatter_factory)
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output', default='stdout',
                        help="'stdout', 'string', or an output file")
    args = parser.parse_args()

    if args.output == 'stdout':
        # Send output to stdout
        print_nitf(args.input_file, dest=sys.stdout)
    elif args.output == 'string':
        # Send output to string
        from io import StringIO
        str_buf = StringIO()
        print_nitf(args.input_file, dest=str_buf)
        nitf_header = str_buf.getvalue()
        print('String representing NITF header is {} bytes'.format(len(nitf_header)))
    else:
        # Send output to file
        with open(args.output, 'w') as f:
            print_nitf(args.input_file, dest=f)
