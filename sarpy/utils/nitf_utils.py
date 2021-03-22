"""
A utility for dumping a NITF header to the console.

Contributed by Austin Lan of L3/Harris.
"""

from __future__ import print_function
import argparse
import functools
import sys
import xml.dom.minidom
import os
from typing import Union

from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.base import TREList, UserHeaderType
from sarpy.io.general.nitf_elements.des import DESUserHeader
from sarpy.io.general.nitf_elements.res import RESUserHeader
from sarpy.io.general.nitf_elements.tres.tre_elements import TREElement
from sarpy.io.general.nitf_elements.image import MaskSubheader

if sys.version_info[0] < 3:
    import cStringIO as StringIO
else:
    from io import StringIO


__classification__ = "UNCLASSIFIED"
__author__ = "Austin Lan"


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
    if output_directory is None:
        return os.path.splitext(input_file)[0] + '.header_dump.txt'
    else:
        return os.path.join(output_directory, os.path.splitext(os.path.split(input_file)[1])[0] + '.header_dump.txt')


def _decode_effort(value):
    # type: (bytes) -> Union[bytes, str]
    try:
        return value.decode()
    except Exception:
        return value


############
# printing methods

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


def print_mask_header(hdr):
    # type: (Union[None, MaskSubheader]) -> None
    if hdr is None:
        return
    print_func('----- Mask Subheader (part of image data segment) -----')
    for field in hdr._ordering:
        if field in ['BMR', 'TMR']:
            value = getattr(hdr, field, None)
            if value is None:
                continue
            else:
                for the_band, subarray in enumerate(value):
                    print_func('{}BND{} = {}'.format(field, the_band, subarray))
        else:
            print_elem_field(hdr, field, prefix='')


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
    print_mask_header(hdr.mask_subheader)


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

    print_func('')
    print_func('Details for file {}'.format(file_name))
    print_func('')
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
                    data.decode()).toprettyxml(indent='    ', newl='\n')
                # NB: this may or not exhibit platform dependent choices in which codec (i.e. latin-1 versus utf-8)
                print_func('DESDATA =')
                for i, entry in enumerate(xml_str.splitlines()):
                    if i == 0:
                        # Remove xml that gets inserted by minidom, if it's not actually there
                        if (not data.startswith(b'<?xml version')) and entry.startswith('<?xml version'):
                            continue
                        print_func(entry)
                    elif entry.strip() != '':
                        # Remove extra new lines if XML is already formatted
                        print_func(entry)
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


##########
# method for dumping file using the print method(s)

def dump_nitf_file(file_name, dest, over_write=True):
    """
    Performs the writing, basically just calls print_nitf directly.

    Parameters
    ----------
    file_name : str
        The path for to a NITF file.
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
        for i, entry in enumerate(filter(_filter_files, entries)):
            if args.output == 'stdout':
                output = args.output
            elif args.output == 'default':
                output = _create_default_output_file(entry, output_directory=None)
            else:
                if not os.path.isdir(args.output):
                    raise IOError('Provided input is a directory, so provided output must be a directory, `stdout`, or `default`.')
                output = _create_default_output_file(entry, output_directory=args.output)
            dump_nitf_file(entry, output)
    else:
        dump_nitf_file(args.input_file, args.output)
