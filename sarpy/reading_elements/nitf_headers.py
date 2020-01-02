# -*- coding: utf-8 -*-
"""
Functionality for dealing with NITF file header information. This is specifically
geared towards SICD file usage, and some funcitonality is not completely general.
"""

import logging
import sys
from collections import OrderedDict
from typing import Union

import numpy

integer_types = (int, )
int_func = int
if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    int_func = long  # to accommodate for 32-bit python 2
    # noinspection PyUnresolvedReferences
    integer_types = (int, long)

##########
# Some hard coded defaults based on current SICD standard
#   used in the NITF Data Extension Header
#   These are used in the SICDDESSSubheader class below

SPECIFICATION_IDENTIFIER = 'SICD Volume 1 Design & Implementation Description Document'
SPECIFICATION_VERSION = '1.1'
SPECIFICATION_DATE = '2014-09-30T00:00:00Z'
SPECIFICATION_NAMESPACE = 'urn:SICD:1.1.0'  # this is expected to be of the form 'urn:SICD:<version>'


class BaseScraper(object):
    """Describes the abstract functionality"""

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def minimum_length(cls):
        """
        The minimum size it takes to write such a header.

        Returns
        -------
        int
        """

        raise NotImplementedError

    @classmethod
    def from_string(cls, value, start, **args):
        """

        Parameters
        ----------
        value: bytes|str
            the header string to scrape

        start : int
            the beginning location in the string
        args : dict
            keyword arguments

        Returns
        -------
        """

        raise NotImplementedError

    def to_string(self):
        """
        Write the object to a properly packed str.

        Returns
        -------
        str
        """

        raise NotImplementedError

    @classmethod
    def _validate(cls, value, start):
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if not isinstance(value, str):
            raise TypeError('Requires a bytes or str type input, got {}'.format(type(value)))

        min_length = cls.minimum_length()
        if len(value[start:]) < min_length:
            raise TypeError('Requires a bytes or str type input of length at least {}, '
                            'got {}'.format(min_length, len(value[start:])))
        return value


class _ItemArrayHeaders(BaseScraper):
    """
    Item array in the NITF header (i.e. Image Segment, Text Segment).
    This is not really meant to be used directly.
    """
    __slots__ = ('subhead_len', 'subhead_sizes', 'item_len', 'item_sizes')

    def __init__(self, subhead_len=0, subhead_sizes=None, item_len=0, item_sizes=None):
        """

        Parameters
        ----------
        subhead_len : int
        subhead_sizes : numpy.ndarray|None
        item_len : int
        item_sizes : numpy.ndarray|None
        """
        if subhead_sizes is None or item_sizes is None:
            subhead_sizes = numpy.array((0, ), dtype=numpy.int64)
            item_sizes = numpy.array((0,), dtype=numpy.int64)
        if subhead_sizes.shape != item_sizes.shape or len(item_sizes.shape) != 1:
            raise ValueError(
                'the subhead_offsets and item_offsets arrays must one-dimensional and the same length')
        self.subhead_len = subhead_len
        self.subhead_sizes = subhead_sizes
        self.item_len = item_len
        self.item_sizes = item_sizes

    def __len__(self):
        return 3 + (self.subhead_len + self.item_len)*self.subhead_sizes.size

    @classmethod
    def minimum_length(cls):
        return 3

    @classmethod
    def from_string(cls, value, start, *args):
        """

        Parameters
        ----------
        value : bytes|str
        start : int
        args : iterable
            this must have two integer elements, which designate the number of
            bytes for the subheader size and the number of bytes for the item size
        Returns
        -------
        _ItemArrayHeaders
        """

        subhead_len, item_len = args
        subhead_len, item_len = int_func(subhead_len), int_func(item_len)

        value = cls._validate(value, start)

        loc = start
        count = int_func(value[loc:loc+3])
        loc += 3
        subhead_sizes = numpy.array((count, ), dtype=numpy.int64)
        item_sizes = numpy.array((count, ), dtype=numpy.int64)
        for i in range(count):
            subhead_sizes[i] = int_func(value[loc: loc+subhead_len])
            loc += subhead_len
            item_sizes[i] = int_func(value[loc: loc+item_len])
            loc += item_len
        return cls(subhead_len, subhead_sizes, item_len, item_sizes)

    def to_string(self):
        out = '{0:3d}'.format(self.subhead_sizes.size)
        subh_frm = '{0:'+str(self.subhead_len) + 'd}'
        item_frm = '{0:' + str(self.item_len) + 'd}'
        for sh_off, it_off in zip(self.subhead_sizes, self.item_sizes):
            out += subh_frm.format(sh_off) + item_frm.format(it_off)
        return out


class ImageComments(BaseScraper):
    """Image comments in the image subheader"""
    __slots__ = ('comments', )

    def __init__(self, comments=None):
        if comments is None:
            self.comments = []
        else:
            if len(comments) > 9:
                raise ValueError('comments must have length less than 10. Got {}'.format(len(comments)))

    def __len__(self):
        return 1 + 80*min(len(self.comments), 9)

    @classmethod
    def minimum_length(cls):
        return 1

    @classmethod
    def from_string(cls, value, start, *args):
        """

        Parameters
        ----------
        value : bytes|str
        start : int
        args : iterable
            this must have two integer elements, which designate the number of
            bytes for the subheader size and the number of bytes for the item size
        Returns
        -------
        _ItemArrayHeaders
        """

        value = cls._validate(value, start)
        loc = start
        count = int_func(value[loc:loc+1])
        loc += 1
        comments = []
        for i in range(count):
            comments.append(value[loc: loc+80])
            loc += 80
        return cls(comments)

    def to_string(self):
        if self.comments is None or len(self.comments) == 0:
            return '0'

        siz = min(len(self.comments), 9)
        out = '{0:1d}'.format(siz)
        for i in range(siz):
            val = self.comments[i]
            if len(val) < 80:
                out += '{0:80s}'.format(val)
            else:
                out += val[:80]
        return out


class ImageBands(BaseScraper):
    """Image bands in the image subheader"""
    __slots__ = ('IREPBAND', 'ISUBCAT', 'IFC', 'IMFLT', 'NLUTS')
    _formats = {'ISUBCAT': '6s', 'IREPBAND': '2s', 'IFC': '1s', 'IMFLT': '3s', 'NLUTS': '1d'}
    _defaults = {'IREPBAND': '\x20'*2, 'IFC': 'N', 'IMFLT': '\x20'*3, 'NLUTS': 0}

    def __init__(self, ISUBCAT, **kwargs):
        if not isinstance(ISUBCAT, (list, tuple)):
            raise ValueError('ISUBCAT must be a list or tuple, got {}'.format(type(ISUBCAT)))
        if len(ISUBCAT) == 0:
            raise ValueError('ISUBCAT must have length > 0.')
        if len(ISUBCAT) > 9:
            raise ValueError('ISUBCAT must have length <= 9. Got {}'.format(ISUBCAT))
        for i, entry in ISUBCAT:
            if not isinstance(entry, str):
                raise TypeError('All entries of ISUBCAT must be an instance of str, '
                                'got {} for entry {}'.format(type(entry), i))
            if len(entry) > 6:
                raise TypeError('All entries of ISUBCAT must be strings of length at most 6, '
                                'got {} for entry {}'.format(len(entry), i))
        self.ISUBCAT = tuple(ISUBCAT)

        for attribute in self.__slots__:
            if attribute == 'ISUBCAT':
                continue
            setattr(self, attribute, kwargs.get(attribute, None))

    def __setattr__(self, attribute, value):
        if attribute not in self.__slots__:
            raise AttributeError('No attribute {}'.format(attribute))
        elif attribute == 'ISUBCAT':
            raise AttributeError('ISUBCAT is intended to be immutable.')

        if value is None:
            super(ImageBands, self).__setattr__(attribute, (self._defaults[attribute], )*len(self.ISUBCAT))
            return

        if not isinstance(value, (list, tuple)):
            raise ValueError('Attribute {} must be a list or tuple, '
                             'got {}'.format(attribute, type(value)))
        if not len(value) == len(self.ISUBCAT):
            raise ValueError('Attribute {} must have the same length as ISUBCAT, '
                             'but {} != {}'.format(attribute, len(value), len(self.ISUBCAT)))

        fmstr = self._formats[attribute]
        flen = int_func(fmstr[:-1])
        for i, entry in value:
            if fmstr[-1] == 's':
                if not isinstance(entry, str):
                    raise TypeError('All entries of {} must be an instance of str, '
                                    'got {} for entry {}'.format(attribute, type(entry), i))
                if len(entry) > flen:
                    raise TypeError('All entries of {} must be strings of length at most {}, '
                                    'got {} for entry {}'.format(attribute, flen, len(entry), i))
            if fmstr[-1] == 'd':
                if not isinstance(entry, integer_types):
                    raise TypeError('All entries of {} must be an instance of int, '
                                    'got {} for entry {}'.format(attribute, type(entry), i))
                if entry >= 10**flen:
                    raise TypeError('All entries of {} must be expressible as strings of length '
                                    'at most {}, got {} for entry {}'.format(attribute, flen, entry, i))
        super(ImageBands, self).__setattr__(attribute, tuple(value))

    def __len__(self):
        return 1 + 13*len(self.ISUBCAT)

    @classmethod
    def minimum_length(cls):
        return 14

    @classmethod
    def from_string(cls, value, start, *args):
        """

        Parameters
        ----------
        value : bytes|str
        start : int
        args : iterable
            this must have two integer elements, which designate the number of
            bytes for the subheader size and the number of bytes for the item size
        Returns
        -------
        _ItemArrayHeaders
        """

        value = cls._validate(value, start)
        loc = start
        count = int_func(value[loc:loc+1])
        loc += 1

        isubcat = []
        irepband = []
        ifc = []
        imflt = []
        nluts = []

        for i in range(count):
            irepband.append(value[loc:loc+2])
            isubcat.append(value[loc+2:loc+8])
            ifc.append(value[loc+8:loc+9])
            imflt.append(value[loc+9:loc+12])
            nluts.append(int_func(value[loc+12:loc+13]))
            loc += 13
        return cls(isubcat, IREPBAND=irepband, IFC=ifc, IMFLT=imflt, NLUTS=nluts)

    def to_string(self):
        siz = len(self.ISUBCAT)
        items = ['{0:1d}'.format(siz), ]
        for i in range(siz):
            for attribute in self.__slots__:
                frmstr = self._formats[attribute]
                val = frmstr.format(getattr(self, attribute)[i])
                if len(val) > int_func(frmstr[:-1]):
                    raise ValueError('Entry {} for attribute {} got formatted as a length {} string, '
                                     'but required to be {}'.format(i, attribute, len(val), frmstr[:-1]))
                items.append(val)
        return ''.join(items)


class OtherHeader(BaseScraper):
    """
    User defined header section at the end of the NITF header
    (i.e. Image Segment, Text Segment). This is not really meant to be
    used directly.
    """
    __slots__ = ('overflow', 'header')

    def __init__(self, overflow=None, header=None):
        if overflow is None:
            self.overflow = 0
        else:
            self.overflow = int_func(overflow)
        if header is None:
            self.header = header
        elif not isinstance(header, str):
            raise ValueError('header must be of string type')
        elif len(header) > 99991:
            logging.warning('Other header will be truncated to 99,9991 characters from {}'.format(header))
            self.header = header[:99991]
            self.overflow += len(header) - 99991
        else:
            self.header = header

    def __len__(self):
        length = 5
        if self.header is not None:
            length += 3 + len(self.header)
        return length

    @classmethod
    def minimum_length(cls):
        return 5

    @classmethod
    def from_string(cls, value, start, *args):
        value = cls._validate(value, start)

        siz = int_func(value[start:start+5])
        if siz == 0:
            return cls()
        else:
            overflow = int_func(value[start+5:start+8])
            header = value[start+8:start+siz]
            return cls(overflow=overflow, header=header)

    def to_string(self):
        out = '{0:5d}'.format(self.__len__())
        if self.header is not None:
            if self.overflow > 999:
                out += '999'
            else:
                out += '{0:3d}'.format(self.overflow)
            out += self.header
        return out


class HeaderScraper(BaseScraper):
    """Generally abstract class for scraping NITF header components"""
    __slots__ = ()  # the possible attribute collection
    _types = {}  # for elements which will be scraped by another class
    _args = {}  # for elements which will be scraped by another class, these are args for the from_string method
    _formats = {}  # {attribute: '<length>d/s'} for each entry in __slots__ not in types
    _defaults = {}  # any default values.
    # If a given attribute doesn't have a give default it is assumed that all spaces is the default

    def __init__(self, **kwargs):
        settable_attributes = self._get_settable_attributes()
        unmatched = [key for key in kwargs if key not in settable_attributes]
        if len(unmatched) > 0:
            raise KeyError('Arguments {} are not in the list of allowed '
                           'attributes {}'.format(unmatched, settable_attributes))
        # the things with no defaults
        no_defaults = [key for key in settable_attributes if
                       (key not in kwargs and not (key in self._defaults or '_'+key in self._types))]

        if len(no_defaults) > 0:
            logging.warning(
                'Attributes {} of class {} are not provided in the construction, '
                'but do not allow for suitable defaults.\nThey will be initialized, '
                'probably to a foolish or invalid value.\nBe sure to set them appropriately '
                'for a valid NITF header.'.format(no_defaults, self.__class__.__name__))

        for attribute in settable_attributes:
            setattr(self, attribute, kwargs.get(attribute, None))

    @classmethod
    def _get_settable_attributes(cls):  # type: () -> tuple
        return tuple(map(lambda x: x[1:] if x[0] == '_' else x, cls.__slots__))

    def __setattr__(self, attribute, value):
        if attribute not in (self.__slots__+self._get_settable_attributes()):
            raise AttributeError('No attribute {}'.format(attribute))
        # is this thing a property? If so, just pass it straight through to setter
        if isinstance(getattr(type(self), attribute, None), property):
            super(HeaderScraper, self).__setattr__(attribute, value)
            return

        if attribute in self._types:
            typ = self._types[attribute]
            if value is None:
                super(HeaderScraper, self).__setattr__(attribute, typ())
            elif isinstance(value, BaseScraper):
                super(HeaderScraper, self).__setattr__(attribute, value)
            else:
                raise ValueError('Attribute {} is expected to be of type {}, '
                                 'got {}'.format(attribute, typ, type(value)))
        else:
            frmt = self._formats[attribute]
            lng = int_func(frmt[:-1])
            if value is None:
                value = self._defaults.get(attribute, None)

            if frmt[-1] == 'd':  # an integer
                if value is None:
                    super(HeaderScraper, self).__setattr__(attribute, 0)
                else:
                    val = int_func(value)
                    if 0 <= val < int_func(10)**lng:
                        super(HeaderScraper, self).__setattr__(attribute, val)
                    else:
                        raise ValueError('Attribute {} is expected to be an integer expressible in {} digits. '
                                         'Got {}.'.format(attribute, lng, value))
            elif frmt[-1] == 's':  # a string
                if value is None:
                    super(HeaderScraper, self).__setattr__(attribute, ' ')
                else:
                    if not isinstance(value, str):
                        raise TypeError('Attribute {} is expected to a string, got {}'.format(attribute, type(value)))

                    if len(value) > lng:
                        logging.warning('Attribute {} is expected to be a string of at most {} characters. '
                                        'Got a value of {} characters, '
                                        'so truncating'.format(attribute, lng, len(value)))
                        super(HeaderScraper, self).__setattr__(attribute, value[:lng])
                    else:
                        super(HeaderScraper, self).__setattr__(attribute, value)

    def __getattribute__(self, attribute):
        if attribute not in (self.__slots__+self._get_settable_attributes()):
            raise AttributeError('No attribute {}'.format(attribute))
        if isinstance(getattr(type(self), attribute, None), property):
            return super(HeaderScraper, self).__getattribute__(attribute)
        if attribute in self._types:
            return super(HeaderScraper, self).__getattribute__(attribute)
        else:
            frmt = self._formats[attribute]
            lng = int_func(frmt[:-1])
            frmtstr = '{0:' + frmt + '}'
            if frmt[-1] == 'd':  # an integer
                return super(HeaderScraper, self).__getattribute__(attribute)
            elif frmt[-1] == 's':  # a string
                val = super(HeaderScraper, self).__getattribute__(attribute)
                if len(val) >= lng:
                    return val[:lng]
                else:
                    return frmtstr.format(val)

    def __len__(self):
        length = 0
        for attribute in self.__slots__:
            if attribute in self._types:
                val = getattr(self, attribute)
                length += len(val)
            else:
                length += int_func(self._formats[attribute][:-1])
        return length

    @classmethod
    def minimum_length(cls):
        min_length = 0
        for attribute in cls.__slots__:
            if attribute in cls._types:
                typ = cls._types[attribute]
                if issubclass(typ, BaseScraper):
                    min_length += typ.minimum_length()
            else:
                min_length += int_func(cls._formats[attribute][:-1])
        return min_length

    @classmethod
    def from_string(cls, value, start, **kwargs):
        if value is None:
            return cls(**kwargs)

        value = cls._validate(value, start)

        fields = OrderedDict()
        loc = start
        for attribute in cls.__slots__:
            if attribute in cls._types:
                typ = cls._types[attribute]
                if not issubclass(typ, BaseScraper):
                    raise TypeError('Invalid class definition, any entry of _types must extend BaseScraper')
                args = cls._args.get(attribute, {})
                val = typ.from_string(value, loc, **args)
                aname = attribute[1:] if attribute[0] == '_' else attribute
                fields[aname] = val  # exclude the underscore from the name
                lngt = len(val)
            else:
                lngt = int_func(cls._formats[attribute][:-1])
                fields[attribute] = value[loc:lngt]
            loc += lngt
        return cls(**fields)

    def to_string(self):
        out = ''
        for attribute in self.__slots__:
            val = getattr(self, attribute)
            if isinstance(val, BaseScraper):
                out += val.to_string()
            elif isinstance(val, integer_types):
                fstr = '{0:'+self._formats[attribute]+'}'
                out += fstr.format(val)
            elif isinstance(val, str):
                out += val
        return out


class NITFSecurityTags(HeaderScraper):
    """The NITF security tags - described in SICD standard 2014-09-30, Volume II, page 20"""
    __slots__ = (
        'CLAS', 'CLSY', 'CODE', 'CTLH',
        'REL', 'DCTP', 'DCDT', 'DCXM',
        'DG', 'DGDT', 'CLTX', 'CAPT',
        'CAUT', 'CRSN', 'SRDT', 'CTLN')
    _formats = {
        'CLAS': '1s', 'CLSY': '2s', 'CODE': '11s', 'CTLH': '2s',
        'REL': '20s', 'DCTP': '2s', 'DCDT': '8s', 'DCXM': '4s',
        'DG': '1s', 'DGDT': '8s', 'CLTX': '43s', 'CAPT': '1s',
        'CAUT': '40s', 'CRSN': '1s', 'SRDT': '8s', 'CTLN': '15s'}
    _defaults = {
        'CLAS': 'U', 'CLSY': '\x20', 'CODE': '\x20', 'CTLH': '\x20',
        'REL': '\x20', 'DCTP': '\x20', 'DCDT': '\x20', 'DCXM': '\x20',
        'DG': '\x20', 'DGDT': '\x20', 'CLTX': '\x20', 'CAPT': '\x20',
        'CAUT': '\x20', 'CRSN': '\x20', 'SRDT': '\x20', 'CTLN': '\x20'}


class NITFHeader(HeaderScraper):
    """
    The NITF file header - described in SICD standard 2014-09-30, Volume II, page 17
    """

    __slots__ = (
        'FHDR', 'FVER', 'CLEVEL', 'STYPE',
        'OSTAID', 'FDT', 'FTITLE', '_Security',
        'FSCOP', 'FSCPYS', 'ENCRYP', 'FBKGC',
        'ONAME', 'OPHONE', 'FL', 'HL',
        '_ImageSegments', 'NUMS', 'NUMX',
        '_TextSegments', '_DataExtensions', 'NUMRES',
        '_UserHeader', '_ExtendedHeader')
    # NB: it appears that NUMS and NUMX are not actually used, and should always be 0
    _formats = {
        'FHDR': '4s', 'FVER': '5s', 'CLEVEL': '2d', 'STYPE': '4s',
        'OSTAID': '10s', 'FDT': '14s', 'FTITLE': '80s',
        'FSCOP': '5d', 'FSCPYS': '5d', 'ENCRYP': '1s', 'FBKGC': '3d',
        'ONAME': '24s', 'OPHONE': '18s', 'FL': '12d', 'HL': '6d',
        'NUMS': '3d', 'NUMX': '3d', 'NUMRES': '3d', }
    _defaults = {
        'FHDR': 'NITF', 'FVER': '02.10', 'STYPE': 'BF01',
        'FSCOP': 0, 'FSCPYS': 0, 'ENCRYP': '0',
        'FBKGC': 0, 'ONAME': '\x20', 'OPHONE': '\x20',
        'HL': 338, 'NUMS': 0, 'NUMX': 0, 'NUMRES': 0}
    _types = {
        '_Security': NITFSecurityTags,
        '_ImageSegments': _ItemArrayHeaders,
        '_TextSegments': _ItemArrayHeaders,
        '_DataExtensions': _ItemArrayHeaders,
        '_UserHeader': OtherHeader,
        '_ExtendedHeader': OtherHeader, }
    _args = {
        '_ImageSegments': {'subhead_len': 6, 'item_len': 10},
        '_TextSegments': {'subhead_len': 4, 'item_len': 5},
        '_DataExtensions': {'subhead_len': 4, 'item_len': 9}, }

    def __init__(self, **kwargs):
        super(NITFHeader, self).__init__(**kwargs)
        self.HL = self.__len__()

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        self.set_attribute('_Security', value)

    @property
    def ImageSegments(self):  # type: () -> _ItemArrayHeaders
        return self._ImageSegments

    @ImageSegments.setter
    def ImageSegments(self, value):
        self.set_attribute('_ImageSegments', value)

    @property
    def TextSegments(self):  # type: () -> _ItemArrayHeaders
        return self._TextSegments

    @TextSegments.setter
    def TextSegments(self, value):
        self.set_attribute('_TextSegments', value)

    @property
    def DataExtensions(self):  # type: () -> _ItemArrayHeaders
        return self._DataExtensions

    @DataExtensions.setter
    def DataExtensions(self, value):
        self.set_attribute('_DataExtensions', value)

    @property
    def UserHeader(self):  # type: () -> OtherHeader
        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        self.set_attribute('_UserHeader', value)

    @property
    def ExtendedHeader(self):  # type: () -> OtherHeader
        return self._ExtendedHeader

    @ExtendedHeader.setter
    def ExtendedHeader(self, value):
        self.set_attribute('_ExtendedHeader', value)


class ImageSegmentHeader(HeaderScraper):
    """The Image Segment header - described in SICD standard 2014-09-30, Volume II, page 24"""
    __slots__ = (
        'IM', 'IID1', 'IDATIM', 'TGTID',
        'IID2', '_Security', 'ENCRYP', 'ISORCE',
        'NROWS', 'NCOLS', 'PVTYPE', 'IREP',
        'ICAT', 'ABPP', 'PJUST', 'ICORDS',
        'IGEOLO', '_ImageComments', 'IC', '_ImageBands',
        'ISYNC', 'IMODE', 'NBPR', 'NBPC', 'NPPBH',
        'NPPBV', 'NBPP', 'IDLVL', 'IALVL',
        'ILOC', 'IMAG', 'UDIDL', 'IXSHDL')
    _formats = {
        'IM': '2s', 'IID1': '10s', 'IDATIM': '14s', 'TGTID': '17s',
        'IID2': '80s', 'ENCRYP': '1s', 'ISORCE': '42s',
        'NROWS': '8d', 'NCOLS': '8d', 'PVTYPE': '3s', 'IREP': '8s',
        'ICAT': '8s', 'ABPP': '2d', 'PJUST': '1s', 'ICORDS': '1s',
        'IGEOLO': '60s', 'IC': '2s', 'ISYNC': '1d', 'IMODE': '1s',
        'NBPR': '4d', 'NBPC': '4d', 'NPPBH': '4d', 'NPPBV': '4d',
        'NBPP': '2d', 'IDLVL': '3d', 'IALVL': '3d', 'ILOC': '10s',
        'IMAG': '4s', 'UDIDL': '5d', 'IXSHDL': '5d'}
    _defaults = {
        'IM': 'IM', 'TGTID': '\x20', 'ENCRYP': '0',
        'IREP': 'NODISPLY', 'ICAT': 'SAR', 'PJUST': 'R',
        'ICORDS': 'G', 'IC': 'NC', 'ISYNC': 0, 'IMODE': 'P',
        'NBPR': 1, 'NBPC': 1, 'IMAG': '1.0 ', 'UDIDL': 0, 'IXSHDL': 0}
    _types = {
        '_Security': NITFSecurityTags,
        '_ImageComments': ImageComments,
        '_ImageBands': ImageBands}

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        self.set_attribute('_Security', value)

    @property
    def ImageComments(self):  # type: () -> ImageComments
        return self._ImageComments

    @ImageComments.setter
    def ImageComments(self, value):
        self.set_attribute('_ImageComments', value)

    @property
    def ImageBands(self):  # type: () -> ImageBands
        return self._ImageBands

    @ImageBands.setter
    def ImageBands(self, value):
        self.set_attribute('_ImageBands', value)


class SICDDESSubheader(HeaderScraper):
    """
    The SICD Data Extension header - described in SICD standard 2014-09-30, Volume II, page 29
    """

    __slots__ = (
        'DESCRC', 'DESSHFT', 'DESSHDT',
        'DESSHRP', 'DESSHSI', 'DESSHSV', 'DESSHSD',
        'DESSHTN', 'DESSHLPG', 'DESSHLPT', 'DESSHLI',
        'DESSHLIN', 'DESSHABS', )
    _formats = {
        'DESCRC': '5d', 'DESSHFT': '8s', 'DESSHDT': '20s',
        'DESSHRP': '40s', 'DESSHSI': '60s', 'DESSHSV': '10s',
        'DESSHSD': '20s', 'DESSHTN': '120s', 'DESSHLPG': '125s',
        'DESSHLPT': '25s', 'DESSHLI': '20s', 'DESSHLIN': '120s',
        'DESSHABS': '200s', }
    _defaults = {
        'DESCRC': 99999, 'DESSHFT': 'XML',
        'DESSHRP': '\x20', 'DESSHSI': SPECIFICATION_IDENTIFIER,
        'DESSHSV': SPECIFICATION_VERSION, 'DESSHSD': SPECIFICATION_DATE,
        'DESSHTN': SPECIFICATION_NAMESPACE,
        'DESSHLPT': '\x20', 'DESSHLI': '\x20', 'DESSHLIN': '\x20',
        'DESSHABS': '\x20', }

    @classmethod
    def minimum_length(cls):
        return 773

    def __len__(self):
        return 773


class DataExtensionHeader(HeaderScraper):
    """
    The data extension header - this may only work for SICD data extension headers?
    described in SICD standard 2014-09-30, Volume II, page 29.
    """

    __slots__ = (
        'DE', 'DESID', 'DESVER', '_Security',
        'DESSHL', '_SICDHeader')
    _formats = {
        'DE': '2s', 'DESID': '25s', 'DESVER': '2d',
        'DESSHL': '4d', }
    _defaults = {
        'DE': 'DE', 'DESID': 'XML_DATA_CONTENT', 'DESVER': 1,
        'DESSHL': 773, }
    _types = {
        '_Security': NITFSecurityTags,
        '_SICDHeader': SICDDESSubheader,
    }

    def __init__(self, **kwargs):
        self.DESSHL = None
        self._Security = None
        self._SICDHeader = None
        super(DataExtensionHeader, self).__init__(**kwargs)

    @classmethod
    def minimum_length(cls):
        return 33

    def __len__(self):
        length = 33
        if self._Security is not None:
            length += len(self._Security)
        if self._SICDHeader is not None:
            length += 773
        return length

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        self.set_attribute('_Security', value)

    @property
    def SICDHeader(self):  # type: () -> Union[SICDDESSubheader, None]
        return self._SICDHeader

    @SICDHeader.setter
    def SICDHeader(self, value):
        if isinstance(value, SICDDESSubheader):
            self.DESSHL = 773
            self._SICDHeader = value
        if value is None:
            if self.DESSHL == 773:
                self._SICDHeader = SICDDESSubheader()
            else:
                self.DESSHL = 0
                self._SICDHeader = None

    @classmethod
    def from_string(cls, value, start, *args):
        if value is None:
            return cls()

        value = cls._validate(value, start)

        fields = OrderedDict()
        loc = start
        for attribute in cls.__slots__[:-1]:
            if attribute in cls._types:
                typ = cls._types[attribute]
                if not issubclass(typ, BaseScraper):
                    raise TypeError('Invalid class definition, any entry of _types must extend BaseScraper')
                args = cls._args.get(attribute, {})
                val = typ.from_string(value, loc, **args)
                aname = attribute[1:] if attribute[0] == '_' else attribute
                fields[aname] = val  # exclude the underscore from the name
                lngt = len(val)
            else:
                lngt = int_func(cls._formats[attribute][:-1])
                fields[attribute] = value[loc:lngt]
            loc += lngt
        if int_func(fields['DESSHL']) == 773:
            fields['SICDHeader'] = SICDDESSubheader.from_string(value[loc:loc+773], 0)
        else:
            fields['DESSHL'] = 0
            fields['SICDHeader'] = None
        return cls(**fields)

    def to_string(self):
        if self.DESSHL == 773 and self.SICDHeader is None:
            self.SICDHeader = SICDDESSubheader()
        elif self.DESSHL != 773:
            self.DESSHL = 0
            self.SICDHeader = None

        out = ''
        for attribute in self.__slots__[:-2]:
            val = getattr(self, attribute)
            if isinstance(val, BaseScraper):
                out += val.to_string()
            elif isinstance(val, integer_types):
                fstr = self._formats[attribute]
                flen = int_func(fstr[:-1])
                val = getattr(self, attribute)
                if val >= 10**flen:
                    raise ValueError('Attribute {} has integer value {}, which cannot be written as '
                                     'a string of length {}'.format(attribute, val, flen))
                out += fstr.format(val)
            elif isinstance(val, str):
                fstr = self._formats[attribute]
                flen = int_func(fstr[:-1])
                val = getattr(self, attribute)
                if len(val) <= flen:
                    out += fstr.format(val)  # left justified of length flen
                else:
                    out += val[:flen]
        if self.SICDHeader is not None:
            out += '0773' + self.SICDHeader.to_string()
        else:
            out += '0000'
        return out
