# -*- coding: utf-8 -*-

import os
import logging
import sys
from xml.etree import ElementTree
from collections import OrderedDict
from typing import Union

import numpy

from .base import BaseChipper, BaseReader, BaseWriter
from .bip import BIPChipper
from ..sicd_elements.SICD import SICDType


##########
# NITF Header reading and writing objects


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
    def from_string(cls, value, start, *args):
        """

        Parameters
        ----------
        value: bytes|str
            the header string to scrape

        start : int
            the beginning location in the string
        args : list
            other argumemnts

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


class ItemArrayHeaders(BaseScraper):
    """Item array in the NITF header (i.e. Image Segment, Text Segment)"""
    __slots__ = ('subhead_len', 'subhead_sizes', 'item_len', 'item_sizes')

    def __init__(self, subhead_len, subhead_sizes, item_len, item_sizes):
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
        ItemArrayHeaders
        """

        subhead_len, item_len = args
        subhead_len, item_len = int(subhead_len), int(item_len)

        value = cls._validate(value, start)

        loc = start
        count = int(value[loc:loc+3])
        loc += 3
        subhead_sizes = numpy.array((count, ), dtype=numpy.int64)
        item_sizes = numpy.array((count, ), dtype=numpy.int64)
        for i in range(count):
            subhead_sizes[i] = int(value[loc: loc+subhead_len])
            loc += subhead_len
            item_sizes[i] = int(value[loc: loc+item_len])
            loc += item_len
        return cls(subhead_len, subhead_sizes, item_len, item_sizes)

    def to_string(self):
        out = '{0:3d}'.format(self.subhead_sizes.size)
        subh_frm = '{0:'+self.subhead_len + 'd}'
        item_frm = '{0:' + self.item_len + 'd}'
        for sh_off, it_off in zip(self.subhead_sizes, self.item_sizes):
            out += subh_frm.format(sh_off) + item_frm.format(it_off)
        return out


class ImageComments(BaseScraper):
    """Image comments in the image subheader"""
    __slots__ = ('comments', )

    def __init__(self, comments):
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
        ItemArrayHeaders
        """

        value = cls._validate(value, start)
        loc = start
        count = int(value[loc:loc+1])
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
            self.set_attribute(attribute, kwargs.get(attribute, None))

    def set_attribute(self, attribute, value):
        if attribute not in self.__slots__:
            raise AttributeError('No attribute {}'.format(attribute))
        elif attribute == 'ISUBCAT':
            raise AttributeError('ISUBCAT is intended to be immutable.')

        if value is None:
            setattr(self, attribute, (self._defaults[attribute], )*len(self.ISUBCAT))
            return

        if not isinstance(value, (list, tuple)):
            raise ValueError('Attribute {} must be a list or tuple, '
                             'got {}'.format(attribute, type(value)))
        if not len(value) == len(self.ISUBCAT):
            raise ValueError('Attribute {} must have the same length as ISUBCAT, '
                             'but {} != {}'.format(attribute, len(value), len(self.ISUBCAT)))

        fmstr = self._formats[attribute]
        flen = int(fmstr[:-1])
        for i, entry in value:
            if fmstr[-1] == 's':
                if not isinstance(entry, str):
                    raise TypeError('All entries of {} must be an instance of str, '
                                    'got {} for entry {}'.format(attribute, type(entry), i))
                if len(entry) > flen:
                    raise TypeError('All entries of {} must be strings of length at most {}, '
                                    'got {} for entry {}'.format(attribute, flen, len(entry), i))
            if fmstr[-1] == 'd':
                if not isinstance(entry, int):
                    raise TypeError('All entries of {} must be an instance of int, '
                                    'got {} for entry {}'.format(attribute, type(entry), i))
                if entry >= 10**flen:
                    raise TypeError('All entries of {} must be expressible as strings of length '
                                    'at most {}, got {} for entry {}'.format(attribute, flen, entry, i))
        setattr(self, tuple(value))

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
        ItemArrayHeaders
        """

        value = cls._validate(value, start)
        loc = start
        count = int(value[loc:loc+1])
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
            nluts.append(int(value[loc+12:loc+13]))
            loc += 13
        return cls(isubcat, IREPBAND=irepband, IFC=ifc, IMFLT=imflt, NLUTS=nluts)

    def to_string(self):
        siz = len(self.ISUBCAT)
        items = ['{0:1d}'.format(siz), ]
        for i in range(siz):
            for attribute in self.__slots__:
                frmstr = self._formats[attribute]
                val = frmstr.format(getattr(self, attribute)[i])
                if len(val) > int(frmstr[:-1]):
                    raise ValueError('Entry {} for attribute {} got formatted as a length {} string, '
                                     'but required to be {}'.format(i, attribute, len(val), frmstr[:-1]))
                items.append(val)
        return ''.join(items)


class OtherHeader(BaseScraper):
    __slots__ = ('overflow', 'header')

    def __init__(self, overflow=None, header=None):
        if overflow is None:
            self.overflow = 0
        else:
            self.overflow = int(overflow)
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

        siz = int(value[start:start+5])
        if siz == 0:
            return cls()
        else:
            overflow = int(value[start+5:start+8])
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
        no_defaults = [key for key in settable_attributes if (key not in kwargs and key not in self._defaults)]

        if len(no_defaults) > 0:
            logging.warning(
                'Attributes {} of class {} are not provided in the construction, '
                'but do not allow for suitable defaults. They will be initialized, '
                'probably to a foolish or invalid value. Be sure to set them appropriately '
                'for a valid NITF header.'.format(no_defaults, self.__class__.__name__))

        for attribute in settable_attributes:
            self.set_attribute(attribute, kwargs.get(attribute, None))

    @classmethod
    def _get_settable_attributes(cls):  # type: () -> tuple
        return tuple(map(lambda x: x[1:] if x[0] == '_' else x, cls.__slots__))

    def set_attribute(self, attribute, value):
        """
        Set the attribute to the given value. This is the preferred method of
        setting values, because it includes validity checks.

        Parameters
        ----------
        attribute : str
            the attribute name
        value : None|str|int|BaseScraper
            the desired value
        """

        if attribute not in (self.__slots__+self._get_settable_attributes()):
            raise AttributeError('No attribute {}'.format(attribute))
        # is this thing a property? If so, just pass it straight through to setter
        if isinstance(getattr(type(self), attribute, None), property):
            setattr(self, attribute, value)
            return

        if attribute in self._types:
            typ = self._types[attribute]
            if value is None:
                setattr(self, attribute, typ())
            elif isinstance(value, BaseScraper):
                setattr(self, attribute, value)
            else:
                raise ValueError('Attribute {} is expected to be of type {}, '
                                 'got {}'.format(attribute, typ, type(value)))
        else:
            frmt = self._formats[attribute]
            lng = int(frmt[:-1])
            if value is None:
                value = self._defaults.get(attribute, None)

            if frmt[-1] == 'd':  # an integer
                if value is None:
                    setattr(self, attribute, 0)
                else:
                    val = int(value)
                    if 0 <= val < 10**lng:
                        setattr(self, attribute, val)
                    else:
                        raise ValueError('Attribute {} is expected to be an integer expressible in {} digits. '
                                         'Got {}.'.format(attribute, lng, value))
            elif frmt[-1] == 's':  # a string
                if value is None:
                    setattr(self, attribute, ' '*lng)
                else:
                    if not isinstance(value, str):
                        raise TypeError('Attribute {} is expected to a string, got {}'.format(attribute, type(value)))

                    if len(value) > lng:
                        logging.warning('Attribute {} is expected to be a string of at most {} characters. '
                                        'Got a value of {} characters, '
                                        'so truncating'.format(attribute, lng, len(value)))
                        setattr(self, attribute, value[:lng])
                    else:
                        setattr(self, attribute, value)

    def __len__(self):
        length = 0
        for attribute in self.__slots__:
            if attribute in self._types:
                val = getattr(self, attribute)
                length += val.length
            else:
                length += int(self._formats[attribute][:-1])
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
                min_length += int(cls._formats[attribute][:-1])
        return min_length

    @classmethod
    def from_string(cls, value, start, *args):
        if value is None:
            return cls()

        value = cls._validate(value, start)

        fields = OrderedDict()
        loc = start
        for attribute in cls.__slots__:
            if attribute in cls._types:
                typ = cls._types[attribute]
                if not issubclass(typ, BaseScraper):
                    raise TypeError('Invalid class definition, any entry of _types must extend BaseScraper')
                args = cls._args.get(attribute, [])
                val = typ.from_string(value, loc, *args)
                aname = attribute[1:] if attribute[0] == '_' else attribute
                fields[aname] = val  # exclude the underscore from the name
                lngt = len(val)
            else:
                lngt = int(cls._formats[attribute][:-1])
                fields[attribute] = value[loc:lngt]
            loc += lngt
        return cls(**fields)

    def to_string(self):
        out = ''
        for attribute in self.__slots__:
            val = getattr(self, attribute)
            if isinstance(val, BaseScraper):
                out += val.to_string()
            elif isinstance(val, int):
                fstr = self._formats[attribute]
                flen = int(fstr[:-1])
                val = getattr(self, attribute)
                if val >= 10**flen:
                    raise ValueError('Attribute {} has integer value {}, which cannot be written as '
                                     'a string of length {}'.format(attribute, val, flen))
                out += fstr.format(val)
            elif isinstance(val, str):
                fstr = self._formats[attribute]
                flen = int(fstr[:-1])
                val = getattr(self, attribute)
                if len(val) <= flen:
                    out += fstr.format(val)  # left justified of length flen
                else:
                    out += val[:flen]
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
        'DG': '\x20', 'CLTX': '\x20', 'CAPT': '\x20', 'CAUT': '\x20',
        'CRSN': '\x20', 'SRDT': '\x20', 'CTLN': '\x20'}


class NITFHeader(HeaderScraper):
    """The NITF file header - described in SICD standard 2014-09-30, Volume II, page 17"""
    __slots__ = (
        'FHDR', 'FVER', 'CLEVEL', 'STYPE',
        'OSTAID', 'FDT', 'FTITLE', '_Security',
        'FSCOP', 'FSCPYS', 'ENCRYP', 'FBKGC',
        'ONAME', 'OPHONE', 'FL', 'HL',
        '_ImageSegments', 'NUMS', 'NUMX',
        '_TextSegments', '_DataExtensions', 'NUMRES',
        '_UserHeader', '_ExtendedHeader')
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
        '_ImageSegments': ItemArrayHeaders,
        '_TextSegments': ItemArrayHeaders,
        '_DataExtensions': ItemArrayHeaders,
        '_UserHeader': OtherHeader,
        '_ExtendedHeader': OtherHeader, }
    _args = {
        '_ImageSegments': (6, 10),  # these are the sizes for the image subheader and image size info
        '_TextSegments': (4, 5),
        '_DataExtensions': (4, 9), }

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
    def ImageSegments(self):  # type: () -> ItemArrayHeaders
        return self._ImageSegments

    @ImageSegments.setter
    def ImageSegments(self, value):
        self.set_attribute('_ImageSegments', value)

    @property
    def TextSegments(self):  # type: () -> ItemArrayHeaders
        return self._TextSegments

    @TextSegments.setter
    def TextSegments(self, value):
        self.set_attribute('_TextSegments', value)

    @property
    def DataExtensions(self):  # type: () -> ItemArrayHeaders
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
        'IM': 'IM', 'TGTIM': '\x20', 'ENCRYP': '0',
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
        'DESSHRP': '\x20', 'DESSHSI': 'SICD Volume 1 Design & Implementation Description Document',
        'DESSHSV': '1.1', 'DESSHSD': '2014-09-30T00:00:00Z', 'DESSHTN': 'urn:SICD:1.1.0',
        'DESSHLPT': '\x20', 'DESSHLI': '\x20', 'DESSHLIN': '\x20',
        'DESSHABS': '\x20', }

    @classmethod
    def minimum_length(cls):
        return 773

    def __len__(self):
        return 773


class DataExtensionHeader(HeaderScraper):
    """The data extension header - this will only really work for SICD headers"""
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
        if self._SICDHeader is None:
            return 33
        else:
            return 773+33

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
                args = cls._args.get(attribute, [])
                val = typ.from_string(value, loc, *args)
                aname = attribute[1:] if attribute[0] == '_' else attribute
                fields[aname] = val  # exclude the underscore from the name
                lngt = len(val)
            else:
                lngt = int(cls._formats[attribute][:-1])
                fields[attribute] = value[loc:lngt]
            loc += lngt
        if int(fields['DESSHL']) == 773:
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
            elif isinstance(val, int):
                fstr = self._formats[attribute]
                flen = int(fstr[:-1])
                val = getattr(self, attribute)
                if val >= 10**flen:
                    raise ValueError('Attribute {} has integer value {}, which cannot be written as '
                                     'a string of length {}'.format(attribute, val, flen))
                out += fstr.format(val)
            elif isinstance(val, str):
                fstr = self._formats[attribute]
                flen = int(fstr[:-1])
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


#########
# Helper object for potentially reading.
# This initially parses NITF header


class NITFDetails(object):
    """
    SICD are stored in NITF 2.1 files. This class allows for suitable parsing
    of the header information in a NITF 2.1 file.
    """
    __slots__ = (
        '_file_name', '_nitf_header', '_is_sicd', '_sicd_meta', '_img_headers',
        'img_segment_offsets', 'img_segment_rows', 'img_segment_columns')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file containing a SICD
        """

        self._file_name = file_name
        self._nitf_header = None
        self._is_sicd = False
        self._sicd_meta = None
        self._img_headers = None
        self.img_segment_offsets = None
        self.img_segment_rows = None
        self.img_segment_columns = None

        with open(file_name, mode='rb') as fi:
            # Read the first 9 bytes to verify NITF
            version_info = fi.read(9).decode('utf-8')
            if version_info != 'NITF02.10':
                raise IOError('Note a NITF 2.1 file, and cannot contain a SICD')
            # get the header length
            fi.seek(354)  # offset to first field of interest
            header_length = int(fi.read(6))
            # go back to the beginning of the file, and parse the whole header
            fi.seek(0)
            header_string = fi.read(header_length)
            self._nitf_header = NITFHeader.from_string(header_string, 0)
            if self._nitf_header.NUMS > 0:
                raise IOError('SICD does not allow for graphics segments.')
            if self._nitf_header.NUMX > 0:
                raise IOError('SICD does not allow for reserved extension segments.')
            if self._nitf_header.DataExtensions.subhead_sizes.size != 1:
                raise IOError('SICD requires exactly one data extension, containing the '
                              'SICD xml structure.')
            # construct the image offset arrays
            # the image data is packed immediately after the header as
            #   subheader data then image data
            self.img_segment_offsets = header_length + numpy.cumsum(self._nitf_header.ImageSegments.subhead_sizes)
            self.img_segment_offsets[1:] += numpy.cumsum(self._nitf_header.ImageSegments.item_sizes)
            self.img_segment_rows = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
            self.img_segment_columns = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
            # we should parse the image subheaders collection
            self._img_headers = []
            for i, img_segment_offset in enumerate(self.img_segment_offsets):
                img_head_size = self._nitf_header.ImageSegments.subhead_sizes[i]
                img_head_offset = img_segment_offset - img_head_size
                fi.seek(img_head_offset)
                header_string = fi.read(img_head_size)
                im_header = ImageSegmentHeader.from_string(header_string, 0)
                self._img_headers.append(im_header)
                self.img_segment_rows[i] = im_header.NROWS
                self.img_segment_columns[i] = im_header.NCOLS
            total_offset = self.img_segment_offsets[-1] + self._nitf_header.ImageSegments.item_sizes[-1]
            # the text extensions are packed next - this appears to play no particular role in SICD
            total_offset += numpy.sum(self._nitf_header.TextSegments.subhead_sizes +
                                      self._nitf_header.TextSegments.item_sizes)
            # the data extensions are packed next - note that we already verify
            # exactly one data extension
            total_offset += self._nitf_header.DataExtensions.subhead_sizes[0]
            des_length = self._nitf_header.DataExtensions.item_sizes[0]
            fi.seek(total_offset)
            data_extension = fi.read(des_length)

            try:
                root_node = ElementTree.fromstring(data_extension)  # handles bytes?
                if root_node.tag.split('}', 1)[-1] == 'SICD':
                    self._is_sicd = True
            except Exception:
                raise

            if self._is_sicd:
                try:
                    self._sicd_meta = SICDType.from_node(root_node.find('SICD'))
                except Exception:
                    self._is_sicd = False
                    raise

    @property
    def is_sicd(self):
        return self._is_sicd

    @property
    def sicd_meta(self):
        return self._sicd_meta


#######
#  The actual reading and writing implementation


def amp_phase_to_complex(lookup_table):
    """
    This constructs the function to convert from AMP8I_PHS8I format data to complex128 data.

    Parameters
    ----------
    lookup_table

    Returns
    -------
    callable
    """

    if not isinstance(lookup_table, numpy.ndarray):
        raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
    if lookup_table.dtype != numpy.float64:
        raise ValueError('requires a numpy.ndarray of float64 dtype, got {}'.format(lookup_table.dtype))
    if lookup_table.shape != (256, ):
        raise ValueError('Requires a one-dimensional numpy.ndarray with 256 elements, '
                         'got shape {}'.format(lookup_table.shape))

    def converter(data):
        if not isinstance(data, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(data)))

        if data.dtype != numpy.uint8:
            raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(data.dtype))

        if len(data.shape) == 3:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the first dimension), got shape {}'.format(data.shape))

        out = numpy.zeros((data.shape[0] / 2, data.shape[1], data.shape[2]), dtype=numpy.complex128)
        amp = lookup_table[data[0::2, :, :]]
        theta = data[1::2, :, :]*(2*numpy.pi/256)  # TODO: complex 64 or 128?
        out.real = amp*numpy.cos(theta)
        out.imag = amp*numpy.sin(theta)
        return out
    return converter


class MultiSegmentChipper(BaseChipper):
    """
    This is required because NITF files have specified size and row/column limits.
    Any sufficiently large SICD collect must be broken into a series of image segments (along rows).
    We must have a parser to transparently extract data between this collection of image segments.
    """
    __slots__ = ('_file_name', '_data_size', '_dtype', '_complex_out',
                 '_symmetry', '_swap_bytes', '_row_starts', '_row_ends',
                 '_bands_ip', '_child_chippers')

    def __init__(self, file_name, data_sizes, data_offsets, data_type, symmetry=None,
                 complex_type=True, swap_bytes=False, bands_ip=1):
        """

        Parameters
        ----------
        file_name : str
            The name of the file from which to read
        data_sizes : numpy.ndarray
            Two-dimensional array of [row, col] sizes
        data_offsets : numpy.ndarray
            Offset for each image segment from the start of the file
        data_type : numpy.dtype
            The data type of the underlying file
        symmetry : tuple
            See `BaseChipper` for description of 3 element tuple of booleans.
        complex_type : callable|bool
            See `BaseChipper` for description of `complex_type`
        swap_bytes : bool
            are the endian-ness of the os and file different?
        bands_ip : int
            number of bands - this will always be one for sicd.
        """

        if not isinstance(data_sizes, numpy.ndarray):
            raise ValueError('data_sizes must be an numpy.ndarray, not {}'.format(type(data_sizes)))
        if not (len(data_sizes.shape) == 2 and data_sizes.shape[1] == 2):
            raise ValueError('data_sizes must be an Nx2 numpy.ndarray, not shape {}'.format(data_sizes.shape))
        if not numpy.all(data_sizes[0, :] == data_sizes[0, 1]):
            raise ValueError(
                'SICDs are broken up by row only, so the the second column of data_sizes '
                'must be have constant value {}'.format(data_sizes))

        if not isinstance(data_offsets, numpy.ndarray):
            raise ValueError('data_offsets must be an numpy.ndarray, not {}'.format(type(data_offsets)))
        if not (len(data_offsets.shape) == 1):
            raise ValueError(
                'data_sizes must be an one-dimensional numpy.ndarray, '
                'not shape {}'.format(data_offsets.shape))

        if data_sizes.shape[0] != data_offsets.size:
            raise ValueError(
                'data_sizes and data_offsets arguments must have compatible '
                'shape {} - {}'.format(data_sizes.shape, data_sizes.size))

        self._file_name = file_name
        if complex_type is not False:
            self._dtype = numpy.complex64
        else:
            self._dtype = data_type
        # all of the actual reading and reorienting work will be done by these
        # child chippers, which will read from their respective image segments
        self._child_chippers = tuple(
            BIPChipper(file_name, data_type, img_siz, symmetry=symmetry,
                       complex_type=complex_type, data_offset=img_off,
                       swap_bytes=swap_bytes, bands_ip=bands_ip)
            for img_siz, img_off in zip(data_sizes, data_offsets))

        self._row_starts = numpy.zeros((data_sizes.shape[0], ), dtype=numpy.int64)
        self._row_ends = numpy.cumsum(data_sizes[: 0])
        self._row_starts[1:] = self._row_ends[:-1]
        self._bands_ip = int(bands_ip)

        data_size = (self._row_ends[-1], data_sizes[0, 1])
        # all of the actual reading and reorienting done by child chippers,
        # so do not reorient or change type at this level
        super(MultiSegmentChipper, self).__init__(data_size, symmetry=(False, False, False), complex_type=False)

    def _read_raw_fun(self, range1, range2):
        range1, range2 = self._reorder_arguments(range1, range2)
        # this method just assembles the final data from the child chipper pieces
        rows = numpy.arange(*range1, dtype=numpy.int64)  # array
        cols_size = int((range2[1] - range2[0] - 1)/range2[2]) + 1
        if self._bands_ip == 1:
            out = numpy.empty((rows.size, cols_size), dtype=self._dtype)
        else:
            out = numpy.empty((self._bands_ip, rows.size, cols_size), dtype=numpy.complex64)
        for row_start, row_end, child_chipper in zip(self._row_starts, self._row_ends, self._child_chippers):
            row_bool = ((rows >= row_start) & (rows < row_end))
            if numpy.any(row_bool):
                crows = rows[row_bool]
                crange1 = (crows[0]-row_start, crows[-1]+1-row_start, range1[2])
                if self._bands_ip == 1:
                    out[row_bool, :] = child_chipper(crange1, range2)
                else:
                    out[:, row_bool, :] = child_chipper(crange1, range2)
        return out


def complex_to_amp_phase(lookup_table):
    """
    This constructs the function to convert from complex64 or 128 to AMP8I_PHS8I format data.

    Parameters
    ----------
    lookup_table

    Returns
    -------
    callable
    """

    if not isinstance(lookup_table, numpy.ndarray):
        raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
    if lookup_table.dtype != numpy.float64:
        raise ValueError('requires a numpy.ndarray of float64 dtype, got {}'.format(lookup_table.dtype))
    if lookup_table.shape != (256, ):
        raise ValueError('Requires a one-dimensional numpy.ndarray with 256 elements, '
                         'got shape {}'.format(lookup_table.shape))

    def converter(data):
        if not isinstance(data, numpy.ndarray):
            raise ValueError('Requires a numpy.ndarray, got {}'.format(type(data)))
        if data.dtype not in (numpy.complex64, numpy.complex128):
            raise ValueError('Requires a numpy.ndarray of complex dtype, got {}'.format(data.dtype))
        if len(data.shape) == 2:
            raise ValueError('Requires a two-dimensional numpy.ndarray, got {}'.format(data.shape))

        new_shape = (data.shape[0], data.shape[1], 2)
        # TODO: BSQ nonsense for 3-d array?

        out = numpy.zeros(new_shape, dtype=numpy.uint8)
        # NB: for numpy before 1.10, digitize requires 1-d
        out[:, :, 0] = numpy.digitize(numpy.abs(data).ravel(), lookup_table, right=False).reshape(data.shape)
        out[:, :, 1] = numpy.arctan2(data.real, data.imag)*(256/(2*numpy.pi))
        # truncation takes care of properly rolling negative to positive
        return out

    return converter


def complex_to_int(data):
    """
    This converts from complex64 or 128 data to int16 data.

    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    if not isinstance(data, numpy.ndarray):
        raise ValueError('Requires a numpy.ndarray, got {}'.format(type(data)))
    if data.dtype not in (numpy.complex64, numpy.complex128):
        raise ValueError('Requires a numpy.ndarray of complex dtype, got {}'.format(data.dtype))
    if len(data.shape) == 2:
        raise ValueError('Requires a two-dimensional numpy.ndarray, got {}'.format(data.shape))

    new_shape = (data.shape[0], data.shape[1], 2)
    # TODO: BSQ nonsense for 3-d array?

    if data.dtype == numpy.complex128:
        view_dtype = numpy.float64
    else:
        view_dtype = numpy.float32

    i16_info = numpy.iinfo(numpy.int16)
    data_view = data.view(dtype=view_dtype).reshape(new_shape)
    out = numpy.zeros(new_shape, dtype=numpy.int16)
    out[:] = numpy.round(numpy.clip(data_view, i16_info.min, i16_info.max))
    # this is nonsense without the clip - gets cast to int64 and then truncated.
    # should we round? Without it, it will be the floor, I believe.
    return out


class SICDReader(BaseReader):
    __slots__ = ('_nitf_details', '_sicd_meta', '_chipper')

    def __init__(self, file_name):
        self._nitf_details = NITFDetails(file_name)
        if not self._nitf_details.is_sicd:
            raise ValueError(
                'The input file passed in appears to be a NITF 2.1 file that does not contain valid sicd metadata.')

        self._sicd_meta = self._nitf_details.sicd_meta

        pixel_type = self._sicd_meta.ImageData.PixelType
        complex_type = True
        if pixel_type == 'RE32F_IM32F':
            dtype = numpy.float32
        elif pixel_type == 'RE16I_IM16I':
            dtype = numpy.int16
        elif pixel_type == 'AMP8I_PHS8I':
            dtype = numpy.uint8
            complex_type = amp_phase_to_complex(self._sicd_meta.ImageData.AmpTable)
            # TODO: is the above correct?
            # raise ValueError('Pixel Type `AMP8I_PHS8I` is not currently supported.')
        else:
            raise ValueError('Pixel Type {} not recognized.'.format(pixel_type))

        data_sizes = numpy.column_stack(
            (self._nitf_details.img_segment_rows, self._nitf_details.img_segment_columns), dtype=numpy.int64)
        # SICDs are required to be stored as big-endian
        swap_bytes = (sys.byteorder != 'big')
        # SICDs require no symmetry reorientation
        symmetry = (False, False, False)
        # construct our chipper
        chipper = MultiSegmentChipper(
            file_name, data_sizes, self._nitf_details.img_segment_rows.copy(), dtype,
            symmetry=symmetry, complex_type=complex_type, swap_bytes=swap_bytes, bands_ip=1)

        super(SICDReader, self).__init__(self._sicd_meta, chipper)


class SICDWriter(BaseWriter):
    pass


# TODO: complete SICDWriter - make sure it's coherent
