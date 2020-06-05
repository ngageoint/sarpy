# -*- coding: utf-8 -*-
"""
The Compensated Phase History Data 0.3 definition.
"""

from typing import Union

from ..base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ...sicd_elements.base import Serializable, _SerializableDescriptor, \
    _IntegerDescriptor, _StringDescriptor

from ...sicd_elements.CollectionInfo import CollectionInfoType
from .Data import DataType
from .Global import GlobalType
from .Channel import ChannelType
from .SRP import SRPTyp
from .Antenna import AntennaType
from .VectorParameters import VectorParametersType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#########
# Module variables
_CPHD_SPECIFICATION_VERSION = '0.3'
_CPHD_SPECIFICATION_DATE = '2011-04-15T00:00:00Z'
_CPHD_SPECIFICATION_NAMESPACE = 'urn:CPHD:0.3'


#########
# CPHD header object

class CPHDHeader(object):
    _fields = (
        'XML_DATA_SIZE', 'XML_BYTE_OFFSET', 'VB_DATA_SIZE', 'VB_BYTE_OFFSET',
        'CPHD_DATA_SIZE', 'CPHD_BYTE_OFFSET', 'CLASSIFICATION', 'RELEASE_INFO')
    _required = (
        'XML_DATA_SIZE', 'XML_BYTE_OFFSET', 'VB_DATA_SIZE', 'VB_BYTE_OFFSET',
        'CPHD_DATA_SIZE', 'CPHD_BYTE_OFFSET')
    # descriptor
    XML_DATA_SIZE = _IntegerDescriptor(
        'XML_DATA_SIZE', _required, strict=True,
        docstring='Size of the XML Metadata in bytes. Does not include the 2 bytes '
                  'of the section terminator.')  # type: int
    XML_BYTE_OFFSET = _IntegerDescriptor(
        'XML_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the XML Metadata in bytes.')  # type: int
    VB_DATA_SIZE = _IntegerDescriptor(
        'VB_DATA_SIZE', _required, strict=True,
        docstring='Size of the Vector Based Metadata in bytes.')  # type: int
    VB_BYTE_OFFSET = _IntegerDescriptor(
        'VB_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the Vector Based Metadata in bytes.')  # type: int
    CPHD_DATA_SIZE = _IntegerDescriptor(
        'CPHD_DATA_SIZE', _required, strict=True,
        docstring='Size of the Compensated PHD arrays in bytes.')  # type: int
    CPHD_BYTE_OFFSET = _IntegerDescriptor(
        'CPHD_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the CPHD data in bytes.')  # type: int
    CLASSIFICATION = _StringDescriptor(
        'CLASSIFICATION', _required, strict=True, default_value='UNCLASSIFIED',
        docstring='Product classification information that is the human-readable banner.')  # type: str
    RELEASE_INFO = _StringDescriptor(
        'RELEASE_INFO', _required, strict=True, default_value='UNRESTRICTED',
        docstring='Product release information.')  # type: str

    def __init__(self, XML_DATA_SIZE=None, XML_BYTE_OFFSET=None,
                 VB_DATA_SIZE=None, VB_BYTE_OFFSET=None,
                 CPHD_DATA_SIZE=None, CPHD_BYTE_OFFSET=None,
                 CLASSIFICATION='UNCLASSIFIED', RELEASE_INFO='UNRESTRICTED'):
        self.XML_DATA_SIZE = XML_DATA_SIZE
        self.XML_BYTE_OFFSET = XML_BYTE_OFFSET
        self.VB_DATA_SIZE = VB_DATA_SIZE
        self.VB_BYTE_OFFSET = VB_BYTE_OFFSET
        self.CPHD_DATA_SIZE = CPHD_DATA_SIZE
        self.CPHD_BYTE_OFFSET = CPHD_BYTE_OFFSET
        self.CLASSIFICATION = CLASSIFICATION
        self.RELEASE_INFO = RELEASE_INFO

    @classmethod
    def from_file_object(cls, fi):
        """
        Extract the CPHD header object from a file opened in byte mode.
        This file object is assumed to be at the correct location for the
        CPHD header.

        Parameters
        ----------
        fi
            The open file object, which will be progressively read.

        Returns
        -------
        CPHDHeader
        """

        the_dict = {}
        while True:
            line = fi.readline().strip()
            if line.startswith(b'\f\n'):
                # we've reached the end of the header section
                break
            parts = line.split(':=')
            if len(parts) != 2:
                raise ValueError('Cannot extract CPHD header value from line {}'.format(line))
            fld = parts[0].strip().encode('utf-8')
            val = parts[1].strip().encode('utf-8')
            if fld not in cls._fields:
                raise ValueError('Cannot extract CPHD header value from line {}'.format(line))
            the_dict[fld] = val
        return cls(**the_dict)



class CPHDType(Serializable):
    """
    """

    _fields = (
        'CollectionInfo', 'Data', 'Global', 'Channel', 'SRP', 'Antenna',
        'VectorParameters')
    _required = (
        'CollectionInfo', 'Data', 'Global', 'Channel', 'SRP', 'VectorParameters')
    # descriptors
    CollectionInfo = _SerializableDescriptor(
        'CollectionInfo', CollectionInfoType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the collection.')  # type: CollectionInfoType
    Data = _SerializableDescriptor(
        'Data', DataType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe binary data components contained in the '
                  'product.')  # type: DataType
    Global = _SerializableDescriptor(
        'Global', GlobalType, _required, strict=DEFAULT_STRICT,
        docstring='Global parameters that apply to metadata components and CPHD '
                  'signal arrays.')  # type: GlobalType
    Channel = _SerializableDescriptor(
        'Channel', ChannelType, _required, strict=DEFAULT_STRICT,
        docstring='Channel specific parameters for CPHD channels.')  # type: ChannelType
    SRP = _SerializableDescriptor(
        'SRP', SRPTyp, _required, strict=DEFAULT_STRICT,
        docstring='The Stabilization Refence Point (SRP) parameters.')  # type: SRPTyp
    Antenna = _SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna parameters that describe antenna orientation, mainlobe '
                  'steering and gain patterns vs. '
                  'time.')  # type: Union[None, AntennaType]
    VectorParameters = _SerializableDescriptor(
        'VectorParameters', VectorParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Structure specifying the Vector parameters provided for '
                  'each channel of a given product.')  # type: VectorParametersType

    def __init__(self, CollectionInfo=None, Data=None, Global=None, Channel=None,
                 SRP=None, Antenna=None, VectorParameters=None, **kwargs):
        """

        Parameters
        ----------
        CollectionInfo : CollectionInfoType
        Data : DataType
        Global : GlobalType
        Channel : ChannelType
        SRP : SRPTyp
        Antenna : NOne|AntennaType
        VectorParameters : VectorParametersType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CollectionInfo = CollectionInfo
        self.Data = Data
        self.Global = Global
        self.Channel = Channel
        self.SRP = SRP
        self.Antenna = Antenna
        self.VectorParameters = VectorParameters
        super(CPHDType, self).__init__(**kwargs)

    def to_xml_bytes(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        return super(CPHDType, self).to_xml_bytes(
            urn=_CPHD_SPECIFICATION_NAMESPACE, tag=tag, check_validity=check_validity, strict=strict)
