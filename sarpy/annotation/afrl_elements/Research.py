"""
Definition for the main AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

from typing import Optional

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import SerializableDescriptor, StringDescriptor

from .base import DEFAULT_STRICT
from .DetailCollectionInfo import DetailCollectionInfoType
from .DetailSubCollectionInfo import DetailSubCollectionInfoType
from .DetailImageInfo import DetailImageInfoType
from .DetailSensorInfo import DetailSensorInfoType
from .DetailFiducialInfo import DetailFiducialInfoType
from .DetailObjectInfo import DetailObjectInfoType


_AFRL_SPECIFICATION_NAMESPACE = None  # TODO: 'urn:AFRL:1.0'


class ResearchType(Serializable):
    _fields = (
        'MetadataVersion', 'DetailCollectionInfo', 'DetailSubCollectionInfo',
        'DetailImageInfo', 'DetailSensorInfo', 'DetailFiducialInfo', 'DetailObjectInfo')
    _required = (
        'MetadataVersion', )
    # descriptors
    MetadataVersion = StringDescriptor(
        'MetadataVersion', _required,
        docstring='The version number')  # type: str
    DetailCollectionInfo = SerializableDescriptor(
        'DetailCollectionInfo', DetailCollectionInfoType, _required,
        docstring='High level information about the data collection'
    )  # type: Optional[DetailCollectionInfoType]
    DetailSubCollectionInfo = SerializableDescriptor(
        'DetailSubCollectionInfo', DetailSubCollectionInfoType, _required,
        docstring='Information about sub-division of the overall data collection'
    )  # type: Optional[DetailSubCollectionInfoType]
    DetailImageInfo = SerializableDescriptor(
        'DetailImageInfo', DetailImageInfoType, _required,
        docstring='Information about the referenced image'
    )  # type: Optional[DetailImageInfoType]
    DetailSensorInfo = SerializableDescriptor(
        'DetailSensorInfo', DetailSensorInfoType, _required,
        docstring='Information about the sensor'
    )  # type: Optional[DetailSensorInfoType]
    DetailFiducialInfo = SerializableDescriptor(
        'DetailFiducialInfo', DetailFiducialInfoType, _required,
        docstring='Information about the ground-truthed fiducials'
    )  # type: Optional[DetailFiducialInfoType]
    DetailObjectInfo = SerializableDescriptor(
        'DetailObjectInfo', DetailObjectInfoType, _required,
        docstring='Information about the ground-truthed objects'
    )  # type: Optional[DetailObjectInfoType]

    def __init__(self, MetadataVersion='Unknown', DetailCollectionInfo=None,
                 DetailSubCollectionInfo=None, DetailImageInfo=None,
                 DetailSensorInfo=None, DetailFiducialInfo=None,
                 DetailObjectInfo=None, **kwargs):
        """
        Parameters
        ----------
        MetadataVersion : str
        DetailCollectionInfo : DetailCollectionInfoType
        DetailSubCollectionInfo : DetailSubCollectionInfoType
        DetailImageInfo : DetailImageInfoType
        DetailSensorInfo : DetailSensorInfo
        DetailFiducialInfo : DetailFiducialInfoType
        DetailObjectInfo : DetailObjectInfoType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.MetadataVersion = MetadataVersion
        self.DetailCollectionInfo = DetailCollectionInfo
        self.DetailSubCollectionInfo = DetailSubCollectionInfo
        self.DetailImageInfo = DetailImageInfo
        self.DetailSensorInfo = DetailSensorInfo
        self.DetailFiducialInfo = DetailFiducialInfo
        self.DetailObjectInfo = DetailObjectInfo
        super(ResearchType, self).__init__(**kwargs)

    def to_xml_bytes(self, urn=None, tag='RESEARCH', check_validity=False, strict=DEFAULT_STRICT):
        if urn is None:
            urn = _AFRL_SPECIFICATION_NAMESPACE
        return super(ResearchType, self).to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict)

    def to_xml_string(self, urn=None, tag='RESEARCH', check_validity=False, strict=DEFAULT_STRICT):
        return self.to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict).decode('utf-8')
