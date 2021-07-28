"""
Definition for the main AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")


# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import _StringDescriptor, \
    _SerializableDescriptor, Serializable

from .base import DEFAULT_STRICT
from .DetailCollectionInfo import DetailCollectionInfoType
from .DetailSubCollectionInfo import DetailSubCollectionInfoType
from .DetailImageInfo import DetailImageInfoType
from .DetailSensorInfo import DetailSensorInfoType
from .DetailFiducialInfo import DetailFiducialInfoType
from .DetailObjectInfo import DetailObjectInfoType


_AFRL_SPECIFICATION_NAMESPACE = 'urn:AFRL:1.0'  # TODO: this is completely made up


class ResearchType(Serializable):
    _fields = (
        'MetadataVersion', 'DetailCollectionInfo', 'DetailSubCollectionInfo',
        'DetailImageInfo', 'DetailSensorInfo', 'DetailFiducialInfo', 'DetailObjectInfo')
    _required = _fields
    # descriptors
    MetadataVersion = _StringDescriptor(
        'MetadataVersion', _required, docstring='The version number')  # type: str
    DetailCollectionInfo = _SerializableDescriptor(
        'DetailCollectionInfo', DetailCollectionInfoType, _required)  # type: DetailCollectionInfoType
    DetailSubCollectionInfo = _SerializableDescriptor(
        'DetailSubCollectionInfo', DetailSubCollectionInfoType, _required)  # type: DetailSubCollectionInfoType
    DetailImageInfo = _SerializableDescriptor(
        'DetailImageInfo', DetailImageInfoType, _required)  # type: DetailImageInfoType
    DetailSensorInfo = _SerializableDescriptor(
        'DetailSensorInfo', DetailSensorInfoType, _required)  # type: DetailSensorInfoType
    DetailFiducialInfo = _SerializableDescriptor(
        'DetailFiducialInfo', DetailFiducialInfoType, _required)  # type: DetailFiducialInfoType
    DetailObjectInfo = _SerializableDescriptor(
        'DetailObjectInfo', DetailObjectInfoType, _required)  # type: DetailObjectInfoType

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
