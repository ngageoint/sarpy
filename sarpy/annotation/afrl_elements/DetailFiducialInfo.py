"""
Definition for the DetailFiducialInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

# TODO: comments on difficulties
#   - the field names starting with #dB are poorly formed
#   - The PhysicalType seems half complete or something?

from typing import Optional, List

# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, \
    _IntegerDescriptor, _SerializableDescriptor, _SerializableListDescriptor, \
    _StringDescriptor, _find_first_child
from sarpy.io.complex.sicd_elements.blocks import RowColType
from .base import DEFAULT_STRICT
from .blocks import LatLonEleType, RangeCrossRangeType


class ImageLocationType(Serializable):
    _fields = ('CenterPixel', )
    _required = _fields
    # descriptors
    CenterPixel = _SerializableDescriptor(
        'CenterPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='The pixel location of the center of the object')  # type: RowColType

    def __init__(self, CenterPixel=None, **kwargs):
        """
        Parameters
        ----------
        CenterPixel : RowColType|numpy.ndarray|list|tuple
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CenterPixel = CenterPixel
        super(ImageLocationType, self).__init__(**kwargs)


class GeoLocationType(Serializable):
    _fields = ('CenterPixel', )
    _required = _fields
    # descriptors
    CenterPixel = _SerializableDescriptor(
        'CenterPixel', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring='The physical location of the center of the object')  # type: LatLonEleType

    def __init__(self, CenterPixel=None, **kwargs):
        """
        Parameters
        ----------
        CenterPixel : LatLonEleType|numpy.ndarray|list|tuple
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CenterPixel = CenterPixel
        super(GeoLocationType, self).__init__(**kwargs)


class PhysicalLocationType(Serializable):
    _fields = ('Physical', )
    _required = _fields
    # descriptors
    Physical = _SerializableDescriptor(
        'Physical', ImageLocationType, _required, strict=DEFAULT_STRICT,
    )  # type: ImageLocationType

    def __init__(self, Physical=None, **kwargs):
        """
        Parameters
        ----------
        Physical : ImageLocationType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Physical = Physical
        super(PhysicalLocationType, self).__init__(**kwargs)


class TheFiducialType(Serializable):
    _fields = (
        'Name', 'SerialNumber', 'FiducialType', 'DatasetFiducialNumber',
        'ImageLocation', 'GeoLocation',
        'Width3dB', 'Width18dB', 'Ratio3dB18dB',
        'PeakSideLobeRatio', 'IntegratedSideLobeRatio',
        'SlantPlane', 'GroundPlane')
    _required = (
        'FiducialType', 'ImageLocation', 'GeoLocation')
    # descriptors
    Name = _StringDescriptor(
        'Name', _required, strict=DEFAULT_STRICT,
        docstring='Name of the fiducial.')  # type: Optional[str]
    SerialNumber = _StringDescriptor(
        'SerialNumber', _required, strict=DEFAULT_STRICT,
        docstring='The serial number of the fiducial')  # type: Optional[str]
    FiducialType = _StringDescriptor(
        'FiducialType', _required, strict=DEFAULT_STRICT,
        docstring='Description for the type of fiducial')  # type: str
    DatasetFiducialNumber = _IntegerDescriptor(
        'DatasetFiducialNumber', _required,
        docstring='Unique number of the fiducial within the selected dataset, '
                  'defined by the RDE system')  # type: Optional[int]
    ImageLocation = _SerializableDescriptor(
        'ImageLocation', ImageLocationType, _required,
        docstring='Center of the fiducial in the image'
    )  # type: Optional[ImageLocationType]
    GeoLocation = _SerializableDescriptor(
        'GeoLocation', GeoLocationType, _required,
        docstring='Real physical location of the fiducial'
    )  # type: Optional[GeoLocationType]
    Width3dB = _SerializableDescriptor(
        'Width3d', RangeCrossRangeType, _required,
        docstring='The 3 dB impulse response width, in meters'
    ) # type: Optional[RangeCrossRangeType]
    Width18dB = _SerializableDescriptor(
        'Width18dB', RangeCrossRangeType, _required,
        docstring='The 18 dB impulse response width, in meters'
    ) # type: Optional[RangeCrossRangeType]
    Ratio3dB18dB = _SerializableDescriptor(
        'Ratio3dB18dB', RangeCrossRangeType, _required,
        docstring='Ratio of the 3 dB to 18 dB system impulse response width'
    ) # type: Optional[RangeCrossRangeType]
    PeakSideLobeRatio = _SerializableDescriptor(
        'PeakSideLobeRatio', RangeCrossRangeType, _required,
        docstring='Ratio of the peak sidelobe intensity to the peak mainlobe intensity, '
                  'in dB') # type: Optional[RangeCrossRangeType]
    IntegratedSideLobeRatio = _SerializableDescriptor(
        'IntegratedSideLobeRatio', RangeCrossRangeType, _required,
        docstring='Ratio of all the energies in the sidelobes of the '
                  'system impulse response to the energy in the mainlobe, '
                  'in dB') # type: Optional[RangeCrossRangeType]
    SlantPlane = _SerializableDescriptor(
        'SlantPlane', PhysicalLocationType, _required,
        docstring='Center of the object in the slant plane'
    )  # type: Optional[PhysicalLocationType]
    GroundPlane = _SerializableDescriptor(
        'GroundPlane', PhysicalLocationType, _required,
        docstring='Center of the object in the ground plane'
    )  # type: Optional[PhysicalLocationType]

    def __init__(self, Name=None, SerialNumber=None, FiducialType=None,
                 DatasetFiducialNumber=None, ImageLocation=None, GeoLocation=None,
                 Width3dB=None, Width18dB=None, Ratio3dB18dB=None,
                 PeakSideLobeRatio=None, IntegratedSideLobeRatio=None,
                 SlantPlane=None, GroundPlane=None,
                 **kwargs):
        """
        Parameters
        ----------
        Name : str
        SerialNumber : None|str
        FiducialType : str
        DatasetFiducialNumber : None|int
        ImageLocation : ImageLocationType
        GeoLocation : GeoLocationType
        Width3dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        Width18dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        Ratio3dB18dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        PeakSideLobeRatio : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        IntegratedSideLobeRatio : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        SlantPlane : None|PhysicalLocationType
        GroundPlane : None|PhysicalLocationType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.SerialNumber = SerialNumber
        self.FiducialType = FiducialType
        self.DatasetFiducialNumber = DatasetFiducialNumber
        self.ImageLocation = ImageLocation
        self.GeoLocation = GeoLocation
        self.Width3dB = Width3dB
        self.Width18dB = Width18dB
        self.Ratio3dB18dB = Ratio3dB18dB
        self.PeakSideLobeRatio = PeakSideLobeRatio
        self.IntegratedSideLobeRatio = IntegratedSideLobeRatio
        self.SlantPlane = SlantPlane
        self.GroundPlane = GroundPlane
        super(TheFiducialType, self).__init__(**kwargs)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = {}

        the_node = _find_first_child(node, '3dBWidth', xml_ns, ns_key)
        if the_node is None:
            the_node = _find_first_child(node, '_3dBWidth', xml_ns, ns_key)
        if the_node is not None:
            kwargs['Width3dB'] = RangeCrossRangeType.from_node(the_node, xml_ns, ns_key=ns_key)

        the_node = _find_first_child(node, '18dBWidth', xml_ns, ns_key)
        if the_node is None:
            the_node = _find_first_child(node, '_18dBWidth', xml_ns, ns_key)
        if the_node is not None:
            kwargs['Width18dB'] = RangeCrossRangeType.from_node(the_node, xml_ns, ns_key=ns_key)

        the_node = _find_first_child(node, '3dB_18dBRatio18dBWidth', xml_ns, ns_key)
        if the_node is None:
            the_node = _find_first_child(node, '_3dB_18dBRatio18dBWidth', xml_ns, ns_key)
        if the_node is not None:
            kwargs['Ratio3dB18dB'] = RangeCrossRangeType.from_node(the_node, xml_ns, ns_key=ns_key)

        super(TheFiducialType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(TheFiducialType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity,
            strict=strict, exclude=exclude+('Width3dB', 'Width18dB', 'Ratio3dB18dB'))

        if self.Width3dB is not None:
            self.Width3dB.to_node(
                doc, tag='_3dBWidth', ns_key=ns_key, parent=node,
                check_validity=check_validity, strict=strict)
        if self.Width18dB is not None:
            self.Width18dB.to_node(
                doc, tag='_18dBWidth', ns_key=ns_key, parent=node,
                check_validity=check_validity, strict=strict)
        if self.Ratio3dB18dB is not None:
            self.Ratio3dB18dB.to_node(
                doc, tag='_3dB_18dBRatio', ns_key=ns_key, parent=node,
                check_validity=check_validity, strict=strict)

        return node


class DetailFiducialInfoType(Serializable):
    _fields = (
        'NumberOfFiducialsInImage', 'NumberOfFiducialsInScene', 'Fiducials')
    _required = (
        'NumberOfFiducialsInImage', 'NumberOfFiducialsInScene', 'Fiducials')
    _collections_tags = {'Fiducials': {'array': False, 'child_tag': 'Fiducial'}}
    # descriptors
    NumberOfFiducialsInImage = _IntegerDescriptor(
        'NumberOfFiducialsInImage', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the image.')  # type: int
    NumberOfFiducialsInScene = _IntegerDescriptor(
        'NumberOfFiducialsInScene', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the scene.')  # type: int
    Fiducials = _SerializableListDescriptor(
        'Fiducials', TheFiducialType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The object collection')  # type: List[TheFiducialType]

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(DetailFiducialInfoType, self).__init__(**kwargs)
