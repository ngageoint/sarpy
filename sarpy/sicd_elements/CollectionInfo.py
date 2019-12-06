"""
The CollectionInfo object definition.
"""

from typing import List

from .base import Serializable, DEFAULT_STRICT, \
    _StringDescriptor, _StringEnumDescriptor, _StringListDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from .blocks import ParameterType


__classification__ = "UNCLASSIFIED"


class RadarModeType(Serializable):
    """Radar mode type container class"""
    _fields = ('ModeType', 'ModeId')
    _required = ('ModeType', )
    # other class variable
    _MODE_TYPE_VALUES = ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP')
    # descriptors
    ModeId = _StringDescriptor(
        'ModeId', _required,
        docstring='Radar imaging mode per Program Specific Implementation Document.')  # type: str
    ModeType = _StringEnumDescriptor(
        'ModeType', _MODE_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="The Radar imaging mode.")  # type: str

    def __init__(self, ModeId=None, ModeType=None, **kwargs):
        """

        Parameters
        ----------
        ModeId : str
        ModeType : str
        kwargs : dict
        """
        self.ModeId, self.ModeType = ModeId, ModeType
        super(RadarModeType, self).__init__(**kwargs)


class CollectionInfoType(Serializable):
    """General information about the collection."""
    _collections_tags = {
        'Parameters': {'array': False, 'child_tag': 'Parameter'},
        'CountryCode': {'array': False, 'child_tag': 'CountryCode'},
    }
    _fields = (
        'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
        'RadarMode', 'Classification', 'Parameters', 'CountryCodes')
    _required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')
    # other class variable
    _COLLECT_TYPE_VALUES = ('MONOSTATIC', 'BISTATIC')
    # descriptors
    CollectorName = _StringDescriptor(
        'CollectorName', _required, strict=DEFAULT_STRICT,
        docstring='Radar platform identifier. For Bistatic collections, list the Receive platform.')  # type: str
    IlluminatorName = _StringDescriptor(
        'IlluminatorName', _required, strict=DEFAULT_STRICT,
        docstring='Radar platform identifier that provided the illumination. For Bistatic collections, '
                  'list the transmit platform.')  # type: str
    CoreName = _StringDescriptor(
        'CoreName', _required, strict=DEFAULT_STRICT,
        docstring='Collection and imaging data set identifier. Uniquely identifies imaging collections per '
                  'Program Specific Implementation Doc.')  # type: str
    CollectType = _StringEnumDescriptor(
        'CollectType', _COLLECT_TYPE_VALUES, _required,
        docstring="Collection type identifier. Monostatic collections include single platform collections with "
                  "unique transmit and receive apertures.")  # type: str
    RadarMode = _SerializableDescriptor(
        'RadarMode', RadarModeType, _required, strict=DEFAULT_STRICT,
        docstring='The radar mode.')  # type: RadarModeType
    Classification = _StringDescriptor(
        'Classification', _required, strict=DEFAULT_STRICT, default_value='UNCLASSIFIED',
        docstring='Contains the human-readable banner. Contains classification, file control and handling, '
                  'file releasing, and/or proprietary markings. Specified per Program Specific '
                  'Implementation Document.')  # type: str
    CountryCodes = _StringListDescriptor(
        'CountryCodes', _required, strict=DEFAULT_STRICT,
        docstring="List of country codes for region covered by the image.")  # type: List[str]
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Free form paramaters object list.')  # type: List[ParameterType]

    def __init__(self, CollectorName=None, IlluminatorName=None, CoreName=None, CollectType=None,
                 RadarMode=None, Classification="UNCLASSIFIED", CountryCodes=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        CollectorName : str
        IlluminatorName : str
        CoreName : str
        CollectType : str
        RadarMode : RadarModeType
        Classification : str
        CountryCodes : list|str
        Parameters : List[ParametersType]
        kwargs : dict
        """

        self.CollectorName, self.IlluminatorName = CollectorName, IlluminatorName
        self.CoreName, self.CollectType = CoreName, CollectType
        self.RadarMode = RadarMode
        self.Classification = Classification
        self.CountryCodes, self.Parameters = CountryCodes, Parameters
        super(CollectionInfoType, self).__init__(**kwargs)
