"""
The CollectionIDType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from .base import DEFAULT_STRICT
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
from sarpy.io.xml.descriptors import StringDescriptor


class CollectionIDType(CollectionInfoType):
    """
    The CollectionID type definition.
    """

    _fields = (
        'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
        'RadarMode', 'Classification', 'ReleaseInfo', 'Parameters', 'CountryCodes')
    _required = ('CollectorName', 'CoreName', 'CollectType', 'RadarMode', 'Classification', 'ReleaseInfo')
    # descriptors
    ReleaseInfo = StringDescriptor(
        'ReleaseInfo', _required, strict=DEFAULT_STRICT, default_value='UNRESTRICTED',
        docstring='The product release information.')  # type: str

    def __init__(self, CollectorName=None, IlluminatorName=None, CoreName=None, CollectType=None,
                 RadarMode=None, Classification="UNCLASSIFIED", ReleaseInfo='UNRESTRICTED',
                 CountryCodes=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        CollectorName : str
        IlluminatorName : str
        CoreName : str
        CollectType : str
        RadarMode : RadarModeType
        Classification : str
        ReleaseInfo : str
        CountryCodes : list|str
        Parameters : ParametersCollection|dict
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ReleaseInfo = ReleaseInfo
        super(CollectionIDType, self).__init__(
            CollectorName=CollectorName, IlluminatorName=IlluminatorName, CoreName=CoreName,
            CollectType=CollectType, RadarMode=RadarMode, Classification=Classification,
            CountryCodes=CountryCodes, Parameters=Parameters, **kwargs)
