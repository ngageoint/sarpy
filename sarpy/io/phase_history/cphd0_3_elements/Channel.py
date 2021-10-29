"""
The Channel definition for CPHD 0.3.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union, List

from sarpy.io.phase_history.cphd1_elements.base import DEFAULT_STRICT, FLOAT_FORMAT
from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import IntegerDescriptor, FloatDescriptor, \
    SerializableListDescriptor


class ParametersType(Serializable):
    """
    Channel dependent parameters.
    """

    _fields = (
        'SRP_Index', 'NomTOARateSF', 'FxCtrNom', 'BWSavedNom', 'TOASavedNom',
        'TxAnt_Index', 'RcvAnt_Index', 'TWAnt_Index')
    _required = (
        'SRP_Index', 'NomTOARateSF', 'FxCtrNom', 'BWSavedNom', 'TOASavedNom')
    _numeric_format = {
        'NomTOARateSF': FLOAT_FORMAT, 'FxCtrNom': FLOAT_FORMAT, 'BWSavedNom': FLOAT_FORMAT,
        'TOASavedNom': FLOAT_FORMAT}
    # descriptors
    SRP_Index = IntegerDescriptor(
        'SRP_Index', _required, strict=DEFAULT_STRICT,
        docstring='Index to identify the SRP position function used for the '
                  'channel.')  # type: int
    NomTOARateSF = FloatDescriptor(
        'NomTOARateSF', _required, strict=DEFAULT_STRICT,
        docstring='Scale factor to indicate the fraction of the Doppler spectrum '
                  'that is clear.')  # type: float
    FxCtrNom = FloatDescriptor(
        'FxCtrNom', _required, strict=DEFAULT_STRICT,
        docstring='Nominal center transmit frequency associated with the channel (Hz). '
                  'For DomainType = TOA, FxCtrNom is the center frequency for all '
                  'vectors.')  # type: float
    BWSavedNom = FloatDescriptor(
        'BWSavedNom', _required, strict=DEFAULT_STRICT,
        docstring='Nominal transmit bandwidth associated with the channel (Hz). '
                  'For DomainType = TOA, BWSavedNom is the bandwidth saved for all '
                  'vectors.')  # type: float
    TOASavedNom = FloatDescriptor(
        'TOASavedNom', _required, strict=DEFAULT_STRICT,
        docstring='Nominal span in TOA saved for the channel. For DomainType = FX, '
                  'TOASavedNom is the bandwidth saved for all '
                  'vectors.')  # type: float
    TxAnt_Index = IntegerDescriptor(
        'TxAnt_Index', _required, strict=DEFAULT_STRICT,
        docstring='Indicates the Transmit Antenna pattern for data collected to form '
                  'the CPHD channel.')  # type: Union[None, int]
    RcvAnt_Index = IntegerDescriptor(
        'RcvAnt_Index', _required, strict=DEFAULT_STRICT,
        docstring='Indicates the Receive Antenna pattern for data collected to form '
                  'the CPHD channel.')  # type: Union[None, int]
    TWAnt_Index = IntegerDescriptor(
        'TWAnt_Index', _required, strict=DEFAULT_STRICT,
        docstring='Indicates the T wo-way Antenna pattern for data collected to form '
                  'the CPHD channel.')  # type: Union[None, int]

    def __init__(self, SRP_Index=None, NomTOARateSF=None, FxCtrNom=None, BWSavedNom=None,
                 TOASavedNom=None, TxAnt_Index=None, RcvAnt_Index=None, TWAnt_Index=None,
                 **kwargs):
        """

        Parameters
        ----------
        SRP_Index : int
        NomTOARateSF : float
        FxCtrNom : float
        BWSavedNom : float
        TOASavedNom : float
        TxAnt_Index : None|int
        RcvAnt_Index : None|int
        TWAnt_Index : None|int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SRP_Index = SRP_Index
        self.NomTOARateSF = NomTOARateSF
        self.FxCtrNom = FxCtrNom
        self.BWSavedNom = BWSavedNom
        self.TOASavedNom = TOASavedNom
        self.TxAnt_Index = TxAnt_Index
        self.RcvAnt_Index = RcvAnt_Index
        self.TWAnt_Index = TWAnt_Index
        super(ParametersType, self).__init__(**kwargs)


class ChannelType(Serializable):
    """
    Channel specific parameters for CPHD.
    """

    _fields = ('Parameters', )
    _required = ('Parameters', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameters'}}
    # descriptors
    Parameters = SerializableListDescriptor(
        'Parameters', ParametersType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Channel dependent parameter list.')  # type: List[ParametersType]

    def __init__(self, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        Parameters : List[ParametersType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Parameters = Parameters
        super(ChannelType, self).__init__(**kwargs)

