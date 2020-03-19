# -*- coding: utf-8 -*-
"""
The ProductDisplayType definition.
"""

from typing import Union, List

from .base import Serializable, _SerializableDescriptor, _SerializableListDescriptor, \
    _IntegerDescriptor, _IntegerListDescriptor, _StringDescriptor, _StringEnumDescriptor, \
    _ParametersDescriptor, DEFAULT_STRICT, ParametersCollection, SerializableArray, _SerializableArrayDescriptor
from .blocks import FilterType, NewLookupTableType


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#########
# NonInteractiveProcessing

class BandLUTType(NewLookupTableType):
    # TODO: put in the fields
    _fields = ('...', 'k')
    _required = ('...', 'k')
    # Descriptor
    k = _IntegerDescriptor(
        'k', _required, strict=DEFAULT_STRICT, bounds=(1, 2**32), default_value=1,
        docstring='The array index.')

    def __init__(self, k=None, **kwargs):
        # TODO: the args
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        k = k
        super(BandLUTType, self).__init__(**kwargs)


class BandLUTArray(SerializableArray):
    _set_size = False
    _set_index = True
    _index_var_name = 'k'


class BandEqualizationType(Serializable):
    """

    """
    _fields = ('Algorithm', 'BandLUTs')
    _required = ('Algorithm', 'BandLUTs')
    _collections_tags = {'BandLUTs': {'array': False, 'child_tag': 'BandLUT'}}
    # Descriptor
    Algorithm = _StringEnumDescriptor(
        'Algorithm', ('LUT 1D', ), _required, strict=DEFAULT_STRICT, default_value='LUT 1D',
        docstring='')
    BandLUTs = _SerializableArrayDescriptor(
        'BandLUTs', BandLUTType, _collections_tags, _required, strict=DEFAULT_STRICT, array_extension=BandLUTArray,
        docstring='')  # type: Union[BandLUTArray, List[BandLUTType]]

    def __init__(self, Algorithm='LUT 1D', BandLUTs=None, **kwargs):
        """

        Parameters
        ----------
        Algorithm : str
            `LUT 1D` is currently the only allowed value.
        BandLUTs : BandLUTArray|List[BandLUTType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.Algorithm = Algorithm
        self.BandLUTs = BandLUTs
        super(BandEqualizationType, self).__init__(**kwargs)


class ProductGenerationOptionsType(Serializable):
    """

    """
    _fields = ('BandEqualization', 'ModularTransferFunctionRestoration', 'DataRemapping', 'AsymmetricPixelCorrection')
    _required = ('DataRemapping', )
    # Descriptor
    BandEqualization = _SerializableDescriptor(
        'BandEqualization', BandEqualizationType, _required, strict=DEFAULT_STRICT,
        docstring='Band equalization ensures that real-world neutral colors have equal digital count values '
                  '(i.e. are represented as neutral colors) across the dynamic range '
                  'of the imaged scene.')  # type: BandEqualizationType
    ModularTransferFunctionRestoration = _SerializableDescriptor(
        'ModularTransferFunctionRestoration', FilterType, _required, strict=DEFAULT_STRICT,
        docstring=r'If present, the filter must not exceed :math:`15 \times 15`.')  # type: FilterType
    DataRemapping = _SerializableDescriptor(
        'DataRemapping', NewLookupTableType, _required, strict=DEFAULT_STRICT,
        docstring='Data remapping refers to the specific need to convert the data of incoming, expanded or '
                  'uncompressed image band data to non-mapped image data.')  # type: NewLookupTableType
    AsymmetricPixelCorrection = _SerializableDescriptor(
        'AsymmetricPixelCorrection', FilterType, _required, strict=DEFAULT_STRICT,
        docstring='The asymmetric pixel correction.')

    def __init__(self, BandEqualization=None, ModularTransferFunctionRestoration=None, DataRemapping=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.BandEqualization = BandEqualization
        self.ModularTransferFunctionRestoration = ModularTransferFunctionRestoration
        self.DataRemapping = DataRemapping
        super(ProductGenerationOptionsType, self).__init__(**kwargs)


class RRDSType(Serializable):
    """
    RRDS type?
    """
    # TODO: improve this docstring
    _fields = ('DownsamplingMethod', 'AntiAlias', 'Interpolation')
    _required = ('DownsamplingMethod', )
    # Descriptor
    DownsamplingMethod = _StringEnumDescriptor(
        'DownsamplingMethod',
        ('DECIMATE', 'MAX PIXEL', 'AVERAGE', 'NEAREST NEIGHBOR', 'BILINEAR', 'LAGRANGE'),
        _required, strict=DEFAULT_STRICT,
        docstring='Algorithm used to perform RRDS downsampling')
    AntiAlias = _SerializableDescriptor(
        'AntiAlias', FilterType, _required, strict=DEFAULT_STRICT,
        docstring='The anti-aliasing filter. Should only be included if '
                  '`DownsamplingMethod= "DECIMATE"`')  # type: FilterType
    Interpolation = _SerializableDescriptor(
        'Interpolation', FilterType, _required, strict=DEFAULT_STRICT,
        docstring='The interpolation filter. Should only be included if '
                  '`DownsamplingMethod= "DECIMATE"`')  # type: FilterType

    def __init__(self, DownsamplingMethod=None, AntiAlias=None, Interpolation=None, **kwargs):
        """

        Parameters
        ----------
        DownsamplingMethod : str
        AntiAlias : None|FilterType
        Interpolation : None|FilterType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.DownsamplingMethod = DownsamplingMethod
        self.AntiAlias = AntiAlias
        self.Interpolation = Interpolation
        super(RRDSType, self).__init__(**kwargs)


class NonInteractiveProcessingType(Serializable):
    """
    The non-interactive processing information.
    """

    _fields = ('ProductGenerationOptions', 'RRDS', 'band')
    _required = ('ProductGenerationOptions', 'RRDS', 'band')
    _set_as_attribute = ('band', )
    # Descriptor
    ProductGenerationOptions = _SerializableDescriptor(
        'ProductGenerationOptions', ProductGenerationOptionsType, _required, strict=DEFAULT_STRICT,
        docstring='Performs several key actions on an image to prepare it for necessary additional processing to '
                  'achieve the desired output product.') # type: ProductGenerationOptionsType
    RRDS = _SerializableDescriptor(
        'RRDS', RRDSType, _required, strict=DEFAULT_STRICT,
        docstring='Creates a set of sub-sampled versions of an image to provide processing chains '
                  'with quick access to lower mangification values for faster computation s'
                  'peeds and performance.')  # type: RRDSType
    band = _IntegerDescriptor(
        'band', _required, strict=DEFAULT_STRICT,
        docstring='The immage band to which this applies.')

    def __init__(self, ProductGenerationOptions=None, RRDS=None, band=1, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.ProductGenerationOptions = ProductGenerationOptions
        self.RRDS = RRDS
        self.band = band
        super(NonInteractiveProcessingType, self).__init__(**kwargs)


##########
# InteractiveProcessing

class GeometricTransformType(Serializable):
    """

    """
    _fields = ()
    _required = ()
    # Descriptor

    # TODO:

    def __init__(self, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(GeometricTransformType, self).__init__(**kwargs)


class SharpnessEnhancementType(Serializable):
    """

    """
    _fields = ()
    _required = ()
    # Descriptor

    # TODO:

    def __init__(self, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(SharpnessEnhancementType, self).__init__(**kwargs)


class ColorSpaceTransformType(Serializable):
    """

    """
    _fields = ()
    _required = ()
    # Descriptor

    # TODO:

    def __init__(self, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(ColorSpaceTransformType, self).__init__(**kwargs)


class DynamicRangeAdjustmentType(Serializable):
    """

    """
    _fields = ()
    _required = ()
    # Descriptor

    # TODO:

    def __init__(self, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(DynamicRangeAdjustmentType, self).__init__(**kwargs)


class InteractiveProcessingType(Serializable):
    """
    The interactive processing information.
    """
    _fields = (
        'GeometricTransform', 'SharpnessEnhancement', 'ColorSpaceTransform',
        'DynamicRangeAdjustment', 'TonalTransferCurve', 'band')
    _required = (
        'GeometricTransform', 'SharpnessEnhancement', 'DynamicRangeAdjustment', 'band')
    _set_as_attribute = ('band', )
    # Descriptor
    GeometricTransform = _SerializableDescriptor(
        'GeometricTransform', GeometricTransformType, _required, strict=DEFAULT_STRICT,
        docstring='The geometric transform element is used to perform various geometric distortions '
                  'to each band of image data. These distortions include image '
                  'chipping, scaling, rotation, shearing, etc.')  # type: GeometricTransformType
    SharpnessEnhancement = _SerializableDescriptor(
        'SharpnessEnhancement', SharpnessEnhancementType, _required, strict=DEFAULT_STRICT,
        docstring='Sharpness enhancement.')  # type: SharpnessEnhancementType
    ColorSpaceTransform = _SerializableDescriptor(
        'ColorSpaceTransform', ColorSpaceTransformType, _required, strict=DEFAULT_STRICT,
        docstring='Color space transform.')  # type: ColorSpaceTransformType
    DynamicRangeAdjustment = _SerializableDescriptor(
        'DynamicRangeAdjustment', DynamicRangeAdjustmentType, _required, strict=DEFAULT_STRICT,
        docstring='Specifies the recommended ELT DRA overrides.')  # type: DynamicRangeAdjustmentType
    TonalTransferCurve = _SerializableDescriptor(
        'TonalTransferCurve', NewLookupTableType, _required, strict=DEFAULT_STRICT,
        docstring="The 1-D LUT element uses one or more 1-D LUTs to stretch or compress tome data "
                  "in valorous regions within a digital image's dynamic range. 1-D LUT can be "
                  "implemented using a Tonal Transfer Curve (TTC). There are 12 families of TTCs "
                  "- Range = [0, 11]. There are 64 members for each "
                  "family - Range=[0, 63].")  # type: NewLookupTableType
    band = _IntegerDescriptor(
        'band', _required, strict=DEFAULT_STRICT,
        docstring='The immage band to which this applies.')

    def __init__(self, GeometricTransform=None, SharpnessEnhancement=None,
                 ColorSpaceTransform=None, DynamicRangeAdjustment=None, band=1, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.GeometricTransform = GeometricTransform
        self.SharpnessEnhancement = SharpnessEnhancement
        self.ColorSpaceTransform = ColorSpaceTransform
        self.DynamicRangeAdjustment = DynamicRangeAdjustment
        self.band = band
        super(InteractiveProcessingType, self).__init__(**kwargs)


class ProductDisplayType(Serializable):
    """

    """
    _fields = (
        'PixelType', 'NumBands', 'DefaultBandDisplay', 'NonInteractiveProcessing',
        'InteractiveProcessing', 'DisplayExtension')
    _required = (
        'PixelType', 'NumBands', 'NonInteractiveProcessing', 'InteractiveProcessing')
    _collections_tags = {
        'NonInteractiveProcessing': {'array': False, 'child_tag': 'NonInteractiveProcessing'},
        'InteractiveProcessing': {'array': False, 'child_tag': 'InteractiveProcessing'},
        'DisplayExtensions': {'array': False, 'child_tag': 'DisplayExtension'}}

    # Descriptors
    PixelType = _StringEnumDescriptor(
        'PixelType', ('MONO8I', 'MONO8LU', 'MONO16I', 'RGBL8U', 'RGB24I'), _required, strict=DEFAULT_STRICT,
        docstring='Enumeration of the pixel type. Definition in '
                  'Design and Exploitation document.')  # type: PixelTypeType
    NumBands = _IntegerDescriptor(
        'NumBands', _required, strict=DEFAULT_STRICT,
        docstring='Number of bands contained in the image. Populate with the number of bands '
                  'present after remapping. For example an 8-bit RGB image (RGBLU) this should '
                  'be populated with 3.')  # type: int
    DefaultBandDisplay = _IntegerDescriptor(
        'DefaultBandDisplay', _required, strict=DEFAULT_STRICT,
        docstring='Indicates which band to display by default. '
                  'Valid range = 1 to NumBands.')  # type: int
    NonInteractiveProcessing = _SerializableListDescriptor(
        'NonInteractiveProcessing', NonInteractiveProcessingType, _required, strict=DEFAULT_STRICT,
        docstring='Non-interactive processing details.')  # type: List[NonInteractiveProcessingType]
    InteractiveProcessing = _SerializableListDescriptor(
        'InteractiveProcessing', InteractiveProcessingType, _required, strict=DEFAULT_STRICT,
        docstring='Interactive processing details.')  # type: List[InteractiveProcessingType]
    DisplayExtensions = _ParametersDescriptor(
        'DisplayExtensions', _collections_tags, required=_required, strict=DEFAULT_STRICT,
        docstring='Optional extensible parameters used to support profile-specific needs related to '
                  'product display. Predefined filter types.')  # type: ParametersCollection

    def __init__(self, PixelType=None, NumBands=1, DefaultBandDisplay=None,
                 NonInteractiveProcessing=None, InteractiveProcessing=None, DisplayExtensions=None, **kwargs):
        """

        Parameters
        ----------
        PixelType : PixelTypeType
        NumBands : int
        DefaultBandDisplay : int|None
        NonInteractiveProcessing : List[NonInteractiveProcessingType]
        InteractiveProcessing : List[InteractiveProcessingType]
        DisplayExtensions : ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.PixelType = PixelType
        self.NumBands = NumBands
        self.DefaultBandDisplay = DefaultBandDisplay
        self.NonInteractiveProcessing = NonInteractiveProcessing
        self.InteractiveProcessing = InteractiveProcessing
        self.DisplayExtensions = DisplayExtensions
        super(ProductDisplayType, self).__init__(**kwargs)
