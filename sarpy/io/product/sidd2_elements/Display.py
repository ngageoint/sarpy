# -*- coding: utf-8 -*-
"""
The ProductDisplayType definition.
"""

from typing import Union, List

from .base import DEFAULT_STRICT
from .blocks import FilterType, NewLookupTableType
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _SerializableDescriptor, _SerializableListDescriptor, \
    _IntegerDescriptor, _FloatDescriptor, _StringDescriptor, _StringEnumDescriptor, \
    _ParametersDescriptor, ParametersCollection, SerializableArray, _SerializableArrayDescriptor


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#########
# NonInteractiveProcessing

class BandLUTType(NewLookupTableType):
    _fields = ('Predefined', 'Custom', 'k')
    _required = ('k', )
    _set_as_attribute = ('k', )
    # Descriptor
    k = _IntegerDescriptor(
        'k', _required, strict=DEFAULT_STRICT, bounds=(1, 2**32), default_value=1,
        docstring='The array index.')

    def __init__(self, Predefined=None, Custom=None, k=None, **kwargs):
        """

        Parameters
        ----------
        Predefined : sarpy.io.product.sidd2_elements.blocks.PredefinedLookupType
        Custom : sarpy.io.product.sidd2_elements.blocks.CustomLookupType
        k : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.k = k
        super(BandLUTType, self).__init__(Predefined=Predefined, Custom=Custom, **kwargs)


class BandLUTArray(SerializableArray):
    _set_size = False
    _set_index = True
    _index_var_name = 'k'


class BandEqualizationType(Serializable):
    """

    """
    _fields = ('Algorithm', 'BandLUTs')
    _required = ('Algorithm', 'BandLUTs')
    _collections_tags = {'BandLUTs': {'array': True, 'child_tag': 'BandLUT'}}
    # Descriptor
    Algorithm = _StringEnumDescriptor(
        'Algorithm', ('LUT 1D', ), _required, strict=DEFAULT_STRICT, default_value='LUT 1D',
        docstring='The algorithm type.')  # type: str
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
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Algorithm = Algorithm
        self.BandLUTs = BandLUTs
        super(BandEqualizationType, self).__init__(**kwargs)


class ProductGenerationOptionsType(Serializable):
    """

    """
    _fields = ('BandEqualization', 'ModularTransferFunctionRestoration', 'DataRemapping', 'AsymmetricPixelCorrection')
    _required = tuple()
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
        docstring='The asymmetric pixel correction.')  # type: FilterType

    def __init__(self, BandEqualization=None, ModularTransferFunctionRestoration=None, DataRemapping=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.BandEqualization = BandEqualization
        self.ModularTransferFunctionRestoration = ModularTransferFunctionRestoration
        self.DataRemapping = DataRemapping
        super(ProductGenerationOptionsType, self).__init__(**kwargs)


class RRDSType(Serializable):
    """
    RRDS type.
    """

    _fields = ('DownsamplingMethod', 'AntiAlias', 'Interpolation')
    _required = ('DownsamplingMethod', )
    # Descriptor
    DownsamplingMethod = _StringEnumDescriptor(
        'DownsamplingMethod',
        ('DECIMATE', 'MAX PIXEL', 'AVERAGE', 'NEAREST NEIGHBOR', 'BILINEAR', 'LAGRANGE'),
        _required, strict=DEFAULT_STRICT,
        docstring='Algorithm used to perform RRDS downsampling')  # type: str
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
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
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
                  'achieve the desired output product.')  # type: ProductGenerationOptionsType
    RRDS = _SerializableDescriptor(
        'RRDS', RRDSType, _required, strict=DEFAULT_STRICT,
        docstring='Creates a set of sub-sampled versions of an image to provide processing chains '
                  'with quick access to lower magnification values for faster computation '
                  'speeds and performance.')  # type: RRDSType
    band = _IntegerDescriptor(
        'band', _required, strict=DEFAULT_STRICT,
        docstring='The image band to which this applies.')  # type: int

    def __init__(self, ProductGenerationOptions=None, RRDS=None, band=1, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ProductGenerationOptions = ProductGenerationOptions
        self.RRDS = RRDS
        self.band = band
        super(NonInteractiveProcessingType, self).__init__(**kwargs)


##########
# InteractiveProcessing

class ScalingType(Serializable):
    """
    Scaling for geometric transformation
    """
    _fields = ('AntiAlias', 'Interpolation')
    _required = _fields
    # Descriptor
    AntiAlias = _SerializableDescriptor(
        'AntiAlias', FilterType, _required, strict=DEFAULT_STRICT,
        docstring='The Anti-Alias Filter used for scaling. Refer to program-specific '
                  'documentation for population guidance.')  # type: FilterType
    Interpolation = _SerializableDescriptor(
        'Interpolation', FilterType, _required, strict=DEFAULT_STRICT,
        docstring='The Interpolation Filter used for scaling. Refer to program-specific '
                  'documentation for population guidance.')  # type: FilterType

    def __init__(self, AntiAlias=None, Interpolation=None, **kwargs):
        """

        Parameters
        ----------
        AntiAlias : FilterType
        Interpolation : FilterType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.AntiAlias = AntiAlias
        self.Interpolation = Interpolation
        super(ScalingType, self).__init__(**kwargs)


class OrientationType(Serializable):
    """
    Parameters describing the default orientation of the product.
    """

    _fields = ('ShadowDirection', )
    _required = _fields
    # Descriptor
    ShadowDirection = _StringEnumDescriptor(
        'ShadowDirection', ('UP', 'DOWN', 'LEFT', 'RIGHT', 'ARBITRARY'), _required,
        strict=DEFAULT_STRICT, default_value='DOWN',
        docstring='Describes the shadow direction relative to the '
                  'pixels in the file.')  # type: str

    def __init__(self, ShadowDirection='DOWN', **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ShadowDirection = ShadowDirection
        super(OrientationType, self).__init__(**kwargs)


class GeometricTransformType(Serializable):
    """
    The geometric transform element is used to perform various geometric distortions to each
    band of image data. These distortions include image chipping, scaling, rotation, shearing, etc.
    """

    _fields = ('Scaling', 'Orientation')
    _required = _fields
    # Descriptor
    Scaling = _SerializableDescriptor(
        'Scaling', ScalingType, _required, strict=DEFAULT_STRICT,
        docstring='The scaling filters.')  # type: ScalingType
    Orientation = _SerializableDescriptor(
        'Orientation', OrientationType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters describing the default orientation of the product.')  # type: OrientationType

    def __init__(self, Scaling=None, Orientation=None, **kwargs):
        """

        Parameters
        ----------
        Scaling : ScalingType
        Orientation : OrientationType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Scaling = Scaling
        self.Orientation = Orientation
        super(GeometricTransformType, self).__init__(**kwargs)


class SharpnessEnhancementType(Serializable):
    """
    Sharpness enhancement filter parameters.
    """
    _fields = ('ModularTransferFunctionCompensation', 'ModularTransferFunctionEnhancement')
    _required = _fields
    _choice = ({'required': True, 'collection': ('ModularTransferFunctionCompensation',
                                                 'ModularTransferFunctionEnhancement')}, )
    # Descriptor
    ModularTransferFunctionCompensation = _SerializableDescriptor(
        'ModularTransferFunctionCompensation', FilterType, _required, strict=DEFAULT_STRICT,
        docstring=r'If defining a custom Filter, it must be no larger than :math:`5\times 5`.')  # type: FilterType
    ModularTransferFunctionEnhancement = _SerializableDescriptor(
        'ModularTransferFunctionEnhancement', FilterType, _required, strict=DEFAULT_STRICT,
        docstring=r'If defining a custom Filter, it must be no larger than :math:`5\times 5`.')  # type: FilterType

    def __init__(self, ModularTransferFunctionCompensation=None, ModularTransferFunctionEnhancement=None, **kwargs):
        """

        Parameters
        ----------
        ModularTransferFunctionCompensation : FilterType
        ModularTransferFunctionEnhancement : FilterType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ModularTransferFunctionCompensation = ModularTransferFunctionCompensation
        self.ModularTransferFunctionEnhancement = ModularTransferFunctionEnhancement
        super(SharpnessEnhancementType, self).__init__(**kwargs)


class ColorManagementModuleType(Serializable):
    """
    Parameters describing the Color Management Module (CMM).
    """

    _fields = ('RenderingIntent', 'SourceProfile', 'DisplayProfile', 'ICCProfile')
    _required = _fields
    # Descriptor
    RenderingIntent = _StringEnumDescriptor(
        'RenderingIntent', ('PERCEPTUAL', 'SATURATION', 'RELATIVE INTENT', 'ABSOLUTE INTENT'),
        _required, strict=DEFAULT_STRICT, default_value='PERCEPTUAL',
        docstring='The rendering intent for this color management.')  # type: str
    SourceProfile = _StringDescriptor(
        'SourceProfile', _required, strict=DEFAULT_STRICT,
        docstring='Name of sensor profile in ICC Profile database.')  # type: str
    DisplayProfile = _StringDescriptor(
        'DisplayProfile', _required, strict=DEFAULT_STRICT,
        docstring='Name of display profile in ICC Profile database.')  # type: str
    ICCProfile = _StringDescriptor(
        'ICCProfile', _required, strict=DEFAULT_STRICT,
        docstring='Valid ICC profile signature.')  # type: str

    def __init__(self, RenderingIntent='PERCEPTUAL', SourceProfile=None,
                 DisplayProfile=None, ICCProfile=None, **kwargs):
        """

        Parameters
        ----------
        RenderingIntent : str
        SourceProfile : str
        DisplayProfile : str
        ICCProfile : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RenderingIntent = RenderingIntent
        self.SourceProfile = SourceProfile
        self.DisplayProfile = DisplayProfile
        self.ICCProfile = ICCProfile
        super(ColorManagementModuleType, self).__init__(**kwargs)


class ColorSpaceTransformType(Serializable):
    """
    Parameters describing any color transformation.
    """
    _fields = ('ColorManagementModule', )
    _required = _fields
    # Descriptor
    ColorManagementModule = _SerializableDescriptor(
        'ColorManagementModule', ColorManagementModuleType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters describing the Color Management Module (CMM).')  # type: ColorManagementModuleType

    def __init__(self, ColorManagementModule=None, **kwargs):
        """

        Parameters
        ----------
        ColorManagementModule : ColorManagementModuleType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ColorManagementModule = ColorManagementModule
        super(ColorSpaceTransformType, self).__init__(**kwargs)


class DRAParametersType(Serializable):
    """
    The basic dynamic range adjustment parameters.
    """
    _fields = ('Pmin', 'Pmax', 'EminModifier', 'EmaxModifier')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    # Descriptor
    Pmin = _FloatDescriptor(
        'Pmin', _required, strict=DEFAULT_STRICT, bounds=(0, 1),
        docstring='DRA clip low point. This is the cumulative histogram percentage value '
                  'that defines the lower end-point of the dynamic range to be displayed.')  # type: float
    Pmax = _FloatDescriptor(
        'Pmax', _required, strict=DEFAULT_STRICT, bounds=(0, 1),
        docstring='DRA clip high point. This is the cumulative histogram percentage value '
                  'that defines the upper end-point of the dynamic range to be displayed.')  # type: float
    EminModifier = _FloatDescriptor(
        'EminModifier', _required, strict=DEFAULT_STRICT, bounds=(0, 1),
        docstring='The pixel value corresponding to the Pmin percentage point in the '
                  'image histogram.')  # type: float
    EmaxModifier = _FloatDescriptor(
        'EmaxModifier', _required, strict=DEFAULT_STRICT, bounds=(0, 1),
        docstring='The pixel value corresponding to the Pmax percentage point in the '
                  'image histogram.')  # type: float

    def __init__(self, Pmin=None, Pmax=None, EminModifier=None, EmaxModifier=None, **kwargs):
        """

        Parameters
        ----------
        Pmin : float
        Pmax : float
        EminModifier : float
        EmaxModifier : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.EminModifier = EminModifier
        self.EmaxModifier = EmaxModifier
        super(DRAParametersType, self).__init__(**kwargs)


class DRAOverridesType(Serializable):
    """
    The dynamic range adjustment overrides.
    """
    _fields = ('Subtractor', 'Multiplier')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    # Descriptor
    Subtractor = _FloatDescriptor(
        'Subtractor', _required, strict=DEFAULT_STRICT, bounds=(0, 2047),
        docstring='Subtractor value used to reduce haze in the image.')  # type: float
    Multiplier = _FloatDescriptor(
        'Multiplier', _required, strict=DEFAULT_STRICT, bounds=(0, 2047),
        docstring='Multiplier value used to reduce haze in the image.')  # type: float

    def __init__(self, Subtractor=None, Multiplier=None, **kwargs):
        """

        Parameters
        ----------
        Subtractor : float
        Multiplier : float
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Subtractor = Subtractor
        self.Multiplier = Multiplier
        super(DRAOverridesType, self).__init__(**kwargs)


class DynamicRangeAdjustmentType(Serializable):
    """
    The dynamic range adjustment (DRA) parameters.
    """
    _fields = ('AlgorithmType', 'BandStatsSource', 'DRAParameters', 'DRAOverrides')
    _required = ('AlgorithmType', 'BandStatsSource', )
    # Descriptor
    AlgorithmType = _StringEnumDescriptor(
        'AlgorithmType', ('AUTO', 'MANUAL', 'NONE'), _required, strict=DEFAULT_STRICT, default_value='NONE',
        docstring='Algorithm used for dynamic range adjustment.')  # type: str
    BandStatsSource = _IntegerDescriptor(
        'BandStatsSource', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: int
    DRAParameters = _SerializableDescriptor(
        'DRAParameters', DRAParametersType, _required, strict=DEFAULT_STRICT,
        docstring='The dynamic range adjustment parameters.')  # type: DRAParametersType
    DRAOverrides = _SerializableDescriptor(
        'DRAOverrides', DRAOverridesType, _required, strict=DEFAULT_STRICT,
        docstring='The dynamic range adjustment overrides.')  # type: DRAOverridesType

    def __init__(self, AlgorithmType='NONE', BandStatsSource=None, DRAParameters=None, DRAOverrides=None, **kwargs):
        """

        Parameters
        ----------
        AlgorithmType : str
        BandStatsSource : int
        DRAParameters : DRAParametersType
        DRAOverrides : DRAOverridesType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.AlgorithmType = AlgorithmType
        self.BandStatsSource = BandStatsSource
        self.DRAParameters = DRAParameters
        self.DRAOverrides = DRAOverrides
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
        docstring="The 1-D LUT element uses one or more 1-D LUTs to stretch or compress tone data "
                  "in valorous regions within a digital image's dynamic range. 1-D LUT can be "
                  "implemented using a Tonal Transfer Curve (TTC). There are 12 families of TTCs "
                  "- Range = [0, 11]. There are 64 members for each "
                  "family - Range=[0, 63].")  # type: NewLookupTableType
    band = _IntegerDescriptor(
        'band', _required, strict=DEFAULT_STRICT,
        docstring='The image band to which this applies.')

    def __init__(self, GeometricTransform=None, SharpnessEnhancement=None,
                 ColorSpaceTransform=None, DynamicRangeAdjustment=None, band=1, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.GeometricTransform = GeometricTransform
        self.SharpnessEnhancement = SharpnessEnhancement
        self.ColorSpaceTransform = ColorSpaceTransform
        self.DynamicRangeAdjustment = DynamicRangeAdjustment
        self.band = band
        super(InteractiveProcessingType, self).__init__(**kwargs)


##########

class ProductDisplayType(Serializable):
    """

    """
    _fields = (
        'PixelType', 'NumBands', 'DefaultBandDisplay', 'NonInteractiveProcessing',
        'InteractiveProcessing', 'DisplayExtensions')
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
                  'Design and Exploitation document.')  # type: str
    NumBands = _IntegerDescriptor(
        'NumBands', _required, strict=DEFAULT_STRICT,
        docstring='Number of bands contained in the image. Populate with the number of bands '
                  'present after remapping. For example an 8-bit RGB image (RGBLU), this will '
                  'be 3.')  # type: int
    DefaultBandDisplay = _IntegerDescriptor(
        'DefaultBandDisplay', _required, strict=DEFAULT_STRICT,
        docstring='Indicates which band to display by default. '
                  'Valid range = 1 to NumBands.')  # type: int
    NonInteractiveProcessing = _SerializableListDescriptor(
        'NonInteractiveProcessing', NonInteractiveProcessingType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Non-interactive processing details.')  # type: List[NonInteractiveProcessingType]
    InteractiveProcessing = _SerializableListDescriptor(
        'InteractiveProcessing', InteractiveProcessingType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Interactive processing details.')  # type: List[InteractiveProcessingType]
    DisplayExtensions = _ParametersDescriptor(
        'DisplayExtensions', _collections_tags, _required, strict=DEFAULT_STRICT,
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
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PixelType = PixelType
        self.NumBands = NumBands
        self.DefaultBandDisplay = DefaultBandDisplay
        self.NonInteractiveProcessing = NonInteractiveProcessing
        self.InteractiveProcessing = InteractiveProcessing
        self.DisplayExtensions = DisplayExtensions
        super(ProductDisplayType, self).__init__(**kwargs)
