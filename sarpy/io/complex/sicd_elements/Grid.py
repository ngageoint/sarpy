"""
The GridType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging

import numpy
from numpy.linalg import norm
from scipy.constants import speed_of_light

from sarpy.processing.windows import general_hamming, taylor, kaiser, find_half_power, \
    get_hamming_broadening_factor
from sarpy.io.xml.base import Serializable, ParametersCollection, find_first_child
from sarpy.io.xml.descriptors import StringDescriptor, StringEnumDescriptor, \
    FloatDescriptor, FloatArrayDescriptor, IntegerEnumDescriptor, \
    SerializableDescriptor, UnitVectorDescriptor, ParametersDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import XYZType, Poly2DType
from .utils import _get_center_frequency


logger = logging.getLogger(__name__)


# module variable
DEFAULT_WEIGHT_SIZE = 512
"""
int: the default size when generating WgtFunct from a named WgtType.
"""


class WgtTypeType(Serializable):
    """
    The weight type parameters of the direction parameters.
    """

    _fields = ('WindowName', 'Parameters')
    _required = ('WindowName',)
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    WindowName = StringDescriptor(
        'WindowName', _required, strict=DEFAULT_STRICT,
        docstring='Type of aperture weighting applied in the spatial frequency domain (Krow) to yield '
                  'the impulse response in the row direction. '
                  '*Example values - "UNIFORM", "TAYLOR", "UNKNOWN", "HAMMING"*')  # type: str
    Parameters = ParametersDescriptor(
        'Parameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Free form parameters list.')  # type: ParametersCollection

    def __init__(self, WindowName=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        WindowName : str
        Parameters : ParametersCollection|dict
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.WindowName = WindowName
        self.Parameters = Parameters
        super(WgtTypeType, self).__init__(**kwargs)

    def get_parameter_value(self, param_name, default=None):
        """
        Gets the value (first value found) associated with a given parameter name.
        Returns `default` if not found.

        Parameters
        ----------
        param_name : str
            the parameter name for which to search.
        default : None|str
            the default value to return if lookup fails.

        Returns
        -------
        str
            the associated parameter value, or `default`.
        """

        if self.Parameters is None:
            return default
        the_dict = self.Parameters.get_collection()
        if len(the_dict) == 0:
            return default
        if param_name is None:
            # get the first value - this is dumb, but appears a use case. Leaving undocumented.
            return list(the_dict.values())[0]
        return the_dict.get(param_name, default)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        win_key = cls._child_xml_ns_key.get('WindowName', ns_key)
        win_name = find_first_child(node, 'WindowName', xml_ns, win_key)
        if win_name is None:
            # SICD 0.4 standard compliance, this could just be a space delimited string of the form
            #   "<WindowName> <name1>=<value1> <name2>=<value2> ..."
            if kwargs is None:
                kwargs = {}
            values = node.text.strip().split()
            kwargs['WindowName'] = values[0]
            params = {}
            for entry in values[1:]:
                try:
                    name, val = entry.split('=')
                    params[name] = val
                except ValueError:
                    continue
            kwargs['Parameters'] = params
            return cls.from_dict(kwargs)
        else:
            return super(WgtTypeType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)


class DirParamType(Serializable):
    """The direction parameters container"""
    _fields = (
        'UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly',
        'WgtType', 'WgtFunct')
    _required = ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2')
    _numeric_format = {
        'SS': FLOAT_FORMAT, 'ImpRespWid': FLOAT_FORMAT, 'Sgn': '+d',
        'ImpRespBW': FLOAT_FORMAT, 'KCtr': FLOAT_FORMAT,
        'DeltaK1': FLOAT_FORMAT, 'DeltaK2': FLOAT_FORMAT, 'WgtFunct': FLOAT_FORMAT}
    _collections_tags = {'WgtFunct': {'array': True, 'child_tag': 'Wgt'}}
    # descriptors
    UVectECF = UnitVectorDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Unit vector in the increasing ``(row/col)`` direction *(ECF)* at '
                  'the SCP pixel.')  # type: XYZType
    SS = FloatDescriptor(
        'SS', _required, strict=DEFAULT_STRICT,
        docstring='Sample spacing in the increasing ``(row/col)`` direction. Precise spacing '
                  'at the SCP.')  # type: float
    ImpRespWid = FloatDescriptor(
        'ImpRespWid', _required, strict=DEFAULT_STRICT,
        docstring='Half power impulse response width in the increasing ``(row/col)`` direction. '
                  'Measured at the scene center point.')  # type: float
    Sgn = IntegerEnumDescriptor(
        'Sgn', (1, -1), _required, strict=DEFAULT_STRICT,
        docstring='Sign for exponent in the DFT to transform the ``(row/col)`` dimension to '
                  'spatial frequency dimension.')  # type: int
    ImpRespBW = FloatDescriptor(
        'ImpRespBW', _required, strict=DEFAULT_STRICT,
        docstring='Spatial bandwidth in ``(row/col)`` used to form the impulse response in '
                  'the ``(row/col)`` direction. Measured at the center of '
                  'support for the SCP.')  # type: float
    KCtr = FloatDescriptor(
        'KCtr', _required, strict=DEFAULT_STRICT,
        docstring='Center spatial frequency in the given dimension. '
                  'Corresponds to the zero frequency of the DFT in the given ``(row/col)`` '
                  'direction.')  # type: float
    DeltaK1 = FloatDescriptor(
        'DeltaK1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum ``(row/col)`` offset from KCtr of the spatial frequency support '
                  'for the image.')  # type: float
    DeltaK2 = FloatDescriptor(
        'DeltaK2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum ``(row/col)`` offset from KCtr of the spatial frequency '
                  'support for the image.')  # type: float
    DeltaKCOAPoly = SerializableDescriptor(
        'DeltaKCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Offset from KCtr of the center of support in the given ``(row/col)`` spatial frequency. '
                  'The polynomial is a function of image given ``(row/col)`` coordinate ``(variable 1)`` and '
                  'column coordinate ``(variable 2)``.')  # type: Poly2DType
    WgtType = SerializableDescriptor(
        'WgtType', WgtTypeType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters describing aperture weighting type applied in the spatial frequency domain '
                  'to yield the impulse response in the given ``(row/col)`` direction.')  # type: WgtTypeType
    WgtFunct = FloatArrayDescriptor(
        'WgtFunct', _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='Sampled aperture amplitude weighting function (array) applied to form the SCP impulse '
                  'response in the given ``(row/col)`` direction.')  # type: numpy.ndarray

    def __init__(self, UVectECF=None, SS=None, ImpRespWid=None, Sgn=None, ImpRespBW=None,
                 KCtr=None, DeltaK1=None, DeltaK2=None, DeltaKCOAPoly=None,
                 WgtType=None, WgtFunct=None, **kwargs):
        """

        Parameters
        ----------
        UVectECF : XYZType|numpy.ndarray|list|tuple
        SS : float
        ImpRespWid : float
        Sgn : int
        ImpRespBW : float
        KCtr : float
        DeltaK1 : float
        DeltaK2 : float
        DeltaKCOAPoly : Poly2DType|numpy.ndarray|list|tuple
        WgtType : WgtTypeType
        WgtFunct : None|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.UVectECF = UVectECF
        self.SS = SS
        self.ImpRespWid, self.ImpRespBW = ImpRespWid, ImpRespBW
        self.Sgn = Sgn
        self.KCtr, self.DeltaK1, self.DeltaK2 = KCtr, DeltaK1, DeltaK2
        self.DeltaKCOAPoly = DeltaKCOAPoly
        self.WgtType = WgtType
        self.WgtFunct = WgtFunct
        super(DirParamType, self).__init__(**kwargs)

    def define_weight_function(self, weight_size=DEFAULT_WEIGHT_SIZE, populate=False):
        """
        Try to derive WgtFunct from WgtType, if necessary. This should likely be called from the `GridType` parent.

        Parameters
        ----------
        weight_size : int
            the size of the `WgtFunct` to generate.
        populate : bool
            Overwrite any populated WgtFunct value?

        Returns
        -------
        None|numpy.ndarray
        """

        if self.WgtType is None or self.WgtType.WindowName is None:
            return  # nothing to be done

        value = None
        window_name = self.WgtType.WindowName.upper()
        if window_name == 'HAMMING':
            # A Hamming window is defined in many places as a raised cosine of weight .54, so this is the default.
            # Some data use a generalized raised cosine and call it HAMMING, so we allow for both uses.
            try:
                # noinspection PyTypeChecker
                coef = float(self.WgtType.get_parameter_value(None, 0.54))  # just get first parameter - name?
            except ValueError:
                coef = 0.54
            value = general_hamming(weight_size, coef, sym=True)
        elif window_name == 'HANNING':
            value = general_hamming(weight_size, 0.5, sym=True)
        elif window_name == 'KAISER':
            beta = 14.0  # suggested default in literature/documentation
            try:
                # noinspection PyTypeChecker
                beta = float(self.WgtType.get_parameter_value(None, beta))  # just get first parameter - name?
            except ValueError:
                pass
            value = kaiser(weight_size, beta, sym=True)
        elif window_name == 'TAYLOR':
            # noinspection PyTypeChecker
            sidelobes = int(self.WgtType.get_parameter_value('NBAR', 4))  # apparently the matlab argument name
            # noinspection PyTypeChecker
            max_sidelobe_level = float(self.WgtType.get_parameter_value('SLL', -30))  # same
            value = taylor(weight_size, nbar=sidelobes, sll=max_sidelobe_level, norm=True, sym=True)
        elif window_name == 'UNIFORM':
            value = numpy.ones((32, ), dtype='float64')

        if self.WgtFunct is None or (populate and value is not None):
            self.WgtFunct = value
        return value

    def get_oversample_rate(self):
        """
        Gets the oversample rate. *Added in version 1.2.35.*

        Returns
        -------
        float
        """

        if self.SS is None or self.ImpRespBW is None:
            raise AttributeError('Both SS and ImpRespBW must be populated.')

        return max(1., 1./(self.SS*self.ImpRespBW))

    def _get_broadening_factor(self):
        """
        Gets the *broadening factor*, assuming that `WgtFunct` has been properly populated.

        Returns
        -------
        float
            the broadening factor
        """

        if self.WgtType is not None and self.WgtType.WindowName is not None:
            window_name = self.WgtType.WindowName.upper()
            coef = None
            if window_name == 'UNIFORM':
                coef = 1.0
            elif window_name == 'HAMMING':
                try:
                    # noinspection PyTypeChecker
                    coef = float(self.WgtType.get_parameter_value(None, 0.54))  # just get first parameter - name?
                except ValueError:
                    coef = 0.54
            elif window_name == 'HANNING':
                coef = 0.5

            if coef is not None:
                return get_hamming_broadening_factor(coef)

        return find_half_power(self.WgtFunct, oversample=1024)

    def define_response_widths(self, populate=False):
        """
        Assuming that `WgtFunct` has been properly populated, define the response widths.
        This should likely be called by `GridType` parent.

        Parameters
        ----------
        populate : bool
            Overwrite populated ImpRespWid and/or ImpRespBW?

        Returns
        -------
        None|(float, float)
            None or (ImpRespBw, ImpRespWid)
        """

        broadening_factor = self._get_broadening_factor()
        if broadening_factor is None:
            return None

        if self.ImpRespBW is not None:
            resp_width = broadening_factor/self.ImpRespBW
            if populate or self.ImpRespWid is None:
                self.ImpRespWid = resp_width
            return self.ImpRespBW, resp_width
        elif self.ImpRespWid is not None:
            resp_bw = broadening_factor/self.ImpRespWid
            if populate or self.ImpRespBW is None:
                self.ImpRespBW = resp_bw
            return resp_bw, self.ImpRespWid
        return None

    def estimate_deltak(self, x_coords, y_coords, populate=False):
        """
        The `DeltaK1` and `DeltaK2` parameters can be estimated from `DeltaKCOAPoly`, if necessary.
        This should likely be called by the `GridType` parent.

        Parameters
        ----------
        x_coords : None|numpy.ndarray
            The physical vertex coordinates to evaluate DeltaKCOAPoly
        y_coords : None|numpy.ndarray
            The physical vertex coordinates to evaluate DeltaKCOAPoly
        populate : bool
            Overwite any populated DeltaK1 and DeltaK2?

        Returns
        -------
        (float, float)
        """

        if self.ImpRespBW is None or self.SS is None:
            return  # nothing can be done

        if self.DeltaKCOAPoly is not None and x_coords is not None:
            deltaks = self.DeltaKCOAPoly(x_coords, y_coords)
            min_deltak = numpy.amin(deltaks) - 0.5*self.ImpRespBW
            max_deltak = numpy.amax(deltaks) + 0.5*self.ImpRespBW
        else:
            min_deltak = -0.5*self.ImpRespBW
            max_deltak = 0.5*self.ImpRespBW

        if (min_deltak < -0.5/abs(self.SS)) or (max_deltak > 0.5/abs(self.SS)):
            min_deltak = -0.5/abs(self.SS)
            max_deltak = -min_deltak

        if populate or (self.DeltaK1 is None or self.DeltaK2 is None):
            self.DeltaK1 = min_deltak
            self.DeltaK2 = max_deltak
        return min_deltak, max_deltak

    def check_deltak(self, x_coords, y_coords):
        """
        Checks the DeltaK values for validity.

        Parameters
        ----------
        x_coords : None|numpy.ndarray
            The physical vertex coordinates to evaluate DeltaKCOAPoly
        y_coords : None|numpy.ndarray
            The physical vertex coordinates to evaluate DeltaKCOAPoly

        Returns
        -------
        bool
        """

        out = True
        try:
            if self.DeltaK2 <= self.DeltaK1 + 1e-10:
                self.log_validity_error(
                    'DeltaK2 ({}) must be greater than DeltaK1 ({})'.format(self.DeltaK2, self.DeltaK1))
                out = False
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if self.DeltaK2 > 1./(2*self.SS) + 1e-10:
                self.log_validity_error(
                    'DeltaK2 ({}) must be <= 1/(2*SS) ({})'.format(self.DeltaK2, 1./(2*self.SS)))
                out = False
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if self.DeltaK1 < -1./(2*self.SS) - 1e-10:
                self.log_validity_error(
                    'DeltaK1 ({}) must be >= -1/(2*SS) ({})'.format(self.DeltaK1, -1./(2*self.SS)))
                out = False
        except (AttributeError, TypeError, ValueError):
            pass

        min_deltak, max_deltak = self.estimate_deltak(x_coords, y_coords, populate=False)
        try:
            if abs(self.DeltaK1/min_deltak - 1) > 1e-2:
                self.log_validity_error(
                    'The DeltaK1 value is populated as {}, but estimated to be {}'.format(self.DeltaK1, min_deltak))
                out = False
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if abs(self.DeltaK2/max_deltak - 1) > 1e-2:
                self.log_validity_error(
                    'The DeltaK2 value is populated as {}, but estimated to be {}'.format(self.DeltaK2, max_deltak))
                out = False
        except (AttributeError, TypeError, ValueError):
            pass
        return out

    def _check_bw(self):
        out = True
        try:
            if self.ImpRespBW > (self.DeltaK2 - self.DeltaK1) + 1e-10:
                self.log_validity_error(
                    'ImpRespBW ({}) must be <= DeltaK2 - DeltaK1 '
                    '({})'.format(self.ImpRespBW, self.DeltaK2 - self.DeltaK1))
                out = False
        except (AttributeError, TypeError, ValueError):
            pass
        return out

    def _check_wgt(self):
        cond = True
        if self.WgtType is None:
            return cond

        wgt_size = self.WgtFunct.size if self.WgtFunct is not None else None
        if self.WgtType.WindowName not in ['UNIFORM', 'UNKNOWN'] and (wgt_size is None or wgt_size < 2):
            self.log_validity_error(
                'Non-uniform weighting indicated, but WgtFunct not properly defined')
            return False

        if wgt_size is not None and wgt_size > 1024:
            self.log_validity_warning(
                'WgtFunct with {} elements is provided.\n'
                'The recommended number of elements is 512, '
                'and many more is likely needlessly excessive.'.format(wgt_size))

        result = self.define_response_widths(populate=False)
        if result is None:
            return cond
        resp_bw, resp_wid = result
        if abs(resp_bw/self.ImpRespBW - 1) > 1e-2:
            self.log_validity_error(
                'ImpRespBW expected as {} from weighting,\n'
                'but populated as {}'.format(resp_bw, self.ImpRespBW))
            cond = False
        if abs(resp_wid/self.ImpRespWid - 1) > 1e-2:
            self.log_validity_error(
                'ImpRespWid expected as {} from weighting,\n'
                'but populated as {}'.format(resp_wid, self.ImpRespWid))
            cond = False
        return cond

    def _basic_validity_check(self):
        condition = super(DirParamType, self)._basic_validity_check()
        if (self.WgtFunct is not None) and (self.WgtFunct.size < 2):
            self.log_validity_error(
                'The WgtFunct array has been defined in DirParamType, '
                'but there are fewer than 2 entries.')
            condition = False
        for attribute in ['SS', 'ImpRespBW', 'ImpRespWid']:
            value = getattr(self, attribute)
            if value is not None and value <= 0:
                self.log_validity_error(
                    'attribute {} is populated as {}, '
                    'but should be strictly positive.'.format(attribute, value))
                condition = False
        condition &= self._check_bw()
        condition &= self._check_wgt()
        return condition


class GridType(Serializable):
    """
    Collection grid details container
    """

    _fields = ('ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col')
    _required = _fields
    _IMAGE_PLANE_VALUES = ('SLANT', 'GROUND', 'OTHER')
    _TYPE_VALUES = ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    # descriptors
    ImagePlane = StringEnumDescriptor(
        'ImagePlane', _IMAGE_PLANE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="Defines the type of image plane that the best describes the sample grid. Precise plane "
                  "defined by Row Direction and Column Direction unit vectors.")  # type: str
    Type = StringEnumDescriptor(
        'Type', _TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Defines the type of spatial sampling grid represented by the image sample grid. 
        Row coordinate first, column coordinate second:

        * `RGAZIM` - Grid for a simple range, Doppler image. Also, the natural grid for images formed with the Polar 
          Format Algorithm.

        * `RGZERO` - A grid for images formed with the Range Migration Algorithm. Used only for imaging near closest 
          approach (i.e. near zero Doppler).

        * `XRGYCR` - Orthogonal slant plane grid oriented range and cross range relative to the ARP at a 
          reference time.

        * `XCTYAT` - Orthogonal slant plane grid with X oriented cross track.

        * `PLANE` - Arbitrary plane with orientation other than the specific `XRGYCR` or `XCTYAT`.
        \n\n
        """)  # type: str
    TimeCOAPoly = SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring="*Time of Center Of Aperture* as a polynomial function of image coordinates. "
                  "The polynomial is a function of image row coordinate ``(variable 1)`` and column coordinate "
                  "``(variable 2)``.")  # type: Poly2DType
    Row = SerializableDescriptor(
        'Row', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Row direction parameters.")  # type: DirParamType
    Col = SerializableDescriptor(
        'Col', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Column direction parameters.")  # type: DirParamType

    def __init__(self, ImagePlane=None, Type=None, TimeCOAPoly=None, Row=None, Col=None, **kwargs):
        """

        Parameters
        ----------
        ImagePlane : str
        Type : str
        TimeCOAPoly : Poly2DType|numpy.ndarray|list|tuple
        Row : DirParamType
        Col : DirParamType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ImagePlane = ImagePlane
        self.Type = Type
        self.TimeCOAPoly = TimeCOAPoly
        self.Row, self.Col = Row, Col
        super(GridType, self).__init__(**kwargs)

    def derive_direction_params(self, ImageData, populate=False):
        """
        Populate the ``Row/Col`` direction parameters from ImageData, if necessary.
        Expected to be called from SICD parent.

        Parameters
        ----------
        ImageData : sarpy.io.complex.sicd_elements.ImageData.ImageDataType
        populate : bool
            Repopulates any present values?

        Returns
        -------
        None
        """

        valid_vertices = None
        if ImageData is not None:
            valid_vertices = ImageData.get_valid_vertex_data()
            if valid_vertices is None:
                valid_vertices = ImageData.get_full_vertex_data()

        x_coords, y_coords = None, None
        if valid_vertices is not None:
            try:
                x_coords = self.Row.SS*(valid_vertices[:, 0] - (ImageData.SCPPixel.Row - ImageData.FirstRow))
                y_coords = self.Col.SS*(valid_vertices[:, 1] - (ImageData.SCPPixel.Col - ImageData.FirstCol))
            except (AttributeError, ValueError):
                pass

        for attribute in ['Row', 'Col']:
            value = getattr(self, attribute, None)
            if value is not None:
                value.define_weight_function(populate=populate)
                value.define_response_widths(populate=populate)
                value.estimate_deltak(x_coords, y_coords, populate=populate)

    def _derive_time_coa_poly(self, CollectionInfo, SCPCOA):
        """
        Expected to be called from SICD parent.

        Parameters
        ----------
        CollectionInfo : sarpy.io.complex.sicd_elements.CollectionInfo.CollectionInfoType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType

        Returns
        -------
        None
        """
        if self.TimeCOAPoly is not None:
            return  # nothing needs to be done

        try:
            if CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT':
                self.TimeCOAPoly = Poly2DType(Coefs=[[SCPCOA.SCPTime, ], ])
        except (AttributeError, ValueError):
            return

    def _derive_rg_az_comp(self, GeoData, SCPCOA, RadarCollection, ImageFormation):
        """
        Expected to be called by SICD parent.

        Parameters
        ----------
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        None
        """

        if self.Row is None:
            self.Row = DirParamType()
        if self.Col is None:
            self.Col = DirParamType()

        if self.ImagePlane is None:
            self.ImagePlane = 'SLANT'
        elif self.ImagePlane != 'SLANT':
            logger.warning(
                'The Grid.ImagePlane is set to {},\n\t'
                'but Image Formation Algorithm is RgAzComp, which requires "SLANT".\n\t'
                'Resetting.'.format(self.ImagePlane))
            self.ImagePlane = 'SLANT'

        if self.Type is None:
            self.Type = 'RGAZIM'
        elif self.Type != 'RGAZIM':
            logger.warning(
                'The Grid.Type is set to {},\n\t'
                'but Image Formation Algorithm is RgAzComp, which requires "RGAZIM".\n\t'
                'Resetting.'.format(self.Type))

        if GeoData is not None and GeoData.SCP is not None and GeoData.SCP.ECF is not None and \
                SCPCOA.ARPPos is not None and SCPCOA.ARPVel is not None:
            scp = GeoData.SCP.ECF.get_array()
            arp = SCPCOA.ARPPos.get_array()
            los = (scp - arp)
            ulos = los/norm(los)
            if self.Row.UVectECF is None:
                self.Row.UVectECF = XYZType.from_array(ulos)

            look = SCPCOA.look
            arp_vel = SCPCOA.ARPVel.get_array()
            uspz = look*numpy.cross(arp_vel, ulos)
            uspz /= norm(uspz)
            uaz = numpy.cross(uspz, ulos)
            if self.Col.UVectECF is None:
                self.Col.UVectECF = XYZType.from_array(uaz)

        center_frequency = _get_center_frequency(RadarCollection, ImageFormation)
        if center_frequency is not None:
            if self.Row.KCtr is None:
                kctr = 2*center_frequency/speed_of_light
                if self.Row.DeltaKCOAPoly is not None:  # assume it's 0 otherwise?
                    kctr -= self.Row.DeltaKCOAPoly.Coefs[0, 0]
                self.Row.KCtr = kctr
            elif self.Row.DeltaKCOAPoly is None:
                self.Row.DeltaKCOAPoly = Poly2DType(Coefs=[[2*center_frequency/speed_of_light - self.Row.KCtr, ], ])

            if self.Col.KCtr is None:
                if self.Col.DeltaKCOAPoly is not None:
                    self.Col.KCtr = -self.Col.DeltaKCOAPoly.Coefs[0, 0]
            elif self.Col.DeltaKCOAPoly is None:
                self.Col.DeltaKCOAPoly = Poly2DType(Coefs=[[-self.Col.KCtr, ], ])

    def _derive_pfa(self, GeoData, RadarCollection, ImageFormation, Position, PFA):
        """
        Expected to be called by SICD parent.

        Parameters
        ----------
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        PFA : sarpy.io.complex.sicd_elements.PFA.PFAType

        Returns
        -------
        None
        """

        if self.Type is None:
            self.Type = 'RGAZIM'  # the natural result for PFA

        if PFA is None:
            return  # nothing to be done
        if GeoData is None or GeoData.SCP is None:
            return  # nothing to be done

        scp = GeoData.SCP.ECF.get_array()

        if Position is not None and Position.ARPPoly is not None \
                and PFA.PolarAngRefTime is not None:
            polar_ref_pos = Position.ARPPoly(PFA.PolarAngRefTime)
        else:
            polar_ref_pos = scp

        if PFA.IPN is not None and PFA.FPN is not None and \
                self.Row.UVectECF is None and self.Col.UVectECF is None:
            ipn = PFA.IPN.get_array()
            fpn = PFA.FPN.get_array()

            dist = numpy.dot((scp - polar_ref_pos), ipn) / numpy.dot(fpn, ipn)
            ref_pos_ipn = polar_ref_pos + (dist * fpn)
            urg = scp - ref_pos_ipn
            urg /= norm(urg)
            uaz = numpy.cross(ipn, urg)  # already unit
            self.Row.UVectECF = XYZType.from_array(urg)
            self.Col.UVectECF = XYZType.from_array(uaz)

        if self.Col is not None and self.Col.KCtr is None:
            self.Col.KCtr = 0  # almost always 0 for PFA

        if self.Row is not None and self.Row.KCtr is None:
            center_frequency = _get_center_frequency(RadarCollection, ImageFormation)
            if PFA.Krg1 is not None and PFA.Krg2 is not None:
                self.Row.KCtr = 0.5*(PFA.Krg1 + PFA.Krg2)
            elif center_frequency is not None and PFA.SpatialFreqSFPoly is not None:
                # APPROXIMATION: may not be quite right, due to rectangular inscription loss in PFA.
                self.Row.KCtr = 2*center_frequency/speed_of_light + PFA.SpatialFreqSFPoly.Coefs[0]

    def _derive_rma(self, RMA, GeoData, RadarCollection, ImageFormation, Position):
        """

        Parameters
        ----------
        RMA : sarpy.io.complex.sicd_elements.RMA.RMAType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType

        Returns
        -------
        None
        """

        if RMA is None:
            return   # nothing can be derived

        im_type = RMA.ImageType

        if im_type is None:
            return
        if im_type == 'INCA':
            self._derive_rma_inca(RMA, GeoData, Position)
        else:
            if im_type == 'RMAT':
                self._derive_rma_rmat(RMA, GeoData, RadarCollection, ImageFormation)
            elif im_type == 'RMCR':
                self._derive_rma_rmcr(RMA, GeoData, RadarCollection, ImageFormation)

    @staticmethod
    def _derive_unit_vector_params(GeoData, RMAParam):
        """
        Helper method.

        Parameters
        ----------
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RMAParam : sarpy.io.complex.sicd_elements.RMA.RMATType|sarpy.io.complex.sicd_elements.RMA.RMCRType

        Returns
        -------
        Tuple[numpy.ndarray,...]
        """

        if GeoData is None or GeoData.SCP is None:
            return None

        scp = GeoData.SCP.ECF.get_array()
        pos_ref = RMAParam.PosRef.get_array()
        upos_ref = pos_ref / norm(pos_ref)
        vel_ref = RMAParam.VelRef.get_array()
        uvel_ref = vel_ref / norm(vel_ref)
        los = (scp - pos_ref)  # it absolutely could be that scp = pos_ref
        los_norm = norm(los)
        if los_norm < 1:
            logger.error(
                msg="Row/Col UVectECF cannot be derived from RMA,\n\t"
                    "because the Reference Position is too close (less than 1 meter) to the SCP.")
        ulos = los/los_norm
        left = numpy.cross(upos_ref, uvel_ref)
        look = numpy.sign(numpy.dot(left, ulos))
        return scp, upos_ref, uvel_ref, ulos, left, look

    def _derive_rma_rmat(self, RMA, GeoData, RadarCollection, ImageFormation):
        """

        Parameters
        ----------
        RMA : sarpy.io.complex.sicd_elements.RMA.RMAType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        None
        """

        if RMA.RMAT is None:
            return

        if self.ImagePlane is None:
            self.ImagePlane = 'SLANT'
        if self.Type is None:
            self.Type = 'XCTYAT'

        if self.Row.UVectECF is None and self.Col.UVectECF is None:
            params = self._derive_unit_vector_params(GeoData, RMA.RMAT)
            if params is not None:
                scp, upos_ref, uvel_ref, ulos, left, look = params
                uyat = -look*uvel_ref
                uspz = numpy.cross(ulos, uyat)
                uspz /= norm(uspz)
                uxct = numpy.cross(uyat, uspz)
                self.Row.UVectECF = XYZType.from_array(uxct)
                self.Col.UVectECF = XYZType.from_array(uyat)

        center_frequency = _get_center_frequency(RadarCollection, ImageFormation)
        if center_frequency is not None and RMA.RMAT.DopConeAngRef is not None:
            if self.Row.KCtr is None:
                self.Row.KCtr = (2*center_frequency/speed_of_light)*numpy.sin(numpy.deg2rad(RMA.RMAT.DopConeAngRef))
            if self.Col.KCtr is None:
                self.Col.KCtr = (2*center_frequency/speed_of_light)*numpy.cos(numpy.deg2rad(RMA.RMAT.DopConeAngRef))

    def _derive_rma_rmcr(self, RMA, GeoData, RadarCollection, ImageFormation):
        """

        Parameters
        ----------
        RMA : sarpy.io.complex.sicd_elements.RMA.RMAType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        None
        """

        if RMA.RMCR is None:
            return

        if self.ImagePlane is None:
            self.ImagePlane = 'SLANT'
        if self.Type is None:
            self.Type = 'XRGYCR'

        if self.Row.UVectECF is None and self.Col.UVectECF is None:
            params = self._derive_unit_vector_params(GeoData, RMA.RMAT)
            if params is not None:
                scp, upos_ref, uvel_ref, ulos, left, look = params
                uxrg = ulos
                uspz = look*numpy.cross(uvel_ref, uxrg)
                uspz /= norm(uspz)
                uycr = numpy.cross(uspz, uxrg)
                self.Row.UVectECF = XYZType.from_array(uxrg)
                self.Col.UVectECF = XYZType.from_array(uycr)

        center_frequency = _get_center_frequency(RadarCollection, ImageFormation)
        if center_frequency is not None:
            if self.Row.KCtr is None:
                self.Row.KCtr = 2*center_frequency/speed_of_light
            if self.Col.KCtr is None:
                self.Col.KCtr = 2*center_frequency/speed_of_light

    def _derive_rma_inca(self, RMA, GeoData, Position):
        """

        Parameters
        ----------
        RMA : sarpy.io.complex.sicd_elements.RMA.RMAType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType

        Returns
        -------
        None
        """

        if RMA.INCA is None:
            return

        if self.Type is None:
            self.Type = 'RGZERO'

        if RMA.INCA.TimeCAPoly is not None and Position is not None and Position.ARPPoly is not None and \
                self.Row.UVectECF is None and self.Col.UVectECF is None and \
                GeoData is not None and GeoData.SCP is not None:
            scp = GeoData.SCP.ECF.get_array()

            t_zero = RMA.INCA.TimeCAPoly.Coefs[0]
            ca_pos = Position.ARPPoly(t_zero)
            ca_vel = Position.ARPPoly.derivative_eval(t_zero, der_order=1)

            uca_pos = ca_pos/norm(ca_pos)
            uca_vel = ca_vel/norm(ca_vel)
            urg = (scp - ca_pos)
            urg_norm = norm(urg)
            if urg_norm > 0:
                urg /= urg_norm
                left = numpy.cross(uca_pos, uca_vel)
                look = numpy.sign(numpy.dot(left, urg))
                uspz = -look*numpy.cross(urg, uca_vel)
                uspz /= norm(uspz)
                uaz = numpy.cross(uspz, urg)
                self.Row.UVectECF = XYZType.from_array(urg)
                self.Col.UVectECF = XYZType.from_array(uaz)

        if self.Row is not None and self.Row.KCtr is None and RMA.INCA.FreqZero is not None:
            self.Row.KCtr = 2*RMA.INCA.FreqZero/speed_of_light
        if self.Col is not None and self.Col.KCtr is None:
            self.Col.KCtr = 0

    def _basic_validity_check(self):
        condition = super(GridType, self)._basic_validity_check()
        if self.Row is not None and self.Row.Sgn is not None and self.Col is not None \
                and self.Col.Sgn is not None and self.Row.Sgn != self.Col.Sgn:
            self.log_validity_warning(
                'Row.Sgn ({}) and Col.Sgn ({}) should almost certainly be the '
                'same value'.format(self.Row.Sgn, self.Col.Sgn))
        return condition

    def check_deltak(self, x_coords, y_coords):
        """
        Checks the validity of DeltaK values.

        Parameters
        ----------
        x_coords : None|numpy.ndarray
        y_coords : None|numpy.ndarray

        Returns
        -------
        bool
        """

        cond = True
        if self.Row is not None:
            cond &= self.Row.check_deltak(x_coords, y_coords)
        if self.Col is not None:
            cond &= self.Col.check_deltak(x_coords, y_coords)
        return cond

    def get_resolution_abbreviation(self):
        """
        Gets the resolution abbreviation for the suggested name.

        Returns
        -------
        str
        """

        if self.Row is None or self.Row.ImpRespWid is None or \
                self.Col is None or self.Col.ImpRespWid is None:
            return '0000'
        else:
            value = int(100*(abs(self.Row.ImpRespWid)*abs(self.Col.ImpRespWid))**0.5)
            if value > 9999:
                return '9999'
            else:
                return '{0:04d}'.format(value)

    def get_slant_plane_area(self):
        """
        Get the weighted slant plane area.

        Returns
        -------
        float
        """

        range_weight_f = azimuth_weight_f = 1.0
        if self.Row.WgtFunct is not None:
            var = numpy.var(self.Row.WgtFunct)
            mean = numpy.mean(self.Row.WgtFunct)
            range_weight_f += var/(mean*mean)
        if self.Col.WgtFunct is not None:
            var = numpy.var(self.Col.WgtFunct)
            mean = numpy.mean(self.Col.WgtFunct)
            azimuth_weight_f += var/(mean*mean)
        return (range_weight_f * azimuth_weight_f)/(self.Row.ImpRespBW*self.Col.ImpRespBW)
