"""
The GridType definition.
"""

import logging
from typing import List

import numpy

from ._base import Serializable, DEFAULT_STRICT, \
    _StringDescriptor, _StringEnumDescriptor, _FloatDescriptor, _FloatArrayDescriptor, _IntegerEnumDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from ._blocks import ParameterType, XYZType, Poly2DType

__classification__ = "UNCLASSIFIED"


class WgtTypeType(Serializable):
    """The weight type parameters of the direction parameters"""
    _fields = ('WindowName', 'Parameters')
    _required = ('WindowName',)
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    WindowName = _StringDescriptor(
        'WindowName', _required, strict=DEFAULT_STRICT,
        docstring='Type of aperture weighting applied in the spatial frequency domain (Krow) to yield '
                  'the impulse response in the row direction. '
                  '*Example values - "UNIFORM", "TAYLOR", "UNKNOWN", "HAMMING"*')  # type: str
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, required=_required, strict=DEFAULT_STRICT,
        docstring='Free form parameters list.')  # type: List[ParameterType]

    @classmethod
    def from_node(cls, node, kwargs=None):
        if node.find('WindowName') is None:
            # SICD 0.4 standard compliance, this could just be a space delimited string of the form
            #   "<WindowName> <name1>=<value1> <name2>=<value2> ..."
            if kwargs is None:
                kwargs = {}
            values = node.text.strip().split()
            kwargs['WindowName'] = values[0]
            params = []
            for entry in values[1:]:
                try:
                    name, val = entry.split('=')
                    params.append(ParameterType(name=name, value=val))
                except ValueError:
                    continue
            kwargs['Parameters'] = params
            return cls.from_dict(kwargs)
        else:
            return super(WgtTypeType, cls).from_node(node, kwargs)


class DirParamType(Serializable):
    """The direction parameters container"""
    _fields = (
        'UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly',
        'WgtType', 'WgtFunct')
    _required = ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2')
    _numeric_format = {
        'SS': '0.8f', 'ImpRespWid': '0.8f', 'Sgn': '+d', 'ImpRespBW': '0.8f', 'KCtr': '0.8f',
        'DeltaK1': '0.8f', 'DeltaK2': '0.8f'}
    _collections_tags = {'WgtFunct': {'array': True, 'child_tag': 'Wgt'}}
    # descriptors
    UVectECF = _SerializableDescriptor(
        'UVectECF', XYZType, required=_required, strict=DEFAULT_STRICT,
        docstring='Unit vector in the increasing (row/col) direction (ECF) at the SCP pixel.')  # type: XYZType
    SS = _FloatDescriptor(
        'SS', _required, strict=DEFAULT_STRICT,
        docstring='Sample spacing in the increasing (row/col) direction. Precise spacing at the SCP.')  # type: float
    ImpRespWid = _FloatDescriptor(
        'ImpRespWid', _required, strict=DEFAULT_STRICT,
        docstring='Half power impulse response width in the increasing (row/col) direction. '
                  'Measured at the scene center point.')  # type: float
    Sgn = _IntegerEnumDescriptor(
        'Sgn', (1, -1), _required, strict=DEFAULT_STRICT,
        docstring='Sign for exponent in the DFT to transform the (row/col) dimension to '
                  'spatial frequency dimension.')  # type: int
    ImpRespBW = _FloatDescriptor(
        'ImpRespBW', _required, strict=DEFAULT_STRICT,
        docstring='Spatial bandwidth in (row/col) used to form the impulse response in the (row/col) direction. '
                  'Measured at the center of support for the SCP.')  # type: float
    KCtr = _FloatDescriptor(
        'KCtr', _required, strict=DEFAULT_STRICT,
        docstring='Center spatial frequency in the given dimension. '
                  'Corresponds to the zero frequency of the DFT in the given (row/col) direction.')  # type: float
    DeltaK1 = _FloatDescriptor(
        'DeltaK1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum (row/col) offset from KCtr of the spatial frequency support for the image.')  # type: float
    DeltaK2 = _FloatDescriptor(
        'DeltaK2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum (row/col) offset from KCtr of the spatial frequency support for the image.')  # type: float
    DeltaKCOAPoly = _SerializableDescriptor(
        'DeltaKCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Offset from KCtr of the center of support in the given (row/col) spatial frequency. '
                  'The polynomial is a function of image given (row/col) coordinate (variable 1) and '
                  'column coordinate (variable 2).')  # type: Poly2DType
    WgtType = _SerializableDescriptor(
        'WgtType', WgtTypeType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters describing aperture weighting type applied in the spatial frequency domain '
                  'to yield the impulse response in the given(row/col) direction.')  # type: WgtTypeType
    WgtFunct = _FloatArrayDescriptor(
        'WgtFunct', _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='Sampled aperture amplitude weighting function (array) applied to form the SCP impulse '
                  'response in the given (row/col) direction.')  # type: numpy.ndarray

    def _basic_validity_check(self):
        condition = super(DirParamType, self)._basic_validity_check()
        if (self.WgtFunct is not None) and (self.WgtFunct.size < 2):
            logging.error(
                'The WgtFunct array has been defined in DirParamType, but there are fewer than 2 entries.')
            condition = False
        return condition


class GridType(Serializable):
    """Collection grid details container"""
    _fields = ('ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col')
    _required = _fields
    _IMAGE_PLANE_VALUES = ('SLANT', 'GROUND', 'OTHER')
    _TYPE_VALUES = ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    # descriptors
    ImagePlane = _StringEnumDescriptor(
        'ImagePlane', _IMAGE_PLANE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="Defines the type of image plane that the best describes the sample grid. Precise plane "
                  "defined by Row Direction and Column Direction unit vectors.")  # type: str
    Type = _StringEnumDescriptor(
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

        * `XCTYAT` – Orthogonal slant plane grid with X oriented cross track.

        * `PLANE` – Arbitrary plane with orientation other than the specific `XRGYCR` or `XCTYAT`.
        \n\n
        """)  # type: str
    TimeCOAPoly = _SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring="*Time of Center Of Aperture* as a polynomial function of image coordinates. "
                  "The polynomial is a function of image row coordinate (variable 1) and column coordinate "
                  "(variable 2).")  # type: Poly2DType
    Row = _SerializableDescriptor(
        'Row', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Row direction parameters.")  # type: DirParamType
    Col = _SerializableDescriptor(
        'Col', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Column direction parameters.")  # type: DirParamType
