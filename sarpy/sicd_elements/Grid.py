"""
The GridType definition.
"""

import logging
from typing import List

import numpy

from scipy.optimize import newton


from ._base import Serializable, DEFAULT_STRICT, \
    _StringDescriptor, _StringEnumDescriptor, _FloatDescriptor, _FloatArrayDescriptor, _IntegerEnumDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from ._blocks import ParameterType, XYZType, Poly2DType

__classification__ = "UNCLASSIFIED"

# module variable
DEFAULT_WEIGHT_SIZE = 512
"""
int: the default size when generating WgtFunct from a named WgtType.
"""


def _hamming_ipr(x, a):  # TODO: what is a? This name stinks.
    """Helper method"""
    return a*numpy.sinc(x) + (1-a)*(numpy.sinc(x-1) + numpy.sinc(x+1)) - a/numpy.sqrt(2)


def _raised_cos(n, coef):  # TODO: This name also stinks
    """Helper method"""
    weights = numpy.zeros((n,), dtype=numpy.float64)
    if (n % 2) == 0:
        theta = 2*numpy.pi*numpy.arange(n/2)/(n-1)  # TODO: why n-1?
        weights[:n] = (coef - (1-coef)*numpy.cos(theta))
        weights[:n] = weights[:n]
    else:
        theta = 2*numpy.pi*numpy.arange((n+1)/2)/(n-1)
        weights[:n+1] = (coef - (1-coef)*numpy.cos(theta))
        weights[:n+1] = weights[:n]
    return weights


def _taylor_win(n, sidelobes=4, max_sidelobe_level=-30, normalize=True):  # TODO: This name stinks
    """
    Generates a Taylor Window as an array of a given size.

    *From mathworks documentation* - Taylor windows are similar to Chebyshev windows. A Chebyshev window
    has the narrowest possible mainlobe for a specified sidelobe level, but a Taylor window allows you to
    make tradeoffs between the mainlobe width and the sidelobe level. The Taylor distribution avoids edge
    discontinuities, so Taylor window sidelobes decrease monotonically. Taylor window coefficients are not
    normalized. Taylor windows are typically used in radar applications, such as weighting synthetic aperture
    radar images and antenna design.

    Parameters
    ----------
    n : int
        size of the generated window

    sidelobes : int
        Number of nearly constant-level sidelobes adjacent to the mainlobe, specified as a positive integer.
        These sidelobes are “nearly constant-level” because some decay occurs in the transition region.

    max_sidelobe_level : float
        Maximum sidelobe level relative to mainlobe peak, specified as a real negative scalar in dB. It produces
        sidelobes with peaks `max_sidelobe_level` *dB* down below the mainlobe peak.

    normalize : bool
        Should the output be normalized so that th max weight is 1?

    Returns
    -------
    numpy.ndarray
        the generated window
    """

    a = numpy.arccosh(10**(-max_sidelobe_level/20.))/numpy.pi
    # Taylor pulse widening (dilation) factor
    sp2 = (sidelobes*sidelobes)/(a*a + (sidelobes-0.5)*(sidelobes-0.5))
    # the angular space in n points
    xi = numpy.linspace(-numpy.pi, numpy.pi, n)
    # calculate the cosine weights
    out = numpy.ones((n, ), dtype=numpy.float64)  # the "constant" term
    coefs = numpy.arange(1, sidelobes)
    sgn = 1
    for m in coefs:
        coefs1 = (coefs - 0.5)
        coefs2 = coefs[coefs != m]  # TODO: why does this have a term removed?
        numerator = numpy.prod(1 - (m*m)/(sp2*(a*a + coefs1*coefs1)))
        denominator = numpy.prod(1 - (m*m)/(coefs2*coefs2))
        out += sgn*(numerator/denominator)*numpy.cos(m*xi)
        sgn *= -1
    if normalize:
        out /= numpy.amax(out)
    return out


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

    def get_parameter_value(self, param_name, default=None):
        """
        Gets the value (first value found) associated with a given parameter name *(case insensitive)*.
        Returns `None` if not found.

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

        if self.Parameters is None or len(self.Parameters) == 0:
            return default
        if param_name is None:
            # get the first value - this is dumb, but appears a use case. Leaving undocumented.
            return self.Parameters[0].value

        the_name = param_name.upper()
        for param in self.Parameters:
            if param.name.upper() == the_name:
                return param.value
        return default

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

    def define_weight_function(self, weight_size=DEFAULT_WEIGHT_SIZE):
        """
        Try to derive WgtFunct from WgtType, if necessary. This should likely be called from GridType.

        Parameters
        ----------
        weight_size : int
            the size of the WgtFunct to generate.

        Returns
        -------
        None
        """

        if self.WgtType is None or self.WgtType.WindowName is None:
            return  # nothing to be done
        # TODO: should we proceed is WgtFunct is already defined? Should we serialize WgtFunct if WgtType is given?

        window_name = self.WgtType.WindowName.upper()
        if window_name == 'HAMMING':
            # A Hamming window is defined in many places as a raised cosine of weight .54, so this is the default.
            # Some data use a generalized raised cosine and call it HAMMING, so we allow for both uses.
            try:
                coef = float(self.WgtType.get_parameter_value(None, 0.54))  # just get first parameter - name?
            except ValueError:
                coef = 0.54
            self.WgtFunct = _raised_cos(weight_size, coef)
        elif window_name == 'HANNING':
            self.WgtFunct = _raised_cos(weight_size, 0.5)
        elif window_name == 'KAISER':
            try:
                beta = float(self.WgtType.get_parameter_value(None, 14))  # just get first parameter - name?
            except ValueError:
                beta = 14.0  # default suggested in numpy.kaiser
            self.WgtFunct = numpy.kaiser(weight_size, beta)
        elif window_name == 'TAYLOR':
            sidelobes = int(self.WgtType.get_parameter_value('NBAR', 4))  # apparently the matlab argument name
            max_sidelobe_level = float(self.WgtType.get_parameter_value('SLL', -30))  # same
            if max_sidelobe_level > 0:
                max_sidelobe_level *= -1
            self.WgtFunct = _taylor_win(weight_size,
                                        sidelobes=sidelobes,
                                        max_sidelobe_level=max_sidelobe_level,
                                        normalize=True)

    def _get_broadening_factor(self):
        """
        Gets the *broadening factor*, assuming that `WgtFunct` has been properly populated.

        Returns
        -------
        float
            the broadening factor
        """
        # TODO: CLARIFY - what is this?

        if self.WgtFunct is None:
            return None  # nothing to be done

        if self.WgtType is not None and self.WgtType.WindowName is not None:
            window_name = self.WgtType.WindowName.upper()
            coef = None
            if window_name == 'UNIFORM':
                coef = 1.0
            elif window_name == 'HAMMING':
                try:
                    coef = float(self.WgtType.get_parameter_value(None, 0.54))  # just get first parameter - name?
                except ValueError:
                    coef = 0.54
            elif window_name == 'HANNING':
                coef = 0.5
            if coef is not None:
                return 2 * newton(_hamming_ipr, 0.1, args=(coef,), tol=1e-12, maxiter=10000)

        # solve for the half-power point in an oversampled impulse response
        OVERSAMPLE = 1024
        impulse_response = numpy.absolute(numpy.fft.fft(self.WgtFunct, self.WgtFunct.size*OVERSAMPLE))/numpy.sum(self.WgtFunct)
        ind = numpy.flatnonzero(impulse_response < 1/numpy.sqrt(2))[0] # find the first index with less than half power.
        # linearly interpolate between impulse_response[ind-1] and impulse_response[ind] to find 1/sqrt(2)
        v0 = impulse_response[ind-1]
        v1 = impulse_response[ind]
        return 2*(ind + (1./numpy.sqrt(2) - v0)/v1)/OVERSAMPLE

    def define_response_widths(self):
        """
        Assuming that `WgtFunct` has been properly populated, define **OTHER THINGS**. This should likely be called
        by `GridType` parent.

        Returns
        -------
        None
        """
        # TODO: fill in the docstring above.

        if self.ImpRespBW is not None and self.ImpRespWid is None:
            broadening_factor = self._get_broadening_factor()
            if broadening_factor is not None:
                self.ImpRespWid = broadening_factor/self.ImpRespBW
        elif self.ImpRespBW is None and self.ImpRespWid is not None:
            broadening_factor = self._get_broadening_factor()
            if broadening_factor is not None:
                self.ImpRespBW = broadening_factor/self.ImpRespWid

    def estimate_deltak(self, valid_vertices):
        """
        The `DeltaK1` and `DeltaK2` parameters can be estimated from `DeltaKCOAPoly`, if necessary.

        Parameters
        ----------
        valid_vertices : None|numpy.ndarray
            The array of corner points.

        Returns
        -------
        None
        """

        if self.DeltaK1 is not None and self.DeltaK2 is not None:
            return  # nothing needs to be done

        if self.ImpRespBW is None or self.SS is None:
            return  # nothing can be done

        if self.DeltaKCOAPoly is not None and valid_vertices is not None:
            deltaKs = self.DeltaKCOAPoly(valid_vertices[0, :], valid_vertices[1, :])
            min_deltak = numpy.amin(deltaKs) - 0.5*self.ImpRespBW
            max_deltak = numpy.amax(deltaKs) - 0.5*self.ImpRespBW
        else:
            min_deltak = max_deltak = -0.5*self.ImpRespBW
        # wrapped spectrum (TLM - what does that mean?)
        if (min_deltak < -0.5/self.SS) or (max_deltak > 0.5/self.SS):
            min_deltak = -0.5/self.SS
            max_deltak = -min_deltak
        self.DeltaK1 = min_deltak
        self.DeltaK2 = max_deltak

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

    def derive(self, valid_vertices):
        """
        Populate any potential derived data for GridType. Expected to be called from SICD parent.

        Parameters
        ----------
        valid_vertices : None|numpy.ndarray
            The Nx2 numpy array of image corner points.

        Returns
        -------
        None
        """

        for attribute in ['Row', 'Col']:
            value = getattr(self, attribute, None)
            if value is not None:
                value.define_weight_function()
                value.define_response_widths()
                value.estimate_deltak(valid_vertices)
