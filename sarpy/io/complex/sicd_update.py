"""
**This module is a work in progress. The eventual structure of this is yet to be determined.**

Object oriented SICD structure definition. Enabling effective documentation and streamlined use of the SICD information
is the main purpose of this approach, versus the matlab struct based effort or using the Python bindings for the C++
SIX library.
"""

# TODO: MEDIUM - update this doc string

from xml.dom import minidom
import numpy
from collections import OrderedDict
from datetime import datetime, date
import logging

from typing import Union

#################
# dom helper functions, because minidom is a little weird


def _get_node_value(nod  # type: minidom.Element
                    ):
    # type: (...) -> str
    """
    XML parsing helper for extracting text value from an minidom node. No error checking performed.

    :param minidom.Element nod: xml node object
    :return str: string value of the node
    """

    return nod.firstChild.wholeText.strip()


def _create_new_node(doc,  # type: minidom.Document
                     tag,  # type: str
                     par=None  # type: Union[None, minidom.Element]
                     ):
    # type: (...) -> minidom.Element
    """
    XML minidom node creation helper function

    :param minidom.Document doc: the xml Document
    :param str tag: name for new node
    :param Union[None, minidom.Element] par: the parent element for the new element. Defaults to the document root
    element if unspecified.
    :return minidom.Element: the new node element
    """

    nod = doc.createElement(tag)
    if par is None:
        doc.documentElement.appendChild(nod)
    else:
        par.appendChild(nod)
    return nod


def _create_text_node(doc,  # type: minidom.Document
                      tag,  # type: str
                      value,  # type: str
                      par=None  # type: Union[None, minidom.Element]
                      ):
    # type: (...) -> minidom.Element
    """
    XML minidom text node creation helper function

    :param minidom.Document doc: xml Document
    :param str tag: name for new node
    :param str value: value for node contents
    :param Union[None, minidom.Element] par: parent element. Defaults to the root element if unspecified.
    :return minidom.Element: the new node element
    """

    nod = doc.createElement(tag)
    nod.appendChild(doc.createTextNode(str(value)))

    if par is None:
        doc.documentElement.appendChild(nod)
    else:
        par.appendChild(nod)
    return nod


#################
# base Serializable class.

class Serializable(object):
    """
    Basic abstract class specifying the serialization pattern. There are no clearly defined Python conventions
    for this issue. Every effort has been made to select sensible choices, but this is an individual effort.

    .. Note: All fields MUST BE LISTED in the __fields tuple. Everything listed in __required tuple will be checked
        for inclusion in __fields tuple. Note that special care must be taken to ensure compatibility of __fields tuple,
        if inheriting from an extension of this class.
    """

    __fields = ()  # collection of field names
    __required = ()  # define a non-empty tuple for required properties
    __numeric_format = {}  # define dict entries of numeric formatting for serialization
    # NB: it may be good practice to use __slots__ to further control class functionality?

    def __init__(self, **kwargs):  # type: (dict) -> None
        """
        The default constructor. For each attribute name in self.__fields(), fetches the value (or None) from
        the kwargs dictionary, and sets the class attribute value. Any fields requiring special validation should
        be implemented as properties.

        :param dict kwargs: the keyword arguments dictionary

        .. Note: The specific keywords applicable for each extension of this base class MUST be clearly specified
            in the given class. As a last resort, look at the __fields tuple specified in the given class definition.
        """
        # TODO: LOW - should we validate the keys in kwargs?

        for attribute in self.__required:
            if attribute not in self.__fields:
                raise AttributeError("Attribute {} is defined in __required, but is not listed in __fields "
                                     "for class {}".format(attribute, self.__class__.__name__))

        for attribute in self.__fields:
            try:
                setattr(self, attribute, kwargs.get(attribute, None))
            except AttributeError:
                # NB: this is included to allow for read only properties without breaking the paradigm
                pass

    def set_numeric_format(self, attribute, format_string):  # type: (str, str) -> None
        """
        Sets the numeric format string for the given attribute

        :param str attribute: attribute for which the format applies - must be in `__fields`.
        :param str format_string: format string to be applied
        """

        if attribute not in self.__fields:
            raise ValueError('attribute {} is not permitted for class {}'.format(attribute, self.__class__.__name__))
        self.__numeric_format[attribute] = format_string

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.

        :param bool recursive: should we recursively check that child are also valid?
        :return bool: condition for validity of this element

        .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
            that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then this
            will result in an infinite loop.
        """

        all_required = True
        if len(self.__required) > 0:
            for attribute in self.__required:
                present = getattr(self, attribute) is not None
                if not present:
                    logging.warning("class {} has missing required "
                                    "attribute {}".format(self.__class__.__name__, attribute))
                all_required &= present
        if not recursive:
            return all_required

        valid_children = True
        for attribute in self.__fields:
            val = getattr(self, attribute)
            if isinstance(val, Serializable):
                good = val.is_valid(recursive=recursive)
                valid_children &= good
                # any issues will be logged as discovered, but we should help with the "stack"
                if not good:
                    logging.warning("Issue discovered with {} attribute of type {} "
                                    "of class {}".format(attribute, type(val), self.__class__.__name__))
        return all_required & valid_children

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> Serializable
        """
        For XML deserialization.

        :param minidom.Element node: dom element for serialized class instance
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return Serializable: corresponding class instance
        """

        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError("Named input argument kwargs for class {} must be "
                             "dictionary instance".format(cls.__class__.__name__))

        for attribute in cls.__fields:
            if attribute in kwargs:
                continue

            pnodes = node.getElementsByTagName(attribute)
            if len(pnodes) > 0:
                kwargs[attribute] = _get_node_value(pnodes[0])
        return cls.from_dict(kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        if not self.is_valid():
            msg = "{} is not valid, and cannot be safely serialized to XML according to " \
                  "the SICD standard.".format(self.__class__.__name__)
            if strict:
                raise ValueError(msg)
            logging.warning(msg)

        if tag is None:
            tag = self.__class__.__name__
        nod = _create_new_node(doc, tag, par=par)

        for attribute in self.__fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue
            elif isinstance(value, Serializable):
                value.to_node(doc, tag=attribute, par=nod, strict=strict)
            elif isinstance(value, str):  # TODO: MEDIUM - unicode issues?
                _create_text_node(doc, attribute, value, par=nod)
            elif isinstance(value, int) or isinstance(value, float):
                if attribute in self.__numeric_format:
                    fmt_function = ('{0:' + self.__numeric_format[attribute] + '}').format
                else:
                    fmt_function = str
                _create_text_node(doc, attribute, fmt_function(value), par=nod)
            elif isinstance(value, date):
                _create_text_node(doc, attribute, value.isoformat(), par=nod)
            elif isinstance(value, datetime):
                _create_text_node(doc, attribute, value.isoformat(sep='T'), par=nod)
                # TODO: LOW - this can be error prone with the naive vs zoned issue. Discourage using this type?
            elif isinstance(value, numpy.datetime64):
                _create_text_node(doc, attribute, str(value), par=nod)
            elif isinstance(value, list) or isinstance(value, tuple):
                # TODO: what to do here
                pass
            elif isinstance(value, numpy.ndarray):
                # TODO: what to do here?
                pass
            else:
                raise ValueError('Attribute {} is of type {}, and not clearly '
                                 'serializable'.format(attribute, type(value)))
        return nod

    @classmethod
    def from_dict(cls, inputDict):  # type: (dict) -> Serializable
        """
        For json deserialization.

        :param dict inputDict: dict instance for deserialization
        :return Serializable: corresponding class instance
        """

        return cls(**inputDict)

    def to_dict(self, strict=False, exclude=()):  # type: (bool, tuple) -> OrderedDict
        """
        For json serialization.

        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: attribute names to exclude from this generic serialization. This allows for child classes
        to provide specific serialization for special properties, but still use this super method.
        :return orderedDict: dictionary representation of class instance appropriate for direct json serialization.
        Recall that any elements of `exclude` will be omitted, and should likely be included by the extension class
        implementation.
        """

        out = OrderedDict()

        # TODO: finish this as above in xml serialization
        raise NotImplementedError("Must provide a concrete implementation.")


#############
# Basic building blocks for SICD standard

class XYZType(Serializable):
    __slots__ = ('_X', '_Y', '_Z')
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    __numeric_format = {}  # TODO: desired precision (default is 0.8f)? This is usually meters?

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['X', 'Y', 'Z'], all required.
        """

        self._X = self._Y = self._Z = None
        super(XYZType, self).__init__(**kwargs)

    @property
    def X(self):  # type: () -> float
        """
        The X attribute
        :return float:
        """

        return self._X

    @X.setter
    def X(self, value):  # type: (Union[str, int, float]) -> None
        """
        The X attribute setter.

        :param Union[str, int, float] value: converts to float
        :return None:
        """
        self._X = float(value)

    @property
    def Y(self):  # type: () -> float
        """
        The Y attribute.
        :return float:
        """

        return self._Y

    @Y.setter
    def Y(self, value):  # type: (Union[str, int, float]) -> None
        """
        The Y attribute setter.

        :param Union[str, int, float] value: converts to float
        :return None:
        """

        self._Y = float(value)

    @property
    def Z(self):  # type: () -> float
        """
        The Y attribute.
        :return float:
        """

        return self._Z

    @Z.setter
    def Z(self, value):  # type: (Union[str, int, float]) -> None
        """
        The Z attribute setter.

        :param Union[str, int, float] value: converts to float
        :return None:
        """

        self._Z = float(value)

    def getArray(self, dtype=numpy.float64):  # type: (numpy.dtype) -> numpy.ndarray
        """
        Gets an [X, Y, Z] array representation of the class.

        :param numpy.dtype dtype: data type of the return
        :return numpy.ndarray:
        """

        return numpy.array([self.X, self.Y, self.Z], dtype=dtype)


class LatLonType(Serializable):
    __slots__ = ('_Lat', '_Lon')
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon'], all required.
        """

        self._Lat = self._Lon = None
        super(LatLonType, self).__init__(**kwargs)

    @property
    def Lat(self):  # type: () -> float
        """
        The Latitude attribute - in decimals degrees.
        :return float:
        """

        return self._Lat

    @Lat.setter
    def Lat(self, value):  # type: (Union[str, int, float]) -> None
        """
        The Lat setter.

        :param Union[str, int, float] value:
        :return None:
        """

        self._Lat = float(value)

    @property
    def Lon(self):  # type: () -> float
        """
        The Longitude attribute - in decimals degrees.
        :return float:
        """

        return self._Lon

    @Lon.setter
    def Lon(self, value):  # type: (Union[str, int, float]) -> None
        """
        The Lon setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._Lon = float(value)

    def getArray(self, order='LON', dtype=numpy.float64):  # type: (str, numpy.dtype) -> numpy.ndarray
        """
        Gets an array representation of the data.

        :param str order: one of ['LAT', 'LON'] first element in array order (e.g. 'Lat' corresponds to [Lat, Lon]).
        :param dtype: data type of the return
        :return numpy.ndarray:
        """

        if order.upper() == 'LAT':
            return numpy.array([self.Lat, self.Lon], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat], dtype=dtype)


class LatLonRestrictionType(LatLonType):
    __slots__ = ('_Lat', '_Lon')
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon'], all required.
        """

        super(LatLonType, self).__init__(**kwargs)

    @property
    def Lat(self):  # type: () -> float
        """
        The Latitude attribute. Guaranteed to be in the range [-90, 90].
        :return float:
        """

        # NB: you cannot redefine only the inherited setter
        return self._Lat

    @Lat.setter
    def Lat(self, value):  # type: (Union[str, int, float]) -> None
        """
        The Lat setter.
        :param Union[str, int, float] value:
        :return None:
        """

        val = float(value)
        if -90 <= val <= 90:
            self._Lat = val
        else:
            val = (val % 180)
            self._Lat = val if val <= 90 else val - 180

    @property
    def Lon(self):  # type: () -> float
        """
        The Longitude attribute. Guaranteed to be in the range [-180, 180].
        :return float:
        """

        return self._Lon

    @Lon.setter
    def Lon(self, value):  # type; (Union[str, int, float]) -> None
        """
        The Lon setter.
        :param Union[str, int, float] value:
        :return None:
        """

        val = float(value)
        if -180 <= val <= 180:
            self._Lon = val
        else:
            val = (val % 360)
            self._Lon = val if val <= 180 else val - 360


class LatLonHAEType(LatLonType):
    __slots__ = ('_Lat', '_Lon', '_HAE')
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon', 'HAE'], all required.
        """

        self._HAE = None
        super(LatLonHAEType, self).__init__(**kwargs)

    @property
    def HAE(self):  # type: () -> float
        """
        The HAE attribute, in meters.
        :return float:
        """

        return self._HAE

    @HAE.setter
    def HAE(self, value):  # type: (Union[str, int, float]) -> None
        """
        The HAE setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._HAE = float(value)

    def getArray(self, order='LON', dtype=numpy.float64):  # type: (str, numpy.dtype) -> numpy.ndarray
        """
        Gets an array representation of the data.

        :param str order: one of ['LAT', 'LON'] first element in array order Specifically, `'LAT'` corresponds to
        `[Lat, Lon, HAE]`, while `'LON'` corresponds to `[Lon, Lat, HAE]`.
        :param dtype: data type of the return
        :return numpy.ndarray:
        """

        if order.upper() == 'LAT':
            return numpy.array([self.Lat, self.Lon, self.HAE], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat, self.HAE], dtype=dtype)


class LatLonHAERestrictionType(LatLonRestrictionType):
    __slots__ = ('_Lat', '_Lon', '_HAE')
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon', 'HAE'], all required.
        """

        self._HAE = None
        super(LatLonHAERestrictionType, self).__init__(**kwargs)

    @property
    def HAE(self):  # type: () -> float
        """
        The HAE attribute, in meters.
        :return float:
        """

        return self._HAE

    @HAE.setter
    def HAE(self, value):  # type: (Union[str, int, float]) -> None
        """
        The HAE setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._HAE = float(value)

    def getArray(self, order='LON', dtype=numpy.float64):  # type: (str, numpy.dtype) -> numpy.ndarray
        """
        Gets an array representation of the data.

        :param str order: one of ['LAT', 'LON'] first element in array order Specifically, `'LAT'` corresponds to
        `[Lat, Lon, HAE]`, while `'LON'` corresponds to `[Lon, Lat, HAE]`.
        :param dtype: data type of the return
        :return numpy.ndarray:
        """

        if order.upper() == 'LAT':
            return numpy.array([self.Lat, self.Lon, self.HAE], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat, self.HAE], dtype=dtype)


class RowColType(Serializable):
    __slots__ = ('_Row', '_Col')
    __fields = ('Row', 'Col')
    __required = __fields

    def __init__(self, **kwargs):
        """
        The RowColType constructor.
        :param dict kwargs: valid keys are ['Row', 'Col'], all required.
        """
        self._Row = self._Col = None
        super(RowColType, self).__init__(**kwargs)

    @property
    def Row(self):  # type: () -> int
        """
        The Row attribute - an integer.
        :return int:
        """

        return self._Row

    @Row.setter
    def Row(self, value):  # type: (Union[str, int, float]) -> None
        """
        The Row attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._Row = int(value)

    @property
    def Col(self):  # type: () -> int
        """
        The Column attribute - an integer.
        :return int:
        """

        return self._Col

    @Col.setter
    def Col(self, value):  # type: (Union[str, int, float]) -> None
        """
        The Col attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._Col = int(value)


class RowColvertexType(RowColType):
    __slots__ = ('_Row', '_Col', '_index')
    __fields = ('Row', 'Col', 'index')
    __required = __fields

    def __init__(self, **kwargs):
        """
        The RowColvertexType constructor.
        :param dict kwargs: valid keys are ['Row', 'Col', 'index'], all required.
        """

        self._index = None
        super(RowColvertexType, self).__init__(**kwargs)

    @property
    def index(self):  # type: () -> int
        """
        The index attribute - an integer.
        :return int:
        """
        return self._index

    @index.setter
    def index(self, value):  # type: (Union[str, int, float]) -> None
        """
        The index attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._index = int(value)

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> RowColvertexType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return RowColvertexType: the deserialized class instance
        """

        if kwargs is None:
            kwargs = {}
        kwargs['index'] = node.getAttribute('index')
        return super(RowColvertexType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(RowColvertexType, self).to_node(doc, tag=tag, par=par, strict=strict, exclude=exclude)
        node.setAttribute('index', str(self.index))
        return node


class PolyCoef1DType(Serializable):
    __slots__ = ('_value', '_exponent1')
    __fields = ('value', 'exponent1')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        """
        This represents a single monomial term of the form - `value * x^{exponent1}`
        :param dict kwargs: valid keys are ['value', 'exponent1'], all required.
        """

        self._value = self._exponent1 = None
        super(PolyCoef1DType, self).__init__(**kwargs)

    @property
    def value(self):  # type: () -> float
        """
        The value attribute.
        :return float:
        """

        return self._value

    @value.setter
    def value(self, val):  # type: (Union[str, int, float]) -> None
        """
        The value attribute setter.
        :param Union[str, int, float] val:
        :return None:
        """

        self._value = float(val)

    @property
    def exponent1(self):  # type: () -> int
        """
        The exponent1 attribute.
        :return:
        """
        return self._exponent1

    @exponent1.setter
    def exponent1(self, value):  # type: (Union[str, int, float]) -> None
        """
        The exponent1 attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._exponent1 = int(value)

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> PolyCoef1DType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return PolyCoef1DType: the deserialized class instance
        """

        if kwargs is None:
            kwargs = {}
        kwargs['exponent1'] = node.getAttribute('exponent1')
        kwargs['value'] = _get_node_value(node)
        return super(PolyCoef1DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        # NB: this class cannot REALLY be extended sensibly (you can add attributes, but that's it),
        # so I'm just short-circuiting the pattern.
        if tag is None:
            tag = self.__class__.__name__
        node = _create_text_node(doc, tag, str(self._value), par=par)
        node.setAttribute('exponent1', str(self.exponent1))
        return node


class PolyCoef2DType(Serializable):
    # NB: based on field names, one could consider PolyCoef2DType an extension of PolyCoef1DType. This has not
    #   be done here, because I would not want an instance of PolyCoef2DType to evaluate as True when testing if
    #   instance of PolyCoef1DType.

    __slots__ = ('_value', '_exponent1', '_exponent2')
    __fields = ('value', 'exponent1', 'exponent2')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        """
        The constructor. This class represents the monomial - `value * x^{exponent1} * y^{exponent2}`.
        :param dict kwargs: valid keys are ['value', 'exponent1', 'exponent2'], all required.
        """

        self._value = self._exponent1 = self._exponent2 = None
        super(PolyCoef2DType, self).__init__(**kwargs)

    @property
    def value(self):  # type: () -> float
        """
        The value attribute.
        :return float:
        """

        return self._value

    @value.setter
    def value(self, val):  # type: (Union[str, int, float]) -> None
        """
        The value attribute setter.
        :param Union[str, int, float] val:
        :return None:
        """

        self._value = float(val)

    @property
    def exponent1(self):  # type: () -> int
        """
        The exponent1 attribute.
        :return:
        """
        return self._exponent1

    @exponent1.setter
    def exponent1(self, value):  # type: (Union[str, int, float]) -> None
        """
        The exponent1 attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._exponent1 = int(value)

    @property
    def exponent2(self):  # type: () -> float
        """
        The exponent2 attribute.
        :return float:
        """
        return self._exponent2

    @exponent2.setter
    def exponent2(self, value):  # type: (Union[str, int, float]) -> None
        """
        The exponent2 attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._exponent2 = int(value)

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> PolyCoef2DType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return PolyCoef2DType: the deserialized class instance
        """

        if kwargs is None:
            kwargs = {}
        kwargs['exponent1'] = node.getAttribute('exponent1')
        kwargs['exponent2'] = node.getAttribute('exponent2')
        kwargs['value'] = _get_node_value(node)
        return super(PolyCoef2DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        # NB: this class cannot REALLY be extended sensibly (you can add attributes, but that's it),
        # so I'm just short-circuiting the pattern.
        if tag is None:
            tag = self.__class__.__name__
        node = _create_text_node(doc, tag, str(self._value), par=par)
        node.setAttribute('exponent1', str(self.exponent1))
        node.setAttribute('exponent2', str(self.exponent2))
        return node


class Poly1DType(Serializable):
    __slots__ = ('_coefs', '_order1')
    __fields = ('coefs', 'order1')
    __required = ('coefs', )

    def __init__(self, **kwargs):
        """
        The constructor. Represents a one-variable polynomial, defined as the sum of monomial terms given in `coefs`.
        :param dict kwargs: valid key is 'coefs'.
        """

        self._coefs = self._order1 = None
        super(Poly1DType, self).__init__(**kwargs)

    @property
    def order1(self):  # type: () -> int
        """
        The order1 attribute [READ ONLY]  - that is, largest exponent presented in the monomial terms of coefs.
        :return int:
        """

        return self._order1

    @property
    def coefs(self):  # type: () -> list
        """
        The coefs attribute - the collection of PolyCoef1DType monomial terms.
        :return list:
        """

        return self._coefs

    @coefs.setter
    def coefs(self, value):  # type: (Union[minidom.NodeList, list]) -> None
        """
        The coefs attribute setter.
        :param Union[minidom.NodeList, list] value:
        :return None:
        """

        if value is None:
            self._coefs = []
        elif isinstance(value, minidom.NodeList):
            self._coefs = []
            for nod in value:
                self._coefs.append(PolyCoef1DType.from_node(nod))
        elif isinstance(value, list):
            if len(value) == 0:
                self._coefs = []
            elif isinstance(value[0], PolyCoef1DType):
                for entry in value:
                    if not isinstance(entry, PolyCoef1DType):
                        raise ValueError('The first coefs entry was an instance of PolyCoef1DType. It is required '
                                         'that all further entries must be instances of PolyCoef1DType')
                self._coefs = value
            elif isinstance(value[0], dict):
                self._coefs = []
                for entry in value:
                    if not isinstance(entry, dict):
                        raise ValueError('The first coefs entry was an instance of PolyCoef1DType. It is required '
                                         'that all further entries must be instances of PolyCoef1DType')
                    self._coefs.append(PolyCoef1DType.from_dict(entry))
            else:
                raise ValueError('Attempted coefs assignment with a list whose first entry was of '
                                 'unsupported type {}'.format(type(value[0])))
        else:
            raise ValueError('Attempted coefs assignment of unsupported type {}'.format(type(value)))
        the_order = 0
        if len(self._coefs) > 0:
            for entry in self._coefs:
                the_order = max(the_order, entry.exponent1)
        self._order1 = the_order

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> Poly1DType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return Poly1DType: the deserialized class instance
        """

        if kwargs is None:
            kwargs = {}
        # NB: do not extract order1, because it SHOULD BE a derived field.
        kwargs['coefs'] = node.getElementsByTagName('Coef')
        return super(Poly1DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(Poly1DType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                               exclude=exclude+('order1', 'coefs'))
        node.setAttribute('order1', str(self.order1))
        for entry in self.coefs:
            entry.to_node(doc, tag='Coef', par=node, strict=strict)
        return node


class Poly2DType(Serializable):
    __slots__ = ('_coefs', '_order1', '_order2')
    __fields = ('coefs', 'order1', 'order2')
    __required = ('coefs', )

    def __init__(self, **kwargs):
        """
        The constructor. Represents a two variable polynomial, defined as the sum of monomial terms given in `coefs`.
        :param dict kwargs: valid key is 'coefs'.
        """

        self._coefs = self._order1 = self._order2 = None
        super(Poly2DType, self).__init__(**kwargs)

    @property
    def order1(self):  # type: () -> int
        """
        The order1 attribute [READ ONLY]  - that is, largest exponent presented for the first variable in the monomial
        terms of coefs.
        :return int:
        """

        return self._order1

    @property
    def order2(self):  # type: () -> int
        """
        The order1 attribute [READ ONLY]  - that is, largest exponent presented for the second variable in the monomial
         terms of coefs.
        :return int:
        """

        return self._order2

    @property
    def coefs(self):  # type: () -> list
        """
        The coefs attribute - the collection of PolyCoef2DType monomial terms.
        :return list:
        """

        return self._coefs

    @coefs.setter
    def coefs(self, value):  # type: (Union[minidom.NodeList, list]) -> None
        """
        The coefs attribute setter.
        :param Union[minidom.NodeList, list] value:
        :return None:
        """

        if value is None:
            self._coefs = []
        elif isinstance(value, minidom.NodeList):
            self._coefs = []
            for nod in value:
                self._coefs.append(PolyCoef2DType.from_node(nod))
        elif isinstance(value, list):
            if len(value) == 0:
                self._coefs = []
            elif isinstance(value[0], PolyCoef2DType):
                for entry in value:
                    if not isinstance(entry, PolyCoef2DType):
                        raise ValueError('The first coefs entry was an instance of PolyCoef2DType. It is required '
                                         'that all further entries must be instances of PolyCoef2DType')
                self._coefs = value
            elif isinstance(value[0], dict):
                self._coefs = []
                for entry in value:
                    if not isinstance(entry, dict):
                        raise ValueError('The first coefs entry was an instance of dict. It is required '
                                         'that all further entries must be instances of dict')
                    self._coefs.append(PolyCoef2DType.from_dict(entry))
            else:
                raise ValueError('Attempted coefs assignment with a list whose first entry was of '
                                 'unsupported type {}'.format(type(value[0])))
        else:
            raise ValueError('Attempted coefs assignment of unsupported type {}'.format(type(value)))
        the_order1 = the_order2 = 0
        if len(self._coefs) > 0:
            for entry in self._coefs:
                the_order1 = max(the_order1, entry.exponent1)
                the_order2 = max(the_order2, entry.exponent2)
        self._order1 = the_order1
        self._order2 = the_order2

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> Poly2DType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return Poly2DType: the deserialized class instance
        """

        if kwargs is None:
            kwargs = {}
        # NB: do not extract order1, order2, because these are READ ONLY attributes
        kwargs['coefs'] = node.getElementsByTagName('Coef')
        return super(Poly2DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(Poly2DType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                               exclude=exclude+('order1', 'order2', 'coefs'))
        node.setAttribute('order1', str(self.order1))
        node.setAttribute('order2', str(self.order2))
        for entry in self.coefs:
            entry.to_node(doc, tag='Coef', par=node, strict=strict)
        return node


class XYZPolyType(Serializable):
    __slots__ = ('_X', '_Y', '_Z')
    __fields = ('X', 'Y', 'Z')
    __required = __fields

    def __init__(self, **kwargs):
        """
        The constructor. Provides single variable polynomials for each of `X`, `Y`, and `Z`.
        :param dict kwargs: valid keys are ['X', 'Y', 'Z'], all required.
        """

        self._X = self._Y = self._Z = None
        super(XYZPolyType, self).__init__(**kwargs)

    @property
    def X(self):  # type: () -> Poly1DType
        """
        The X polynomial attribute.
        :return Poly1DType:
        """

        return self._X

    @X.setter
    def X(self, value):  # type: (Union[Poly1DType, minidom.Element, dict]) -> None
        """
        The X attribute setter.
        :param Union[Poly1DType, minidom.Element, dict] value:
        :return None:
        """

        if value is None:
            self._X = None
        elif isinstance(value, Poly1DType):
            self._X = value
        elif isinstance(value, minidom.Element):
            self._X = Poly1DType.from_node(value)
        elif isinstance(value, dict):
            self._X = Poly1DType.from_dict(value)
        else:
            raise ValueError('Attempted X assignment of unsupported type {}'.format(type(value)))

    @property
    def Y(self):  # type: () -> Poly1DType
        """
        The Y polynomial attribute.
        :return Poly1DType:
        """

        return self._Y

    @Y.setter
    def Y(self, value):  # type: (Union[Poly1DType, minidom.Element, dict]) -> None
        """
        The Y attribute setter.
        :param Union[Poly1DType, minidom.Element, dict] value:
        :return None:
        """

        if value is None:
            self._Y = None
        elif isinstance(value, Poly1DType):
            self._Y = value
        elif isinstance(value, minidom.Element):
            self._Y = Poly1DType.from_node(value)
        elif isinstance(value, dict):
            self._Y = Poly1DType.from_dict(value)
        else:
            raise ValueError('Attempted Y assignment of unsupported type {}'.format(type(value)))

    @property
    def Z(self):  # type: () -> Poly1DType
        """
        The Z polynomial attribute.
        :return Poly1DType:
        """

        return self._Z

    @Z.setter
    def Z(self, value):  # type: (Union[Poly1DType, minidom.Element, dict]) -> None
        """
        The Z attribute setter.
        :param Union[Poly1DType, minidom.Element, dict] value:
        :return None:
        """

        if value is None:
            self._Z = None
        elif isinstance(value, Poly1DType):
            self._Z = value
        elif isinstance(value, minidom.Element):
            self._Z = Poly1DType.from_node(value)
        elif isinstance(value, dict):
            self._Z = Poly1DType.from_dict(value)
        else:
            raise ValueError('Attempted Z assignment of unsupported type {}'.format(type(value)))

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyAttributeType(XYZPolyType):
    __slots__ = ('_X', '_Y', '_Z', '_index')
    __fields = ('X', 'Y', 'Z', 'index')
    __required = __fields

    def __init__(self, **kwargs):
        """
        The constructor. This class extension enables making an array/list of XYZPolyType instances.
        :param dict kwargs: valid keys are in ['X', 'Y', 'Z', 'index'], all required.
        """

        self._index = None
        super(XYZPolyAttributeType, self).__init__(**kwargs)

    @property
    def index(self):  # type: () -> int
        """
        The index attribute.
        :return int:
        """
        return self._index

    @index.setter
    def index(self, value):  # type: (Union[str, int, float]) -> None
        """
        The index attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._index = int(value)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> XYZPolyAttributeType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return XYZPolyAttributeType: the deserialized class instance
        """

        if kwargs is None:
            kwargs = {}
        kwargs['index'] = node.getAttribute('index')
        return super(XYZPolyAttributeType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(XYZPolyAttributeType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                                         exclude=exclude+('index', ))
        node.setAttribute('index', str(self.index))
        return node


class GainPhasePolyType(Serializable):
    __slots__ = ('_GainPoly', '_PhasePoly')
    __fields = ('GainPoly', 'PhasePoly')
    __required = __fields

    def __init__(self, **kwargs):
        """
        The constructor. The class is simply a container for the Gain and Phase Polygon definitions.
        :param dict kwargs: valid keys are ['GainPoly', 'PhasePoly'], all required.
        """

        self._GainPoly = self._PhasePoly = None
        super(GainPhasePolyType, self).__init__(**kwargs)

    @property
    def GainPoly(self):  # type: () -> Poly2DType
        """
        The GainPoly attribute.
        :return Poly2DType:
        """

        return self._GainPoly

    @GainPoly.setter
    def GainPoly(self, value):  # type: (Union[Poly2DType, minidom.Element, dict]) -> None
        """
        The GainPoly attribute setter.
        :param Union[Poly2DType, minidom.Element, dict] value:
        :return None:
        """

        if value is None:
            self._GainPoly = None
        elif isinstance(value, Poly2DType):
            self._GainPoly = value
        elif isinstance(value, minidom.Element):
            self._GainPoly = Poly2DType.from_node(value)
        elif isinstance(value, dict):
            self._GainPoly = Poly2DType.from_dict(value)
        else:
            raise ValueError('Attempted GainPoly assignment of unsupported type {}'.format(type(value)))

    @property
    def PhasePoly(self):  # type: () -> Poly2DType
        """
        The PhasePoly attribute.
        :return Poly2DType:
        """
        return self._PhasePoly

    @PhasePoly.setter
    def PhasePoly(self, value):  # type: (Union[Poly2DType, minidom.Element, dict]) -> None
        """
        The PhasePoly attribute setter.
        :param Union[Poly2DType, minidom.Element, dict] value:
        :return None:
        """

        if value is None:
            self._PhasePoly = None
        elif isinstance(value, Poly2DType):
            self._PhasePoly = value
        elif isinstance(value, minidom.Element):
            self._PhasePoly = Poly2DType.from_node(value)
        elif isinstance(value, dict):
            self._PhasePoly = Poly2DType.from_dict(value)
        else:
            raise ValueError('Attempted PhasePoly assignment of unsupported type {}'.format(type(value)))

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class LineType(Serializable):
    __slots__ = ('_Endpoints', )
    __fields = ('Endpoints', 'size')
    __required = ('Endpoints', )

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the only valid key is 'Endpoints'.
        """

        self._Endpoints = []
        super(LineType, self).__init__(**kwargs)

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. In this case, that len(Endpoints) > 1.
        :param bool recursive: Recursively check whether all attributes for validity.
        :return bool: condition for validity of this element
        """

        all_required = True
        if self.Endpoints is None:
            all_required = False
            logging.warning('Required field Endpoints is not populated for class {}'.format(self.__class__.__name__))
        elif self.size < 2:
            all_required = False
            logging.warning('Required field Endpoints only has length {} for '
                            'class {}'.format(self.size, self.__class__.__name__))
        if not recursive or not all_required:
            return all_required
        all_children = True
        for entry in self.Endpoints:
            all_children &= entry.is_valid(recursive=recursive)
        return all_required and all_children

    @property
    def size(self):  # type: () -> int
        """
        The size attribute.
        :return int:
        """

        return 0 if self.Endpoints is None else len(self.Endpoints)

    @property
    def Endpoints(self):  # type: () -> list
        """
        The Endpoints attribute. A list of elements of type LatLonType. This isn't itself part of the SICD standard,
        and is just an intermediate convenience method.
        :return list:
        """

        return self._Endpoints

    @Endpoints.setter
    def Endpoints(self, value):  # type: (Union[minidom.NodeList, list]) -> None
        """
        The Endpoints attribute setter.
        :param Union[minidom.NodeList, list] value: if list, the entries are expected to be (uniformly) either
        instances of LatLonType (the list reference is simply used), or dict instances suitable for constructing
        LatLonType instances (the list entries are used to construct a new list of LatLonType entries.
        :return None:
        """

        if value is None:
            self._Endpoints = []
        elif isinstance(value, minidom.NodeList):
            tlist = []
            for node in value:
                tlist.append((int(node.getAttribute('index')), LatLonType.from_node(node)))
            self._Endpoints = [entry for ind, entry in sorted(tlist, key=lambda x: x[0])]  # I've cheated an attribute in here
        elif isinstance(value, list):
            if len(value) == 0:
                self._Endpoints = value
            elif isinstance(value[0], LatLonType):
                for entry in value:
                    if not isinstance(entry, LatLonType):
                        raise ValueError('The first Endpoints entry was an instance of LatLonType. It is required '
                                         'that all further entries must be instances of LatLonType.')
                self._Endpoints = value
            elif isinstance(value[0], dict):
                self._Endpoints = []
                for entry in value:
                    if not isinstance(entry, dict):
                        raise ValueError('The first Endpoints entry was an instance of dict. It is required '
                                         'that all further entries must be instances of dict.')
                    self._Endpoints.append(LatLonType.from_dict(entry))
            else:
                raise ValueError('Attempted Endpoints assignment using a list with first element of '
                                 'unsupported type {}'.format(type(value[0])))
        else:
            raise ValueError('Attempted Endpoints assignment of unsupported type {}'.format(type(value)))

    # TODO: helper methods for functionality, again?

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> LineType
        """
        For XML deserialization.

        :param minidom.Element node: dom element for serialized class instance
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return LineType: corresponding class instance
        """

        if kwargs is None:
            kwargs = {}
        kwargs['Endpoints'] = node.getElementsByTagName('EndPoint')
        return super(LineType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(LineType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                             exclude=exclude+('Endpoints', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Endpoints):
            entry.to_node(doc, par=node, tag='EndPoint', strict=strict, exclude=()).setAttribute('index', str(i))
        return node


class PolygonType(Serializable):
    __slots__ = ('_Vertices', )
    __fields = ('Vertices', 'size')
    __required = ('Vertices', )

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the only valid key is 'Vertices'.
        """
        self._Vertices = []
        super(PolygonType, self).__init__(**kwargs)

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. In this case, that len(Vertices) > 1.
        :param bool recursive: whether to recursively check all attributes for validity.
        :return bool: condition for validity of this element
        """

        all_required = True
        if self.Vertices is None:
            all_required = False
            logging.warning('Required field Vertices is not populated for class {}'.format(self.__class__.__name__))
        elif self.size < 3:
            all_required = False
            logging.warning('Required field Vertices only has length {} for '
                            'class {}'.format(self.size, self.__class__.__name__))
        if not recursive or not all_required:
            return all_required
        all_children = True
        for entry in self.Vertices:
            all_children &= entry.is_valid(recursive=recursive)
        return all_required and all_children

    @property
    def size(self):  # type: () -> int
        """
        The size attribute.
        :return int:
        """
        return 0 if self.Vertices is None else len(self.Vertices)

    @property
    def Vertices(self):  # type: () -> list
        """
        The Vertices attribute. A list of elements of type LatLonRestrictionType. This isn't itself part of the SICD
        standard, and is just an intermediate convenience method.
        :return list:
        """

        return self._Vertices

    @Vertices.setter
    def Vertices(self, value):  # type: (Union[minidom.NodeList, list]) -> None
        """
        The Vertices attribute setter.
        :param Union[minidom.NodeList, list] value: if list, the entries are expected to be (uniformly) either
        instances of LatLonRestrictionType(the list reference is simply used), or dict instances suitable for
        constructing LatLonRestrictionType instances (the list entries are used to construct a new list of
        LatLonRestrictionType entries.
        :return None:
        """

        if value is None:
            self._Vertices = []
        elif isinstance(value, minidom.NodeList):
            tlist = []
            for node in value:
                tlist.append((int(node.getAttribute('index')), LatLonRestrictionType.from_node(node)))
            self._Vertices = [entry for ind, entry in sorted(tlist, key=lambda x: x[0])]
        elif isinstance(value, list):
            if len(value) == 0:
                self._Vertices = value
            elif isinstance(value[0], LatLonRestrictionType):
                for entry in value:
                    if not isinstance(entry, LatLonRestrictionType):
                        raise ValueError('The first Vertices entry was an instance of LatLonRestrictionType. '
                                         'It is required that all further entries must be instances of '
                                         'LatLonRestrictionType.')
                self._Vertices = value
            elif isinstance(value[0], dict):
                self._Vertices = []
                for entry in value:
                    if not isinstance(entry, dict):
                        raise ValueError('The first Vertices entry was an instance of dict. It is required '
                                         'that all further entries must be instances of dict.')
                    self._Vertices.append(LatLonRestrictionType.from_dict(entry))
            else:
                raise ValueError('Attempted Vertices assignment using a list with first element of '
                                 'unsupported type {}'.format(type(value[0])))
        else:
            raise ValueError('Attempted Vertices assignment of unsupported type {}'.format(type(value)))

    # TODO: helper methods for functionality, again?

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> PolygonType
        """
        For XML deserialization.

        :param minidom.Element node: dom element for serialized class instance
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return PolygonType: corresponding class instance
        """

        if kwargs is None:
            kwargs = {}
        kwargs['Vertices'] = node.getElementsByTagName('Vertex')
        return super(PolygonType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(PolygonType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                                exclude=exclude+('Vertices', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Vertices):
            entry.to_node(doc, par=node, tag='Vertex', strict=strict, exclude=())\
                .setAttribute('index', str(i))
        return node


class ErrorDecorrFuncType(Serializable):
    __slots__ = ('_CorrCoefZero', '_DecorrRate')
    __fields = ('CorrCoefZero', 'DecorrRate')
    __required = __fields
    __numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '0.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['CorrCoefZero', 'DecorrRate'], all required.
        """

        self._CorrCoefZero = self._DecorrRate = None
        super(ErrorDecorrFuncType, self).__init__(**kwargs)

    @property
    def CorrCoefZero(self):  # type: () -> float
        """
        The CorrCoefZero attribute.
        :return float:
        """
        # TODO: MEDIUM - a useful explanation here is probably good.

        return self._CorrCoefZero

    @CorrCoefZero.setter
    def CorrCoefZero(self, value):  # type: (Union[str, int, float]) -> None
        """
        The CorrCoefZero attribute setter.
        :param Union[str, int, float] value:
        :return:
        """

        self._CorrCoefZero = float(value)

    @property
    def DecorrRate(self):  # type: () -> float
        """
        The DecorrRate attribute.
        :return float:
        """
        # TODO: MEDIUM - a useful explanation here is probably good.

        return self._DecorrRate

    @DecorrRate.setter
    def DecorrRate(self, value):  # type: (Union[str, int, float]) -> None
        """
        The DecorrRate attribute setter.
        :param Union[str, int, float] value:
        :return:
        """

        self._DecorrRate = float(value)

    # TODO: HIGH - this is supposed to be a "function". We should implement the functionality here.


class ParametersType(Serializable):
    # This isn't actually part of the SICD standard exactly, this is just a convenience
    # helper class for the functionality
    __slots__ = ('_entries', )
    __fields = ('entries', )
    __required = ()

    def __init__(self, **kwargs):
        """
        The constructor. This isn't directly part of the SICD standard, but rather a convenience helper class for
        consistent functionality. This is just a dictionary holding {name:value}.
        :param dict kwargs: The only valid key is 'entries'
        """

        self._entries = OrderedDict()
        super(ParametersType, self).__init__(**kwargs)

    @property
    def entries(self):  # type: () -> OrderedDict
        """
        The entries attribute, contains ordered dictionary {name:value} for parameters list.
        :return dict:
        """

        return self._entries

    @entries.setter
    def entries(self, value):  # type: (Union[minidom.NodeList, list, dict]) -> None
        """
        The entries setter.
        :param Union[minidom.nodeList, list, dict] value:
        :return None:
        """

        if value is None or (isinstance(value, list) and len(value) == 0):
            self._entries = OrderedDict()
        elif isinstance(value, dict):
            self._entries = value
        elif isinstance(value, minidom.NodeList) or (isinstance(value, list) and isinstance(value[0], minidom.Element)):
            self._entries = OrderedDict()
            for nod in value:
                self._entries[nod.getAttribute("name")] = _get_node_value(nod)

    @classmethod
    def from_node(cls, node, kwargs=None, tag='Parameter'):
        # type: (minidom.Element, Union[None, dict], str) -> ParametersType
        """
        For XML deserialization. Note that this one extracts all items that are direct children of the given node
        with given tag name.

        :param minidom.Element node: dom element for PARENT element of Parameter class instances
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :param str tag: what tag ito use for the Parameter elements?
        :return ParametersType: corresponding class instance
        """

        # NB: fish out all nodes with tag=tag that are the direct children of this node
        if kwargs is None:
            kwargs = {}
        kwargs['entries'] = [entry for entry in node.getElementsByTagName(tag) if entry.parentNode == node]
        return cls.from_dict(**kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> None
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return None:
        """

        if tag is None:
            tag = 'Parameter'
        for entry in self.entries:
            value = self.entries[entry]
            nod = _create_text_node(doc, tag, value, par=par)
            nod.setAttribute('name', entry)


class RadarModeType(Serializable):
    # Really just subordinate to CollectionInfo, but this doesn't hurt at all
    __slots__ = ('_ModeType', '_ModeId')
    __fields = ('ModeType', 'ModeId')
    __required = ('ModeType', )

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['ModeType', 'ModeId'], and 'ModeType' is required.
        """

        self._ModeType = self._ModeId = None
        super(RadarModeType, self).__init__(**kwargs)

    @property
    def ModeId(self):  # type: () -> Union[None, str]
        """
        Optional ModeId attribute.
        :return Union[None, str]:
        """

        return self._ModeId

    @ModeId.setter
    def ModeId(self, value):  # type: (object) -> None
        """
        ModeId attribute setter.
        :param object value:
        :return None:
        """

        if value is None:
            self._ModeId = value
        else:
            self._ModeId = str(value)

    @property
    def ModeType(self):  # type: () -> str
        """
        The ModeType attribute. It's value will be one of ["SPOTLIGHT", "STRIPMAP", "DYNAMIC STRIPMAP"].
        :return str:
        """

        return self._ModeType

    @ModeType.setter
    def ModeType(self, value):  # type: (Union[None, str]) -> None
        """
        The ModeType attribute setter. Input value must be one of ["SPOTLIGHT", "STRIPMAP", "DYNAMIC STRIPMAP"].
        :param str value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._ModeType = None
            logging.warning("Required attribute ModeType of class RadarModeType has been set to None.")
        elif isinstance(value, str):
            val = value.upper()
            allowed = ["SPOTLIGHT", "STRIPMAP", "DYNAMIC STRIPMAP"]
            if val in allowed:
                self._ModeType = val
            else:
                raise ValueError('Received {} for ModeType, which is restricted to values {}'.format(value, allowed))
        else:
            raise ValueError('value is required to be an instance of str')


class FullImage(Serializable):
    __slots__ = ('_NumRows', '_NumCols')
    __fields = ('NumRows', 'NumCols')
    __required = __fields

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['NumRows', 'NumCols'], all required.
        """

        self._NumRows = self._NumCols = None
        super(FullImage, self).__init__(**kwargs)

    @property
    def NumRows(self):  # type: () -> int
        """
        The NumRows attribute.
        :return int:
        """

        return self._NumRows

    @NumRows.setter
    def NumRows(self, value):  # type: (Union[str, int, float]) -> None
        """
        The NumRows attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._NumRows = int(value)

    @property
    def NumCols(self):  # type: () -> int
        """
        The NumCols attribute.
        :return int:
        """

        return self._NumCols

    @NumCols.setter
    def NumCols(self, value):  # type: (Union[str, int, float]) -> None
        """
        The NumCols attribute setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._NumCols = int(value)


class ValidDataType(Serializable):
    __slots__ = ('_Vertices', )
    __fields = ('Vertices', 'size')
    __required = ('Vertices', )

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the only valid key is 'Vertices'.
        """
        self._Vertices = []
        super(ValidDataType, self).__init__(**kwargs)

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. In this case, that len(Vertices) >= 3.
        :param bool recursive: whether to recursively check all attributes for validity.
        :return bool: condition for validity of this element
        """

        all_required = True
        if self.Vertices is None:
            all_required = False
            logging.warning('Required field Vertices is not populated for class {}'.format(self.__class__.__name__))
        elif self.size < 3:
            all_required = False
            logging.warning('Required field Vertices only has length {} for '
                            'class {}'.format(self.size, self.__class__.__name__))
        if not recursive or not all_required:
            return all_required
        all_children = True
        for entry in self.Vertices:
            all_children &= entry.is_valid(recursive=recursive)
        return all_required and all_children

    @property
    def size(self):  # type: () -> int
        """
        The size attribute.
        :return int:
        """
        return 0 if self.Vertices is None else len(self.Vertices)

    @property
    def Vertices(self):  # type: () -> list
        """
        The Vertices attribute. A list of elements of type RowColvertexType. This isn't itself part of the SICD
        standard, and is just an intermediate convenience method.
        :return list:
        """

        return self._Vertices

    @Vertices.setter
    def Vertices(self, value):  # type: (Union[minidom.NodeList, list]) -> None
        """
        The Vertices attribute setter.
        :param Union[minidom.NodeList, list] value: if list, the entries are expected to be (uniformly) either
        instances of RowColvertexType(the list reference is simply used), or dict instances suitable for
        constructing RowColvertexType instances (the list entries are used to construct a new list of
        RowColvertexType entries.
        :return None:
        """

        if value is None:
            self._Vertices = []
        elif isinstance(value, minidom.NodeList):
            tlist = []
            for node in value:
                tlist.append((int(node.getAttribute('index')), RowColvertexType.from_node(node)))
            self._Vertices = [entry for ind, entry in sorted(tlist, key=lambda x: x[0])]
        elif isinstance(value, list):
            if len(value) == 0:
                self._Vertices = value
            elif isinstance(value[0], RowColvertexType):
                for entry in value:
                    if not isinstance(entry, RowColvertexType):
                        raise ValueError('The first Vertices entry was an instance of RowColvertexType. '
                                         'It is required that all further entries must be instances of '
                                         'RowColvertexType.')
                self._Vertices = value
            elif isinstance(value[0], dict):
                self._Vertices = []
                for entry in value:
                    if not isinstance(entry, dict):
                        raise ValueError('The first Vertices entry was an instance of dict. It is required '
                                         'that all further entries must be instances of dict.')
                    self._Vertices.append(RowColvertexType.from_dict(entry))
            else:
                raise ValueError('Attempted Vertices assignment using a list with first element of '
                                 'unsupported type {}'.format(type(value[0])))
        else:
            raise ValueError('Attempted Vertices assignment of unsupported type {}'.format(type(value)))

    # TODO: helper methods for functionality, again?

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> ValidDataType
        """
        For XML deserialization.

        :param minidom.Element node: dom element for serialized class instance
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return PolygonType: corresponding class instance
        """

        if kwargs is None:
            kwargs = {}
        kwargs['Vertices'] = node.getElementsByTagName('Vertex')
        return super(ValidDataType, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(PolygonType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                                exclude=exclude+('Vertices', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Vertices):
            entry.to_node(doc, par=node, tag='Vertex', strict=strict, exclude=())\
                .setAttribute('index', str(i))
        return node


# TODO: corner type mumbo-jumbo


# direct building blocks for SICD


class CollectionInfo(Serializable):
    __slots__ = ('_CollectorName', '_IlluminatorName', '_CoreName', '_CollectType', '_RadarMode', '_Classification',
                 '_CountryCodes', '_Parameters')
    __fields = ('CollectorName', 'IlluminatorName', 'CoreName', 'CollectType', 'RadarMode', 'Classification',
                'CountryCodes', 'Parameters')
    __required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')

    def __init__(self, **kwargs):
        self.CollectorName = self.IlluminatorName = self.CoreName = self._CollectType = self._RadarMode = None
        self.Classification = self._CountryCodes = self._Parameters = None
        super(CollectionInfo, self).__init__(**kwargs)

    @property
    def CollectorName(self):  # type: () -> Union[None, str]
        """
        The [Optional] CollectorName attribute.
        :return Union[None, str]:
        """

        return self._CollectorName

    @CollectorName.setter
    def CollectorName(self, value):  # type: (Union[None, str]) -> None
        """
        The CollectorName attribute setter.
        :param Union[None, str] value:
        :return None:
        """

        if value is None or isinstance(value, str):
            self._CollectorName = value
        else:
            self._CollectorName = str(value)

    @property
    def IlluminatorName(self):  # type: () -> Union[None, str]
        """
        The [Optional] IlluminatorName property.
        :return Union[None, str]:
        """

        return self._IlluminatorName

    @IlluminatorName.setter
    def IlluminatorName(self, value):  # type: (Union[None, str]) -> None
        """
        The IlluminatorName attribute setter.
        :param Union[None, str] value:
        :return None:
        """

        if value is None or isinstance(value, str):
            self._IlluminatorName = value
        else:
            self._IlluminatorName = str(value)

    @property
    def CoreName(self):  # type: () -> str
        """
        The CoreName attribute.
        :return str:
        """

        return self._CoreName

    @CoreName.setter
    def CoreName(self, value):  # type: (str) -> None
        """
        The CoreName attribute setter.
        :param str value:
        :return None:
        """

        if value is None:
            self._CoreName = None
            logging.warning("The required CoreName attribute of the CollectionInfo class has been set to None.")
        elif isinstance(value, str):
            self._CoreName = value
        else:
            raise ValueError("CoreName is required to be a string")

    @property
    def CollectType(self):  # type: () -> Union[None, str]
        """
        The [optional] CollectType attribute, which is None or one of ["MONOSTATIC", "BISTATIC"]
        :return Union[None, str]:
        """

        return self._CollectType

    @CollectType.setter
    def CollectType(self, value):  # type: (Union[None, str]) -> None
        """
        The CollectType attribute setter. Requires input is None, or one of ["MONOSTATIC", "BISTATIC"]
        :param Union[None, str] value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._CollectType = None
        elif not isinstance(value, str):
            raise ValueError('The CollectType attribute must be either None or a string.')

        val = value.upper()
        allowed = ["MONOSTATIC", "BISTATIC"]
        if val in allowed:
            self._CollectType = val
        else:
            raise ValueError('Received {} for CollectType, which is restricted to values {}'.format(value, allowed))

    @property
    def RadarMode(self):  # type: () -> RadarModeType
        """
        The RadarMode attribute.
        :return RadarModeType:
        """
        return self._RadarMode

    @RadarMode.setter
    def RadarMode(self, value):  # type: (Union[RadarModeType, minidom.Element, dict]) -> None
        """
        The RadarMode attribute setter.
        :param Union[RadarModeType, minidom.Element, dict] value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._RadarMode = None
            logging.warning("Required attribute RadarMode of class CollectionInfo has been set to None.")
        elif isinstance(value, RadarModeType):
            self._RadarMode = value
        elif isinstance(value, minidom.Element):
            self._RadarMode = RadarModeType.from_node(value)
        elif isinstance(value, dict):
            self._RadarMode = RadarModeType.from_dict(value)
        else:
            raise ValueError('Attempted RadarMode assignment of unsupported type {}'.format(type(value)))

    @property
    def Classification(self):  # type: () -> str
        """
        The Classification attribute.
        :return str:
        """

        return self._Classification

    @Classification.setter
    def Classification(self, value):  # type: (str) -> None
        """
        The Classification attribute setter.
        :param str value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._Classification = None
            logging.warning("The required Classification attribute of the CollectionInfo class has been set to None.")
        elif isinstance(value, str):
            self._Classification = value
        else:
            raise ValueError("Classification is required to be a string")

    @property
    def CountryCodes(self):  # type: () -> list
        """
        The [optional] CountryCodes attribute - a list of strings.
        :return list:
        """

        return self._CountryCodes

    @CountryCodes.setter
    def CountryCodes(self, value):  # type: (Union[None, str, list, minidom.NodeList]) -> None
        """
        The CountryCodes setter.
        :param Union[None, str, tuple, list, minidom.NodeList] value:
        :return None:
        """

        if value is None:
            self._CountryCodes = None
        elif isinstance(value, str):
            self._CountryCodes = [value, ]
        elif isinstance(value, minidom.NodeList) or (isinstance(value, list) and len(value) > 0
                                                     and isinstance(value[0], minidom.Element)):
            self._CountryCodes = []
            for nod in value:
                self._CountryCodes.append(_get_node_value(nod))
        elif isinstance(value, list):
            if len(value) == 0:
                self._CountryCodes = value
            elif isinstance(value[0], str):
                for entry in value:
                    if not isinstance(entry, str):
                        raise ValueError('CountryCodes received a list with first element an instance of str, '
                                         'so then requires that all elements are an instance of str.')
                self._CountryCodes = value
            else:
                raise ValueError('CountryCodes setter received a list with first element of '
                                 'incompatible type {}'.format(type(value[0])))
        else:
            raise ValueError('CountryCodes setter received incompatible type {}'.format(type(value)))

    @property
    def Parameters(self):  # type: () -> ParametersType
        """
        The Parameters attribute.
        :return ParametersType:
        """

        return self._Parameters

    @Parameters.setter
    def Parameters(self, value):  # type: (Union[None, ParametersType, dict, minidom.Element]) -> None
        """
        The Parameters attribute setter.
        :param Union[None, dict, minidom.Element] value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._Parameters = None
        elif isinstance(value, ParametersType):
            self._Parameters = value
        elif isinstance(value, dict) or isinstance(value, minidom.Element):
            self._Parameters = ParametersType(entries=value)
        else:
            raise ValueError('Parameters setter received incompatible type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> CollectionInfo
        """
        For XML deserialization.

        :param minidom.Element node: dom element for serialized class instance
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return CollectionInfo: corresponding class instance
        """

        if kwargs is None:
            kwargs = {}
        # Get the direct Children with tagname == 'CountryCode' - this should be unnecessary, but is safe
        kwargs['CountryCodes'] = [entry for entry in node.getElementsByTagName('CountryCode') if entry.parentNode == node]
        kwargs['Parameters'] = ParametersType.from_node(node, kwargs=None, tag='Parameter')  # extracts Parameters directly
        super(CollectionInfo, cls).from_node(node, kwargs=kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=False,  # type: bool
                exclude=()  # type: tuple
                ):
        # type: (...) -> minidom.Element
        """
        For XML serialization, to a dom element.

        :param  minidom.Document doc: the xml Document
        :param Union[None, str] tag: the tag name. The class name is unspecified.
        :param Union[None, minidom.Element] par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        node = super(CollectionInfo, self).to_node(doc, tag=tag, par=par, strict=strict,
                                                   exclude=exclude+('CountryCodes', 'Parameters'))
        if self.CountryCodes is not None:
            for entry in self.CountryCodes:
                _create_text_node(doc, 'CountryCode', entry, par=node)
        if self.Parameters is not None:
            self.Parameters.to_node(doc, tag='Parameter', par=node, strict=strict, exclude=())
        return node


class ImageCreationType(Serializable):
    __slots__ = ('_Application', '_DateTime', '_Site', '_Profile')
    __fields = ('Application', 'DateTime', 'Site', 'Profile')
    __required = ()

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['Application', 'DateTime', 'Site', 'Profile'], and only 'DateTime'
        is required.
        """

        self._Application = self._DateTime = self._Site = self._Profile = None
        super(ImageCreationType, self).__init__(**kwargs)

    @property
    def Application(self):  # type: () -> Union[None, str]
        """
        The [optional] Application attribute.
        :return Union[None, str]:
        """

        return self._Application

    @Application.setter
    def Application(self, value):  # type: (Union[None, str]) -> None
        """
        The Application attribute setter.
        :param Union[None, str] value:
        :return None:
        """

        if value is None:
            self._Application = None
        elif isinstance(value, str):
            self._Application = value
        else:
            raise ValueError('The Application attribute requires None or str')

    @property
    def DateTime(self):  # type: () -> numpy.datetime64
        """
        The DateTime attribute. Unless manually set by the user, the units will be `us`.
        :return numpy.datetime64:
        """

        return self._DateTime

    @DateTime.setter
    def DateTime(self, value):  # type: (Union[date, datetime, str, numpy.datetime64]) -> None
        """
        The DateTime attribute setter.
        :param Union[date, datetime, str, numpy.datetime64] value:
        :return:
        """
        if value is None:
            self._DateTime = None
            logging.warning("Required attribute DateTime of class ImageCreationType has been set to None.")
        elif isinstance(value, (date, datetime, str)):
            self._DateTime = numpy.datetime64(value, 'us')  # let's default to microsecond precision
        elif isinstance(value, numpy.datetime64):
            self._DateTime = value
        else:
            raise ValueError("The DateTime attribute requires input to be an instance of date, datetime, "
                             "str, or numpy.datetime64")

    @property
    def Site(self):  # type: () -> Union[None, str]
        """
        The [optional] Site attribute.
        :return Union[None, str]:
        """

        return self._Site

    @Site.setter
    def Site(self, value):  # type: (Union[None, str]) -> None
        """
        The Site attribute setter.
        :param Union[None, str] value:
        :return None:
        """

        if value is None:
            self._Site = None
        elif isinstance(value, str):
            self._Site = value
        else:
            raise ValueError('The Site attribute requires None or str')

    @property
    def Profile(self):  # type: () -> Union[None, str]
        """
        The [optional] Profile attribute.
        :return Union[None, str]:
        """

        return self._Profile

    @Profile.setter
    def Profile(self, value):  # type: (Union[None, str]) -> None
        """
        The Profile attribute setter.
        :param Union[None, str] value:
        :return None:
        """

        if value is None:
            self._Profile = None
        elif isinstance(value, str):
            self._Profile = value
        else:
            raise ValueError('The Profile attribute requires None or str')


class ImageDataType(Serializable):
    __slots__ = ('_PixelType', '_AmpTable', '_NumRows', '_NumCols', '_FirstRow', '_FirstCol', '_FullImage',
                 '_SCPPixel', '_ValidData')
    __fields = ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel',
                'ValidData')
    __required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    __numeric_format = {'AmpTable': '0.8f'}  # TODO: precision for AmpTable?

    def __init__(self, **kwargs):
        self._PixelType = self._AmpTable = self._NumRows = self._NumCols = self._FirstRow = self._FirstCol = None
        self._FullImage = self._SCPPixel = self._ValidData = None
        super(ImageDataType, self).__init__(**kwargs)

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.
        :param bool recursive: whether to recursively check validity of attributes
        :return bool: condition for validity of this element

        .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`
        """

        condition = super(ImageDataType, self).is_valid(recursive=recursive)
        pixel_type = not (self.PixelType == 'AMP8I_PHS8I' and self.AmpTable is None)
        if not pixel_type:
            logging.warning("We have `PixelType='AMP8I_PHS8I'` and `AmpTable` is undefined for ImageDataType.")
        return condition and pixel_type

    @property
    def PixelType(self):  # type: () -> str
        """
        The PixelType attribute which specifies the interpretation of the file data:
            * `"RE32F_IM32F"` - a pixel is specified as `(real, imaginary)` each 32 bit floating point.
            * `"RE16I_IM16I"` - a pixel is specified as `(real, imaginary)` each a 16 bit (short) integer.
            * `"AMP8I_PHS8I"` - a pixel is specified as `(amplitude, phase)` each an 8 bit unsigned integer. The
                `amplitude` actually specifies the index into the `AmpTable` attribute. The `angle` is properly
                interpreted (in radians) as `theta = 2*pi*angle/256`.
        :return str:
        """

        return self._PixelType

    @PixelType.setter
    def PixelType(self, value):   # type: (str) -> None
        """
        The PixelType attribute setter. Requires one of ["RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I"]
        :param str value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._PixelType = None
            logging.warning("The required PixelType attribute of the ImageDataType class has been set to None.")
        elif isinstance(value, str):
            val = value.upper()
            allowed = ["RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I"]
            if val in allowed:
                self._PixelType = val
            else:
                raise ValueError('Received {} for PixelType, which is restricted to values {}'.format(value, allowed))
        else:
            raise ValueError('The PixelType attribute must be either None or a string.')

    @property
    def AmpTable(self):  # type: () -> Union[None, numpy.ndarray]
        """
        The AmpTable attribute. This must be defined if PixelType == 'AMP8I_PHS8I'
        :return Union[None, numpy.ndarray]:
        """

        return self._AmpTable

    @AmpTable.setter
    def AmpTable(self, value):  # type: (Union[None, numpy.ndarray, minidom.Element]) -> None
        """
        The AmpTable attribute setter.
        :param Union[None, numpy.ndarray, minidom.Element] value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._AmpTable = None
        elif isinstance(value, minidom.Element):
            tarr = numpy.full((256, ), numpy.nan, dtype=numpy.float64)
            anodes = value.getElementsByTagName('Amplitude')
            if len(anodes) != 256:
                raise ValueError("The AmpTable attribute requires 256 amplitude entries, "
                                 "and we found {}".format(len(anodes)))
            for anode in anodes:
                index = int(anode.getAttribute('index'))
                avalue = float(_get_node_value(anode))
                tarr[index] = avalue
            if numpy.any(numpy.isnan(tarr)):
                raise ValueError("The AmpTable attribute did not have all [0-255] entries defined.")
            self._AmpTable = tarr
        elif isinstance(value, numpy.ndarray):
            if value.dtype != numpy.float64:
                logging.warning("The AmpTable attribute generally is expected to be of dtype float64, "
                                "and we got {}".format(value.dtype))
            if value.shape != (256, ):
                raise ValueError("The AmpTable attribute requires an ndarray of shape (256, ).")
            self._AmpTable = numpy.copy(value)  # I'm never sure whether to copy...we probably should here

    @property
    def NumRows(self):  # type: () -> int
        """
        The NumRows attribute.
        :return int:
        """

        # TODO: how is this related to FullImage? Different? What?
        return self._NumRows

    @NumRows.setter
    def NumRows(self, value):  # type: (Union[str, int, float]) -> None
        """
        The NumRows setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._NumRows = int(value)

    @property
    def NumCols(self):  # type: () -> int
        """
        The NumCols attribute.
        :return int:
        """

        return self._NumCols

    @NumCols.setter
    def NumCols(self, value):  # type: (Union[str, int, float]) -> None
        """
        The NumCols setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._NumCols = int(value)

    @property
    def FirstRow(self):  # type: () -> int
        """
        The FirstRow attribute.
        :return int:
        """

        return self._FirstRow

    @FirstRow.setter
    def FirstRow(self, value):  # type: (Union[str, int, float]) -> None
        """
        The FirstRow setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._FirstRow = int(value)

    @property
    def FirstCol(self):  # type: () -> int
        """
        The FirstCol attribute.
        :return int:
        """

        return self._FirstCol

    @FirstCol.setter
    def FirstCol(self, value):  # type: (Union[str, int, float]) -> None
        """
        The FirstRow setter.
        :param Union[str, int, float] value:
        :return None:
        """

        self._FirstCol = int(value)

    @property
    def FullImage(self):  # type: () -> FullImage
        """
        The FullImage attribute.
        :return FullImage:
        """

        return self._FullImage

    @FullImage.setter
    def FullImage(self, value):  # type: (Union[FullImage, dict, minidom.Element]) -> None
        """
        The FullImage attribute setter.
        :param Union[FullImage, dict, minidom.Element] value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._FullImage = None
            logging.warning("The required FullImage attribute of the ImageDataType class has been set to None")
        elif isinstance(value, FullImage):
            self._FullImage = value
        elif isinstance(value, dict):
            self._FullImage = FullImage.from_dict(value)
        elif isinstance(value, minidom.Element):
            self._FullImage = FullImage.from_node(value)
        else:
            raise ValueError('FullImage setter got incompatible type {}'.format(type(value)))

    @property
    def SCPPixel(self):  # type: () -> RowColType
        """
        The SCPPixel attribute.
        :return RowColType:
        """

        # TODO: some good description of this attribute?
        return self._SCPPixel

    @SCPPixel.setter
    def SCPPixel(self, value):  # type: (Union[RowColType, dict, minidom.Element]) -> None
        """
        The SCPPixel attribute setter.
        :param Union[RowColType, dict, minidom.Element] value:
        :return None:
        """

        if value is None:
            self._SCPPixel = None
            logging.warning("The required attribute SCPPixel for class ImageDataType has been set to None")
        elif isinstance(value, RowColType):
            self._SCPPixel = value
        elif isinstance(value, dict):
            self._SCPPixel = RowColType.from_dict(value)
        elif isinstance(value, minidom.Element):
            self._SCPPixel = RowColType.from_node(value)
        else:
            raise ValueError('SCPPixel setter got incompatible type {}'.format(type(value)))

    @property
    def ValidData(self): # type: () -> ValidDataType
        """
        The ValidData attribute.
        :return ValidDataType:
        """

        return self._ValidData

    @ValidData.setter
    def ValidData(self, value):  # type: (Union[None, ValidDataType, dict, minidom.Element]) -> None
        """
        The ValidData attribute setter.
        :param Union[None, ValidDataType, dict, minidom.Element] value:
        :return None:
        :raises: ValueError
        """

        if value is None:
            self._ValidData = None
        elif isinstance(value, ValidDataType):
            self._ValidData = value
        elif isinstance(value, dict):
            self._ValidData = ValidDataType.from_dict(value)
        elif isinstance(value, minidom.Element):
            self._ValidData = ValidDataType.from_node(value)
        else:
            raise ValueError("ValidData setter got incompatible type {}".format(type(value)))


