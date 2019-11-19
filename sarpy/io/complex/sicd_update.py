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


#################
# dom helper functions, because minidom is a little weird

def _get_node_value(nod):
    """
    XML parsing helper for extracting text value from an minidom node. No error checking performed.

    :param nod: xml node object
    :return: string value of the node
    """

    return nod.firstChild.wholeText.strip()


def _create_new_node(doc, tag, par=None):
    """
    XML minidom node creation helper function

    :param doc: xml Document
    :param tag: text name for new node
    :param par: parent node, which defaults to the root document node
    :return: the new node element
    """

    nod = doc.createElement(tag)
    if par is None:
        doc.documentElement.appendChild(nod)
    else:
        par.appendChild(nod)
    return nod


def _create_text_node(doc, tag, value, par=None):
    """
    XML minidom text node creation helper function

    :param doc: xml Document
    :param tag: text name for new node
    :param value: string value for node contents
    :param par: parent node, which defaults to the root document node
    :return: the new node element
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

    def __init__(self, **kwargs):
        """
        The default constructor. For each attribute name in self.__fields(), fetches the value (or None) from
        the kwargs dictionary, and sets the class attribute value. Any fields requiring special validation should
        be implemented as properties.

        :param kwargs: the keyword arguments dictionary
        """
        # TODO: LOW - should we validate the keys in kwargs?

        for attribute in self.__required:
            if attribute not in self.__fields:
                raise AttributeError("Attribute {} is defined in __required, but is not listed in __fields "
                                     "for class {}".format(attribute, self.__class__.__name__))

        for attribute in self.__fields:
            setattr(self, attribute, kwargs.get(attribute, None))

    def set_numeric_format(self, attribute, format_string):
        """
        Set the numeric format string for the given attribute

        :param attribute: attribute string (must be in __fields)
        :param format_string: format string to be applied
        """

        if attribute not in self.__fields:
            raise ValueError('attribute {} is not permitted for class {}'.format(attribute, self.__class__.__name__))
        self.__numeric_format[attribute] = format_string

    def _is_valid(self):
        """
        Verify that this standard element is valid
        :return: boolean condition for validity of this element
        """

        if len(self.__required) == 0:
            return True
        else:
            cond = True
            for attribute in self.__required:
                cond &= getattr(self, attribute) is not None
            return cond

    @classmethod
    def from_node(cls, node, kwargs=None):
        """
        For XML deserialization.

        :param node: dom node for serialized class instance
        :param kwargs: None or dictionary of previously serialized attributes. For use in inheritance call,
            when certain attributes require specific deserialization.
        :return: corresponding class instance
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

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        """
        For XML serialization, to a dom node object.

        :param doc: dom (xml.dom.minidom) Document
        :param par: potential parent node. The document root node will be used if not provided.
        :param strict: boolean condition of whether to raise an Exception if the structure is not valid.
        :param exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return: the constructed dom node object, already assigned as a child to parent node.
        """

        if not self._is_valid():
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
    def from_dict(cls, inputDict):
        """
        For json deserialization.

        :param inputDict: dict instance for deserialization
        :return: corresponding class instance
        """

        return cls(**inputDict)

    def to_dict(self, strict=False, exclude=()):
        """
        For json serialization.

        :param strict: boolean condition of whether to raise an Exception if the structure is not valid.
        :param exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return: dict representation (OrderedDict) of class instance appropriate for json serialization. Recall
            that any elements of `exclude` will be omitted, and should likely be included by the extension class
            implementation.
        """

        out = OrderedDict()

        # TODO: finish this as above in xml serialization
        raise NotImplementedError("Must provide a concrete implementation.")


#############
# Basic building blocks for SICD standard

# TODO: note that this shadows a builtin type. We should probably use the builtin type and define serialization
#  appropriately in Serializable. I'll leave this here for now.
class ComplexType(Serializable):
    __slots__ = ('_Real', '_Imag')
    __fields = ('Real', 'Imag')
    __required = __fields
    __numeric_format = {'Real': '0.8f', 'Imag': '0.8f'}

    def __init__(self, **kwargs):
        super(ComplexType, self).__init__(**kwargs)

    @property
    def Real(self):
        return self._Real

    @Real.setter
    def Real(self, value):
        self._Real = float(value)

    @property
    def Imag(self):
        return self._Imag

    @Imag.setter
    def Imag(self, value):
        self._Imag = float(value)

    def get_value(self):
        return self.Real + self.Imag*1j


class XYZType(Serializable):
    __slots__ = ('_X', '_Y', '_Z')
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    __numeric_format = {}  # TODO: desired precision (default is 0.8f)? This is usually meters?

    def __init__(self, **kwargs):
        self._X = self._Y = self._Z = None
        super(XYZType, self).__init__(**kwargs)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = float(value)

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
        self._Y = float(value)

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, value):
        self._Z = float(value)

    def getArray(self, dtype=numpy.float64):
        return numpy.array([self.X, self.Y, self.Z], dtype=dtype)


class LatLonType(Serializable):
    __slots__ = ('_Lat', '_Lon')
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        self._Lat = self._Lon = None
        super(LatLonType, self).__init__(**kwargs)

    @property
    def Lat(self):
        return self._Lat

    @Lat.setter
    def Lat(self, value):
        self._Lat = float(value)

    @property
    def Lon(self):
        return self._Lon

    @Lon.setter
    def Lon(self, value):
        self._Lon = float(value)

    def getArray(self, order='Lon', dtype=numpy.float64):
        if order == 'Lat':
            return numpy.array([self.Lat, self.Lon], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat], dtype=dtype)


class LatLonRestrictionType(LatLonType):
    __slots__ = ('_Lat', '_Lon')
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        super(LatLonType, self).__init__(**kwargs)

    @property
    def Lat(self):  # NB: you cannot redefine only the inherited setter
        return self._Lat

    @Lat.setter
    def Lat(self, value):
        val = float(value)
        if -90 <= val <= 90:
            self._Lat = val
        else:
            raise ValueError("Received {} for Latitude, but required to be in [-90, 90]".format(value))

    @property
    def Lon(self):
        return self._Lon

    @Lon.setter
    def Lon(self, value):
        val = float(value)
        if -180 <= val <= 180:
            self._Lon = val
        else:
            raise ValueError("Received {} for Longitude, but required to be in [-180, 180]".format(value))

    def getArray(self, order='Lon', dtype=numpy.float64):
        if order == 'Lat':
            return numpy.array([self.Lat, self.Lon], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat], dtype=dtype)


class LatLonHAEType(LatLonType):
    __slots__ = ('_Lat', '_Lon', '_HAE')
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        self._HAE = None
        super(LatLonHAEType, self).__init__(**kwargs)

    @property
    def HAE(self):
        return self._HAE

    @HAE.setter
    def HAE(self, value):
        self._HAE = float(value)

    def getArray(self, order='Lon', dtype=numpy.float64):
        if order == 'Lat':
            return numpy.array([self.Lat, self.Lon, self.HAE], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat, self.HAE], dtype=dtype)


class LatLonHAERestrictionType(LatLonRestrictionType):
    __slots__ = ('_Lat', '_Lon', '_HAE')
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        self._HAE = None
        super(LatLonHAERestrictionType, self).__init__(**kwargs)

    @property
    def HAE(self):
        return self._HAE

    @HAE.setter
    def HAE(self, value):
        self._HAE = float(value)

    def getArray(self, order='Lon', dtype=numpy.float64):
        if order == 'Lat':
            return numpy.array([self.Lat, self.Lon, self.HAE], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat, self.HAE], dtype=dtype)


class RowColType(Serializable):
    __slots__ = ('_Row', '_Col')
    __fields = ('Row', 'Col')
    __required = __fields

    def __init__(self, **kwargs):
        self._Row = self._Col = None
        super(RowColType, self).__init__(**kwargs)

    @property
    def Row(self):
        return self._Row

    @Row.setter
    def Row(self, value):
        self._Row = int(value)

    @property
    def Col(self):
        return self._Col

    @Col.setter
    def Col(self, value):
        self._Col = int(value)


class RowColvertexType(RowColType):
    __slots__ = ('_Row', '_Col', '_index')
    __fields = ('Row', 'Col', 'index')
    __required = __fields

    def __init__(self, **kwargs):
        self._index = None
        super(RowColvertexType, self).__init__(**kwargs)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = int(value)

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['index'] = node.getAttribute('index')
        return super(RowColvertexType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        node = super(RowColvertexType, self).to_node(doc, tag=tag, par=par, strict=strict, exclude=exclude)
        node.setAttribute('index', str(self.index))
        return node


class PolyCoef1DType(Serializable):
    __slots__ = ('_value', '_exponent1')
    __fields = ('value', 'exponent1')
    __required = __fields

    def __init__(self, **kwargs):
        self._value = self._exponent1 = None
        super(PolyCoef1DType, self).__init__(**kwargs)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = float(val)

    @property
    def exponent1(self):
        return self._exponent1

    @exponent1.setter
    def exponent1(self, value):
        self._exponent1 = int(value)

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['exponent1'] = node.getAttribute('exponent1')
        kwargs['value'] = _get_node_value(node)
        return super(PolyCoef1DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        # NB: this class cannot REALLY be extended sensibly (you can add attributes, but that's it),
        # so I'm just short-circuiting the pattern.
        if tag is None:
            tag = self.__class__.__name__
        node = _create_text_node(doc, tag, str(self._value), par=par)
        node.setAttribute('exponent1', str(self.exponent1))
        return node


class PolyCoef2DType(PolyCoef1DType):
    __slots__ = ('_value', '_exponent1', '_exponent2')
    __fields = ('value', 'exponent1', 'exponent2')
    __required = __fields

    def __init__(self, **kwargs):
        self._exponent2 = None
        super(PolyCoef2DType, self).__init__(**kwargs)

    @property
    def exponent2(self):
        return self._exponent2

    @exponent2.setter
    def exponent2(self, value):
        self._exponent2 = int(value)

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['exponent1'] = node.getAttribute('exponent1')
        kwargs['exponent2'] = node.getAttribute('exponent2')
        kwargs['value'] = _get_node_value(node)
        return super(PolyCoef1DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        node = super(PolyCoef2DType, self).to_node(doc, tag=tag, par=par, strict=strict, exclude=exclude)
        node.setAttribute('exponent2', str(self.exponent2))
        return node


class Poly1DType(Serializable):
    __slots__ = ('_coefs', '_order1')
    __fields = ('coefs', 'order1')
    __required = __fields

    def __init__(self, **kwargs):
        self._coefs = self._order1 = None
        super(Poly1DType, self).__init__(**kwargs)

    @property
    def order1(self):
        return self._order1

    @order1.setter
    def order1(self, value):
        self._order1 = int(value)

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, value):
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
        self.order1 = the_order

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['order1'] = node.getAttribute('order1')
        kwargs['coefs'] = node.getElementsByTagName('Coef')
        return super(Poly1DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        node = super(Poly1DType, self).to_node(doc, tag=tag, par=par, strict=strict, exclude=exclude+('order1', 'coefs'))
        node.setAttribute('order1', str(self.order1))
        for entry in self.coefs:
            entry.to_node(doc, tag='Coef', par=node, strict=strict)
        return node


class Poly2DType(Serializable):
    __slots__ = ('_coefs', '_order1', '_order2')
    __fields = ('coefs', 'order1', 'order2')
    __required = __fields

    def __init__(self, **kwargs):
        self._coefs = self._order1 = self._order2 = None
        super(Poly2DType, self).__init__(**kwargs)

    @property
    def order1(self):
        return self._order1

    @order1.setter
    def order1(self, value):
        self._order1 = int(value)

    @property
    def order2(self):
        return self._order2

    @order2.setter
    def order2(self, value):
        self._order2 = int(value)

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, value):
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
        self.order1 = the_order1
        self.order2 = the_order2

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['order1'] = node.getAttribute('order1')
        kwargs['order2'] = node.getAttribute('order2')
        kwargs['coefs'] = node.getElementsByTagName('Coef')
        return super(Poly2DType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
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
        self._X = self._Y = self._Z = None
        super(XYZPolyType, self).__init__(**kwargs)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
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
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
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

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, value):
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


class XYZPolyAttributeType(Serializable):
    __slots__ = ('_X', '_Y', '_Z', '_index')
    __fields = ('X', 'Y', 'Z', 'index')
    __required = __fields

    def __init__(self, **kwargs):
        self._index = None
        super(XYZPolyAttributeType, self).__init__(**kwargs)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = int(value)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['index'] = node.getAttribute('index')
        return super(XYZPolyAttributeType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        node = super(XYZPolyAttributeType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                               exclude=exclude+('index', ))
        node.setAttribute('index', str(self.index))
        return node


class GainPhasePolyType(Serializable):
    __slots__ = ('_GainPoly', '_PhasePoly')
    __fields = ('GainPoly', 'PhasePoly')
    __required = __fields

    def __init__(self, **kwargs):
        self._GainPoly = self._PhasePoly = None
        super(GainPhasePolyType, self).__init__(**kwargs)

    @property
    def GainPoly(self):
        return self._GainPoly

    @GainPoly.setter
    def GainPoly(self, value):
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
    def PhasePoly(self):
        return self._PhasePoly

    @PhasePoly.setter
    def PhasePoly(self, value):
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
    __required = ('Endpoints', 'size')

    def __init__(self, **kwargs):
        self._Endpoints = []
        super(LineType, self).__init__(**kwargs)

    def _is_valid(self):
        return len(self.Endpoints) > 1

    @property
    def size(self):
        return len(self.Endpoints)

    @property
    def Endpoints(self):
        return self._Endpoints

    @Endpoints.setter
    def Endpoints(self, value):
        if value is None:
            self._Endpoints = []
        elif isinstance(value, minidom.NodeList):
            self._Endpoints = [None, ]*len(value)
            for node in value:
                i = int(node.getAttribute('index'))
                self._Endpoints[i] = LatLonType.from_node(node)  # Note tht I've cheated an attribute in here
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
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['Endpoints'] = node.getElementsByTagName('EndPoint')
        return super(LineType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        node = super(LineType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                             exclude=exclude+('Endpoints', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Endpoints):
            entry.to_node(doc, par=node, strict=strict, exclude=())\
                .setAttribute('index', str(i))
        return node


class PolygonType(Serializable):
    __slots__ = ('_Vertices', )
    __fields = ('Vertices', 'size')
    __required = ('Vertices', 'size')

    def __init__(self, **kwargs):
        self._Vertices = []
        super(PolygonType, self).__init__(**kwargs)

    def _is_valid(self):
        return len(self.Vertices) > 2

    @property
    def size(self):
        return len(self.Vertices)

    @property
    def Vertices(self):
        return self._Vertices

    @Vertices.setter
    def Vertices(self, value):
        if value is None:
            self._Vertices = []
        elif isinstance(value, minidom.NodeList):
            self._Vertices = [None, ]*len(value)
            for node in value:
                i = int(node.getAttribute('index'))
                self._Vertices[i] = LatLonRestrictionType.from_node(node)  # Note tht I've cheated an attribute in here
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
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['Vertices'] = node.getElementsByTagName('Vertex')
        return super(PolygonType, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        node = super(PolygonType, self).to_node(doc, tag=tag, par=par, strict=strict,
                                                exclude=exclude+('Vertices', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Vertices):
            entry.to_node(doc, par=node, strict=strict, exclude=())\
                .setAttribute('index', str(i))
        return node


class ErrorDecorrFuncType(Serializable):
    __slots__ = ('_CorrCoefZero', '_DecorrRate')
    __fields = ('CorrCoefZero', 'DecorrRate')
    __required = __fields
    __numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '3.8f'}  # TODO: desired precision?

    def __init__(self, **kwargs):
        self._CorrCoefZero = self._DecorrRate = None
        super(ErrorDecorrFuncType, self).__init__(**kwargs)

    @property
    def CorrCoefZero(self):
        return self._CorrCoefZero

    @CorrCoefZero.setter
    def CorrCoefZero(self, value):
        self._CorrCoefZero = float(value)

    @property
    def DecorrRate(self):
        return self._DecorrRate

    @DecorrRate.setter
    def DecorrRate(self, value):
        self._DecorrRate = float(value)


class Parameters(Serializable):
    # This isn't actually part of the SICD standard exactly, this is just a convenience
    # helper class for the functionality
    __slots__ = ('_entries', )
    __fields = ('entries', )
    __required = ()

    def __init__(self, **kwargs):
        self._entries = OrderedDict()
        super(Parameters, self).__init__(**kwargs)

    @property
    def entries(self):
        return self._entries

    @entries.setter
    def entries(self, value):
        if value is None:
            self._entries = OrderedDict()
        elif isinstance(value, dict):
            self._entries = value
        elif isinstance(value, minidom.NodeList):
            self._entries = OrderedDict()
            for nod in value:
                self._entries[nod.getAttribute("name")] = _get_node_value(nod)

    @classmethod
    def from_node(cls, node, kwargs=None):
        # NB: fish out all Parameter nodes from node, children at any level
        if kwargs is None:
            kwargs = {}
        pnodes = node.getElementsByTagName('Parameter')
        kwargs['entries'] = pnodes
        return cls.from_dict(**kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        if tag is None:
            tag = 'Parameter'
        for entry in self.entries:
            value = self.entries[value]
            nod = _create_text_node(doc, tag, value, par=par)
            nod.setAttribute('name', entry)


class RadarMode(Serializable):
    # Really just subordinate to CollectionInfo, but this doesn't hurt at all
    __slots__ = ('_ModeType', 'ModeId')
    __fields = ('ModeType', 'ModeId')
    __required = ('ModeType', )

    def __init__(self, **kwargs):
        self._ModeType =  self.ModeId = None
        super(RadarMode, self).__init__(**kwargs)

    @property
    def ModeType(self):
        return self._ModeType

    @ModeType.setter
    def ModeType(self, value):
        val = value.upper()
        allowed = ["SPOTLIGHT", "STRIPMAP", "DYNAMIC STRIPMAP"]
        if val in allowed:
            self._ModeType = val
        else:
            raise ValueError('Received {} for ModeType, which is restricted to values {}'.format(value, allowed))


class FullImage(Serializable):
    __slots__ = ('_NumRows', '_NumCols')
    __fields = ('NumRows', 'NumCols')
    __required = __fields

    def __init__(self, **kwargs):
        self._NumRows = self._NumCols = None
        super(FullImage, self).__init__(**kwargs)

    @property
    def NumRows(self):
        return self._NumRows

    @NumRows.setter
    def NumRows(self, value):
        self._NumRows = int(value)

    @property
    def NumCols(self):
        return self._NumCols

    @NumCols.setter
    def NumCols(self, value):
        self._NumCols = int(value)

class ValidData(Serializable): # TODO: define ValidData type - part of ImageDataType
    __slots__ = ()
    __fields = ()
    __required = ()

    def __init__(self, **kwargs):
        super(ValidData, self).__init__(**kwargs)


# TODO: corner type mumbo-jumbo

# direct building blocks for SICD

class CollectionInfo(Serializable):
    __slots__ = ('CollectorName', 'IlluminatorName', 'CoreName', '_CollectType', '_RadarMode', 'Classification',
                 '_CountryCodes', '_Parameters')
    __fields = ('CollectorName', 'IlluminatorName', 'CoreName', 'CollectType', 'RadarMode', 'Classification',
                'CountryCodes', 'Parameters')
    __required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')

    def __init__(self, **kwargs):
        self.CollectorName = self.IlluminatorName = self.CoreName = self._CollectType = self._RadarMode = None
        self.Classification = self._CountryCodes = self._Parameters = None
        super(CollectionInfo, self).__init__(**kwargs)

    @property
    def CollectType(self):
        return self._CollectType

    @CollectType.setter
    def CollectType(self, value):
        val = value.upper()
        allowed = ["MONOSTATIC", "BISTATIC"]
        if val in allowed:
            self._CollectType = val
        else:
            raise ValueError('Received {} for CollectType, which is restricted to values {}'.format(value, allowed))

    @property
    def RadarMode(self):
        return self._RadarMode

    @RadarMode.setter
    def RadarMode(self, value):
        if value is None:
            self._RadarMode = None
        elif isinstance(value, RadarMode):
            self._RadarMode = value
        elif isinstance(value, minidom.Element):
            self._RadarMode = RadarMode.from_node(value)
        elif isinstance(value, dict):
            self._RadarMode = RadarMode.from_dict(value)
        else:
            raise ValueError('Attempted RadarMode assignment of unsupported type {}'.format(type(value)))

    @property
    def CountryCodes(self):
        return self._CountryCodes

    @CountryCodes.setter
    def CountryCodes(self, value):
        if value is None:
            self._CountryCodes = None
        elif isinstance(value, str):
            self._CountryCodes = [value, ]
        elif isinstance(value, tuple):
            self._CountryCodes = list(value)
        elif isinstance(value, minidom.NodeList):
            self._CountryCodes = []
            for nod in value:
                self._CountryCodes.append(_get_node_value(nod))
        elif isinstance(value, list):
            if len(value) == 0:
                self._CountryCodes = value
            elif isinstance(value[0], str):
                for entry in value:
                    if not isinstance(entry, str):
                        raise ValueError('CountryCodes received a list ith first element an instance of str, '
                                         'so then requires that all elements are an instance of str.')
                self._CountryCodes = value
            else:
                raise ValueError('CountryCodes setter received a list with first element of '
                           'incompatible type {}'.format(type(value[0])))
        else:
            raise ValueError('CountryCodes setter received incompatible type {}'.format(type(value)))

    @property
    def Parameters(self):
        return self._Parameters

    @Parameters.setter
    def Parameters(self, value):
        if value is None:
            self._Parameters = None
        elif isinstance(value, dict) or isinstance(value, minidom.NodeList):
            self._Parameters = Parameters(entries=value)
        else:
            raise ValueError('Parameters setter received incompatible type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs['CountryCodes'] = node.getElementsByTagName('CountryCode')
        kwargs['Parameters'] = node.getElementsByTagName('Parameter')
        super(CollectionInfo, cls).from_node(node, kwargs=kwargs)

    def to_node(self, doc, tag=None, par=None, strict=False, exclude=()):
        node = super(CollectionInfo, self).to_node(doc, tag=tag, par=par, strict=strict,
                                                   exclude=exclude+('CountryCodes', 'Parameters'))
        if self.CountryCodes is not None:
            for entry in self.CountryCodes:
                _create_text_node(doc, 'CountryCode', entry, par=node)
        if self.Parameters is not None:
            self.Parameters.to_node(doc, tag='Parameter', par=node, strict=strict, exclude=())


class ImageCreation(Serializable):
    __slots__ = ('Application', '_DateTime', 'Site', 'Profile')
    __fields = ('Application', 'DateTime', 'Site', 'Profile')
    __required = ()

    def __init__(self, **kwargs):
        self.Application = self._DateTime = self.Site = self.Profile = None
        super(ImageCreation, self).__init__(**kwargs)

    @property
    def DateTime(self):
        return self._DateTime

    @DateTime.setter
    def DateTime(self, value):
        if value is None:
            self._DateTime = None
        elif isinstance(value, (date, datetime, str)):
            self._DateTime = numpy.datetime64(value, 'us')  # let's default to microsecond precision
        elif isinstance(value, numpy.datetime64):
            self._DateTime = value


class ImageDataType(Serializable):
    __slots__ = ('_PixelType', '_AmpTable', '_NumRows', '_NumCols', '_FirstRow', '_FirstCol', '_FullImage',
                 '_SCPPixel', '_ValidData')
    __fields = ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel',
                'ValidData')
    __required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')

    def __init__(self, **kwargs):
        self._PixelType = self._AmpTable = self._NumRows = self._NumCols = self._FirstRow = self._FirstCol = None
        self._FullImage = self._SCPPixel = self._ValidData = None
        super(ImageDataType, self).__init__(**kwargs)

    @property
    def NumRows(self):
        return self._NumRows

    @NumRows.setter
    def NumRows(self, value):
        self._NumRows = int(value)

    @property
    def NumCols(self):
        return self._NumCols

    @NumCols.setter
    def NumCols(self, value):
        self._NumCols = int(value)

    @property
    def FirstRow(self):
        return self._FirstRow

    @FirstRow.setter
    def FirstRow(self, value):
        self._FirstRow = int(value)

    @property
    def FirstCol(self):
        return self._FirstCol

    @FirstCol.setter
    def FirstCol(self, value):
        self._FirstCol = int(value)

    @property
    def SCPPixel(self):
        return self._SCPPixel

    @SCPPixel.setter
    def SCPPixel(self, value):
        if value is None:
            self._SCPPixel = None
        elif isinstance(value, RowColType):
            self._SCPPixel = value
        elif isinstance(value, dict):
            self._SCPPixel = RowColType.from_dict(value)
        elif isinstance(value, minidom.Element):
            self._SCPPixel = RowColType.from_node(value)
        else:
            raise ValueError('SCPPixel setter got incompatible type {}'.format(type(value)))

    @property
    def FullImage(self):
        return self._FullImage

    @FullImage.setter
    def FullImage(self, value):
        if value is None:
            self._FullImage = None
        elif isinstance(value, FullImage):
            self._FullImage = value
        elif isinstance(value, dict):
            self._FullImage = FullImage.from_dict(value)
        elif isinstance(value, minidom.Element):
            self._FullImage = FullImage.from_node(value)
        else:
            raise ValueError('FullImage setter got incompatible type {}'.format(type(value)))

    # TODO: define ValidData types

