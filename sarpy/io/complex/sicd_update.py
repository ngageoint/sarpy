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
from weakref import WeakKeyDictionary
from typing import Union

#################
# any module constants?
DEFAULT_STRICT = False  # type: bool


#################
# descriptor (i.e. reusable properties) definition

class _BasicDescriptor(object):
    """A descriptor object for reusable properties. Note that is is required that the calling instance is hashable."""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, docstring=''):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "Basic descriptor"

    def __get__(self, instance, owner):
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return str: the return value
        """

        fetched = self.data.get(instance, None)
        if fetched is not None or not self.required:
            return fetched
        elif self.strict:
            raise AttributeError(
                'Required field {} of class {} is not populated.'.format(self.name, instance.__class__.__name__))
        else:
            logging.info(
                'Required field {} of class {} is not populated.'.format(self.name, instance.__class__.__name__))
            return fetched

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return bool:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = None
            return True
        # note that the remainder must be implemented in each extension
        return False  # this is probably a bad habit, but this returns something for convenience alone


class _StringDescriptor(_BasicDescriptor):
    """A descriptor for string type properties"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, docstring=None):
        super(_StringDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):  # type: (object, str) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_StringDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, str):
            self.data[instance] = value
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))


class _StringListDescriptor(_BasicDescriptor):
    """A descriptor for properties of a assumed to be an array type item for specified extension of Serializable"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, docstring=None):
        super(_StringListDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_StringListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, str):
            self.data[instance] = [value, ]
        elif isinstance(value, minidom.Element):
            self.data[instance] = [_get_node_value(value), ]
        elif isinstance(value, minidom.NodeList):
            self.data[instance] = [_get_node_value(nod) for nod in value]
        elif isinstance(value, list):
            if len(value) == 0:
                self.data[instance] = value
            elif isinstance(value[0], str):
                self.data[instance] = value
            elif isinstance(value[0], minidom.Element):
                self.data[instance] = [_get_node_value(nod) for nod in value]
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _StringEnumDescriptor(_BasicDescriptor):
    """A descriptor for enumerated string type properties"""

    def __init__(self, name, values, required=False, strict=DEFAULT_STRICT, docstring=None, default_value=None):
        super(_StringEnumDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.values = values
        self.defaultValue = default_value if default_value in values else None

    def __set__(self, instance, value):  # type: (object, str) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_StringEnumDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, str):
            val = value.upper()
            if val in self.values:
                self.data[instance] = value
            elif self.strict:
                raise ValueError(
                    'Received value {} for field {} of class {}, but values are required to be one of {}'.format(
                        value, self.name, instance.__class__.__name__, self.values))
            else:
                logging.warning(
                    'Received value {} for field {} of class {}, but values are required to be one of {}. '
                    'The value has been set to None.'.format(
                        value, self.name, instance.__class__.__name__, self.values))
                self.data[instance] = None
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))


class _IntegerDescriptor(_BasicDescriptor):
    """A descriptor for integer type properties"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, bounds=None, docstring=None):
        super(_IntegerDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.bounds = bounds

    def __set__(self, instance, value):  # type: (object, int) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_IntegerDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        iv = int(value)
        if self.bounds is None or (self.bounds[0] <= iv <= self.bounds[1]):
            self.data[instance] = iv
        else:
            if self.strict:
                raise ValueError(
                    'field {} of class {} must take value between {}'.format(
                        self.name, instance.__class__.__name__, self.bounds))
            else:
                logging.info(
                    'Required field {} of class {} is required to take value between {}.'.format(
                        self.name, instance.__class__.__name__, self.bounds))
            self.data[instance] = iv


class _IntegerEnumDescriptor(_BasicDescriptor):
    """A descriptor for integer type properties"""

    def __init__(self, name, values, required=False, strict=DEFAULT_STRICT, docstring=None):
        super(_IntegerEnumDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.values = values

    def __set__(self, instance, value):  # type: (object, int) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_IntegerEnumDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        iv = int(value)
        if iv in self.values:
            self.data[instance] = iv
        else:
            if self.strict:
                raise ValueError(
                    'field {} of class {} must take value in {}'.format(
                        self.name, instance.__class__.__name__, self.values))
            else:
                logging.info(
                    'Required field {} of class {} is required to take value in {}.'.format(
                        self.name, instance.__class__.__name__, self.values))
            self.data[instance] = iv


class _FloatDescriptor(_BasicDescriptor):
    """A descriptor for float type properties"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, docstring=None):
        super(_FloatDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):  # type: (object, float) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_FloatDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        self.data[instance] = float(value)


class _FloatArrayDescriptor(_BasicDescriptor):
    """A descriptor for float array type properties"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, child_tag='ArrayDouble', docstring=None):
        super(_FloatArrayDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.child_tag = child_tag

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_FloatArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, numpy.ndarray):
            if not (len(value) == 1) and (numpy.dtype == numpy.float64):
                raise ValueError('Only one-dimensional ndarrays of dtype float64 are supported here.')
            self.data[instance] = value
        elif isinstance(value, minidom.Element):
            new_value = []
            for node in value.getElementsByTagName(self.child_tag):
                new_value.append((int(node.getAttribute('index')), float(_get_node_value(node))))
            self.data[instance] = numpy.array([
                val[0] for val in sorted(new_value, key=lambda x: x[0])], dtype=numpy.float64)
        elif isinstance(value, minidom.NodeList) or \
                (isinstance(value, list) and len(value) > 0 and isinstance(value, minidom.Element)):
            new_value = []
            for node in value:
                new_value.append((int(node.getAttribute('index')), float(_get_node_value(node))))
            self.data[instance] = numpy.array([
                entry for ind, entry in sorted(new_value, key=lambda x: x[0])], dtype=numpy.float64)
        elif isinstance(value, list):
            self.data[instance] = numpy.array(value, dtype=numpy.float64)
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _DateTimeDescriptor(_BasicDescriptor):
    """A descriptor for date time type properties"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, docstring=None, numpy_datetime_units='us'):
        super(_DateTimeDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.units = numpy_datetime_units  # s, ms, us, ns are likely choices here, depending on needs

    def __set__(self, instance, value):  # type: (object, Union[date, datetime, str, int, numpy.datetime64]) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_DateTimeDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, numpy.datetime64):
            self.data[instance] = value
        else:
            self.data[instance] = numpy.datetime64(value, self.units)


class _FloatModularDescriptor(_BasicDescriptor):
    """A descriptor for float type properties"""

    def __init__(self, name, limit, required=False, strict=DEFAULT_STRICT, docstring=None):
        super(_FloatModularDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.limit = limit

    def __set__(self, instance, value):  # type: (object, float) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_FloatModularDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        val = float(value)  # do modular arithmatic manipulations
        val = (val % (2*self.limit))  # NB: % and * have same precedence, so it can be super dumb
        self.data[instance] = val if val <= self.limit else val - 2*self.limit


class _SerializableDescriptor(_BasicDescriptor):
    """A descriptor for properties of a specified type assumed to be an extension of Serializable"""

    def __init__(self, name, the_type, docstring=None, required=False, strict=DEFAULT_STRICT, tag=None):
        super(_SerializableDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.the_type = the_type
        self.tag = tag

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_SerializableDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, self.the_type):
            self.data[instance] = value
        elif isinstance(value, dict):
            self.data[instance] = self.the_type.from_dict(value).modify_tag(self.tag)
        elif isinstance(value, minidom.Element):
            self.data[instance] = self.the_type.from_node(value).modify_tag(self.tag)
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _SerializableArrayDescriptor(_BasicDescriptor):
    """A descriptor for properties of a assumed to be an array type item for specified extension of Serializable"""

    def __init__(self, name, the_type, required=False, strict=DEFAULT_STRICT, tag=None, docstring=None):
        super(_SerializableArrayDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.the_type = the_type
        self.tag = tag

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_SerializableArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, self.the_type):
            self.data[instance] = [value, ]
        elif isinstance(value, minidom.Element):
            self.data[instance] = [self.the_type.from_node(value).modify_tag(self.tag), ]
        elif isinstance(value, minidom.NodeList):
            new_value = []
            for node in value:  # NB: I am ignoring the index attribute (if it exists) and leaving it in doc order
                new_value.append(self.the_type.from_node(node).modify_tag(self.tag))
            self.data[instance] = value
        elif isinstance(value, list):
            if len(value) == 0:
                self.data[instance] = value
            elif isinstance(value[0], self.the_type):
                self.data[instance] = value
            elif isinstance(value[0], dict):
                # NB: charming errors are possible if something stupid has been done.
                self.data[instance] = [self.the_type.from_dict(node).modify_tag(self.tag) for node in value]
            elif isinstance(value[0], minidom.Element):
                self.data[instance] = [self.the_type.from_node(node).modify_tag(self.tag) for node in value]
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


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

    __tag = None  # tag name when serializing
    __fields = ()  # collection of field names
    __required = ()  # define a non-empty tuple for required properties
    __child_tags = {}  # only needed for children where attribute_name != tag_name
    __numeric_format = {}  # define dict entries of numeric formatting for serialization
    __set_as_attribute = ()  # serialize these fields as xml attributes
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

        for attribute in self.__required:
            if attribute not in self.__fields:
                raise AttributeError(
                    "Attribute {} is defined in __required, but is not listed in __fields for class {}".format(
                        attribute, self.__class__.__name__))

        for attribute in self.__fields:
            try:
                setattr(self, attribute, kwargs.get(attribute, None))
            except AttributeError:
                # NB: this is included to allow for read only properties without breaking the paradigm
                pass

    def modify_tag(self, value):  # type: (str) -> None
        """
        Sets the default tag for serialization.
        :param str value:
        :return Serializable: returns a reference to self, for calling pattern simplicity
        """

        if value is None:
            return self
        elif isinstance(value, str):
            self.__tag = value
            return self
        else:
            raise TypeError("tag requires string input")

    def set_numeric_format(self, attribute, format_string):  # type: (str, str) -> None
        """
        Sets the numeric format string for the given attribute

        :param str attribute: attribute for which the format applies - must be in `__fields`.
        :param str format_string: format string to be applied
        """

        if attribute not in self.__fields:
            raise ValueError('attribute {} is not permitted for class {}'.format(attribute, self.__class__.__name__))
        self.__numeric_format[attribute] = format_string

    def _get_numeric_format(self, attribute):  # type: (str) -> str
        return None if attribute not in self.__numeric_format else '{0:' + self.__numeric_format[attribute] + '}'

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.

        :param bool recursive: should we recursively check that child are also valid?
        :return bool: condition for validity of this element

        .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
            that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then
            this will result in an infinite loop.
        """

        def check_item(value):
            good = True
            if isinstance(val, Serializable):
                good = val.is_valid(recursive=recursive)
            return good

        all_required = True
        if len(self.__required) > 0:
            for attribute in self.__required:
                present = getattr(self, attribute) is not None
                if not present:
                    logging.warning(
                        "class {} has missing required attribute {}".format(self.__class__.__name__, attribute))
                all_required &= present
        if not recursive:
            return all_required

        valid_children = True
        for attribute in self.__fields:
            val = getattr(self, attribute)
            good = True
            if isinstance(val, Serializable):
                good = check_item(val)
            elif isinstance(val, list):
                for entry in val:
                    good &= check_item(entry)
            valid_children &= good
            # any issues will be logged as discovered, but we should help with the "stack"
            if not good:
                logging.warning(
                    "Issue discovered with {} attribute of type {} of class {}".format(
                    attribute, type(val), self.__class__.__name__))
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
            raise ValueError(
                "Named input argument kwargs for class {} must be dictionary instance".format(cls.__class__.__name__))

        for attribute in cls.__fields:
            if attribute in kwargs:
                continue

            if attribute in cls.__set_as_attribute:
                val = node.getAttribute(attribute)
                kwargs[attribute] = None if len(val) == 0 else val
            else:
                child_tag = cls.__child_tags.get(attribute, None)
                pnodes = [entry for entry in node.getElementsByTagName(attribute) if entry.parentNode == node]
                cnodes = [] if child_tag is None else [
                    entry for entry in node.getElementsByTagName(child_tag) if entry.parentNode == node]
                if len(pnodes) == 1:
                    kwargs[attribute] = pnodes[0]
                elif len(pnodes) > 1:
                    # this is for list type attributes. Probably should have an entry in __child_tags.
                    kwargs[attribute] = pnodes
                elif len(cnodes) > 0:
                    kwargs[attribute] = cnodes
        return cls.from_dict(kwargs)

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=DEFAULT_STRICT,  # type: bool
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

        def serialize_child(value, attribute, child_tag, fmt, parent):
            if isinstance(value, Serializable):
                value.to_node(doc, tag=tag, par=parent, strict=strict)
            elif isinstance(value, str):  # TODO: MEDIUM - unicode issues?
                _create_text_node(doc, tag, value, par=parent)
            elif isinstance(value, int) or isinstance(value, float):
                if fmt is None:
                    _create_text_node(doc, attribute, str(value), par=parent)
                else:
                    _create_text_node(doc, attribute, fmt.format(value), par=parent)
            elif isinstance(value, complex):
                raise NotImplementedError  # TODO: let's figure it out
            elif isinstance(value, date):
                _create_text_node(doc, attribute, value.isoformat(), par=parent)
            elif isinstance(value, datetime):
                _create_text_node(doc, attribute, value.isoformat(sep='T'), par=parent)
                # TODO: LOW - this can be error prone with the naive vs zoned issue.
                #   Discourage using this type?
            elif isinstance(value, numpy.datetime64):
                _create_text_node(doc, attribute, str(value), par=parent)
            elif isinstance(value, list) or isinstance(value, tuple):
                # this is assumed to be serialized straight in, versus having a parent tag
                for val in value:
                    serialize_child(val, child_tag, child_tag, fmt, parent)
            elif isinstance(value, numpy.ndarray):
                if len(value.shape) != 1 or numpy.dtype != numpy.float64:
                    raise NotImplementedError
                anode = _create_new_node(doc, attribute, par=parent)
                anode.setAttribute('size', str(value.size))
                fmt_func = str if fmt is None else fmt.format
                for i, val in enumerate(value):
                    vnode = _create_text_node(doc, child_tag, fmt_func(val), par=anode)
                    vnode.setAttribute('index', str(i))
            else:
                raise ValueError(
                    'Attribute {} is of type {}, and not clearly serializable'.format(attribute, type(value)))

        if not self.is_valid():
            msg = "{} is not valid, and cannot be safely serialized to XML according to " \
                  "the SICD standard.".format(self.__class__.__name__)
            if strict:
                raise ValueError(msg)
            logging.warning(msg)

        if tag is None:
            tag = self.__tag
        nod = _create_new_node(doc, tag, par=par)

        for attribute in self.__fields:
            if attribute in exclude:
                continue
            value = getattr(self, attribute)
            if value is None:
                continue
            child_tag = self.__child_tags.get(attribute, attribute)
            fmt = self._get_numeric_format(attribute)
            if attribute in self.__set_as_attribute:
                if fmt is None:
                    nod.setAttribute(attribute, str(value))
                else:
                    nod.setAttribute(attribute, fmt.format(value))
            else:
                serialize_child(value, attribute, child_tag, fmt)
        return nod

    @classmethod
    def from_dict(cls, inputDict):  # type: (dict) -> Serializable
        """
        For json deserialization.

        :param dict inputDict: dict instance for deserialization
        :return Serializable: corresponding class instance
        """

        return cls(**inputDict)

    def to_dict(self, strict=DEFAULT_STRICT, exclude=()):  # type: (bool, tuple) -> OrderedDict
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

class PlainValueType(Serializable):
    """
    This is a basic xml building block element, and not actually specified in the SICD standard
    """
    __fields = ('value', )
    __required = __fields
    value = _StringDescriptor('value', required=True, strict=True, docstring='The value')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key is 'value', and it is required.
        """
        super(PlainValueType, self).__init__(**kwargs)

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> PlainValueType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return PlainValueType: the deserialized class instance
        """

        return cls(value=_get_node_value(node))

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=DEFAULT_STRICT,  # type: bool
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

        # we have to short-circuit the call here, because this is a really primitive element
        if tag is None:
            tag = self.__tag
        node = _create_text_node(doc, tag, self.value, par=par)
        return node


class FloatValueType(Serializable):
    """
    This is a basic xml building block element, and not actually specified in the SICD standard
    """
    __fields = ('value', )
    __required = __fields
    value = _FloatDescriptor('value', required=True, strict=True, docstring='The value')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key is 'value', and it is required.
        """
        super(FloatValueType, self).__init__(**kwargs)

    @classmethod
    def from_node(cls, node, kwargs=None):  # type: (minidom.Element, Union[None, dict]) -> FloatValueType
        """
        For XML deserialization.
        :param minidom.Element node: the dom Element to deserialize.
        :param Union[None, dict] kwargs: None or dictionary of previously serialized attributes. For use in
        inheritance call, when certain attributes require specific deserialization.
        :return FloatValueType: the deserialized class instance
        """

        return cls(value=_get_node_value(node))

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=DEFAULT_STRICT,  # type: bool
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

        # we have to short-circuit the call here, because this is a really primitive element
        if tag is None:
            tag = self.__tag
        fmt = self._get_numeric_format('value')
        if fmt is None:
            node = _create_text_node(doc, tag, str(self.value), par=par)
        else:
            node = _create_text_node(doc, tag, fmt.format(self.value), par=par)
        return node


class ParameterType(PlainValueType):
    __tag = 'Parameter'
    __fields = ('name', 'value')
    __required = __fields
    __set_as_attribute = ('name', )
    # descriptor
    name = _StringDescriptor('name', required=True, strict=True, docstring='The name')
    value = _StringDescriptor('value', required=True, strict=True, docstring='The value')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key are ('name', 'value'), all required.
        """
        super(ParameterType, self).__init__(**kwargs)


class XYZType(Serializable):
    __slots__ = ()  # prevent adhoc field definition, for speed and safety.
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    __numeric_format = {'X': '0.8f', 'Y': '0.8f', 'Z': '0.8f'}  # TODO: desired precision? This is usually meters?
    X = _FloatDescriptor(
        'X', required=True, strict=DEFAULT_STRICT, docstring='The X attribute. Assumed to ECF or other, similar coordinates.')
    Y = _FloatDescriptor(
        'Y', required=True, strict=DEFAULT_STRICT, docstring='The Y attribute. Assumed to ECF or other, similar coordinates.')
    Z = _FloatDescriptor(
        'Z', required=True, strict=DEFAULT_STRICT, docstring='The Z attribute. Assumed to ECF or other, similar coordinates.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('X', 'Y', 'Z'), all required.
        """
        super(XYZType, self).__init__(**kwargs)

    def getArray(self, dtype=numpy.float64):  # type: (numpy.dtype) -> numpy.ndarray
        """
        Gets an [X, Y, Z] array representation of the class.

        :param numpy.dtype dtype: data type of the return
        :return numpy.ndarray:
        """

        return numpy.array([self.X, self.Y, self.Z], dtype=dtype)


class LatLonType(Serializable):
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    Lat = _FloatDescriptor(
        'Lat', required=True, strict=DEFAULT_STRICT, docstring='The Latitude attribute. Assumed to be WGS 84 coordinates.')
    Lon = _FloatDescriptor(
        'Lon', required=True, strict=DEFAULT_STRICT, docstring='The Longitude attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon'), all required.
        """
        super(LatLonType, self).__init__(**kwargs)

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
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, required=True, strict=DEFAULT_STRICT,
        docstring='The Latitude attribute. Assumed to be WGS 84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, required=True, strict=DEFAULT_STRICT,
        docstring='The Longitude attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        super(LatLonRestrictionType, self).__init__(**kwargs)


class LatLonHAEType(LatLonType):
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?
    HAE = _FloatDescriptor(
        'HAE', required=True, strict=DEFAULT_STRICT,
        docstring='The Height Above Ellipsoid (in meters) attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'HAE'), all required.
        """
        super(LatLonHAEType, self).__init__(**kwargs)

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


class LatLonHAERestrictionType(LatLonHAEType):
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, required=True, strict=DEFAULT_STRICT,
        docstring='The Latitude attribute. Assumed to be WGS 84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, required=True, strict=DEFAULT_STRICT,
        docstring='The Longitude attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'HAE', 'index'), all required.
        """
        super(LatLonHAERestrictionType, self).__init__(**kwargs)


class LatLonCornerType(LatLonType):
    __fields = ('Lat', 'Lon', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', required=True, strict=True, bounds=(1, 4),
        docstring='The integer index in 1-4. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'index'), all required.
        """
        super(LatLonCornerType, self).__init__(**kwargs)


class LatLonCornerStringType(LatLonType):
    __fields = ('Lat', 'Lon', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, required=True, strict=True,
        docstring="The string index in {}".format(_CORNER_VALUES))

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'index'), all required.
        """
        super(LatLonCornerStringType, self).__init__(**kwargs)


class LatLonHAECornerRestrictionType(LatLonHAERestrictionType):
    __fields = ('Lat', 'Lon', 'HAE', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', required=True, strict=True,
        docstring='The integer index in 1-4. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'HAE', 'index'), all required.
        """
        super(LatLonHAECornerRestrictionType, self).__init__(**kwargs)


class LatLonHAECornerStringType(LatLonHAEType):
    __fields = ('Lat', 'Lon', 'HAE', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, required=True, strict=True, docstring="The string index in {}".format(_CORNER_VALUES))

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'HAE', 'index'), all required.
        """
        super(LatLonHAECornerStringType, self).__init__(**kwargs)


class RowColType(Serializable):
    __fields = ('Row', 'Col')
    __required = __fields
    Row = _IntegerDescriptor(
        'Row', required=True, strict=DEFAULT_STRICT, docstring='The (integer valued) Row attribute.')
    Col = _IntegerDescriptor(
        'Col', required=True, strict=DEFAULT_STRICT, docstring='The (integer valued) Column attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Row', 'Col'), both required.
        """
        super(RowColType, self).__init__(**kwargs)


class RowColvertexType(RowColType):
    __fields = ('Row', 'Col', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    index = _IntegerDescriptor(
        'index', required=True, strict=DEFAULT_STRICT, docstring='The (integer valued) index attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Row', 'Col', 'index'), all required.
        """
        super(RowColvertexType, self).__init__(**kwargs)


class PolyCoef1DType(FloatValueType):
    """
    Represents a monomial term of the form `value * x^{exponent1}`.
    """
    __fields = ('value', 'exponent1')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    __set_as_attribute = ('exponent1', )
    exponent1 = _IntegerDescriptor(
        'exponent1', required=True, strict=DEFAULT_STRICT, docstring='The (power) exponent1 attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('value', 'exponent1'), both required.
        """
        super(PolyCoef1DType, self).__init__(**kwargs)


class PolyCoef2DType(FloatValueType):
    """
    Represents a monomial term of the form `value * x^{exponent1} * y^{exponent2}`.
    """
    # NB: based on field names, one could consider PolyCoef2DType an extension of PolyCoef1DType. This has not
    #   be done here, because I would not want an instance of PolyCoef2DType to evaluate as True when testing if
    #   instance of PolyCoef1DType.

    __fields = ('value', 'exponent1', 'exponent2')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    __set_as_attribute = ('exponent1', 'exponent2')
    exponent1 = _IntegerDescriptor(
        'exponent1', required=True, strict=DEFAULT_STRICT, docstring='The (power) exponent1 attribute.')
    exponent2 = _IntegerDescriptor(
        'exponent2', required=True, strict=DEFAULT_STRICT, docstring='The (power) exponent2 attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('value', 'exponent1', 'exponent2'), both required.
        """
        super(PolyCoef2DType, self).__init__(**kwargs)


class Poly1DType(Serializable):
    """
    Represents a one-variable polynomial, defined as the sum of the given monomial terms.
    """
    __fields = ('Coefs', 'order1')
    __required = ('Coefs', )
    __child_tags = {'Coefs': 'Coef'}
    __set_as_attribute = ('order1', )
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef1DType, required=True, strict=DEFAULT_STRICT, tag='Coef',
        docstring='the collection of PolyCoef1DType monomial terms.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the only valid key is 'Coefs', and is required.
        """
        super(Poly1DType, self).__init__(**kwargs)

    @property
    def order1(self):  # type: () -> int
        """
        The order1 attribute [READ ONLY]  - that is, largest exponent presented in the monomial terms of coefs.
        :return int:
        """

        return 0 if self.Coefs is None else max(entry.exponent1 for entry in self.Coefs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class Poly2DType(Serializable):
    """
    Represents a one-variable polynomial, defined as the sum of the given monomial terms.
    """
    __fields = ('Coefs', 'order1', 'order2')
    __required = ('Coefs', )
    __child_tags = {'Coefs': 'Coef'}
    __set_as_attribute = ('order1', 'order2')
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef2DType, required=True, strict=DEFAULT_STRICT, tag='Coef',
        docstring='the collection of PolyCoef2DType monomial terms.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the only valid key is 'Coefs', and is required.
        """
        super(Poly2DType, self).__init__(**kwargs)

    @property
    def order1(self):  # type: () -> int
        """
        The order1 attribute [READ ONLY]  - that is, the largest exponent1 presented in the monomial terms of coefs.
        :return int:
        """

        return 0 if self.Coefs is None else max(entry.exponent1 for entry in self.Coefs)

    @property
    def order2(self):  # type: () -> int
        """
        The order2 attribute [READ ONLY]  - that is, the largest exponent2 presented in the monomial terms of coefs.
        :return int:
        """

        return 0 if self.Coefs is None else max(entry.exponent2 for entry in self.Coefs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyType(Serializable):
    """
    Represents a single variable polynomial for each of `X`, `Y`, and `Z`.
    """
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    # descriptors
    X = _SerializableDescriptor(
        'X', Poly1DType, required=True, strict=DEFAULT_STRICT, docstring='The X polynomial.')
    Y = _SerializableDescriptor(
        'Y', Poly1DType, required=True, strict=DEFAULT_STRICT, docstring='The Y polynomial.')
    Z = _SerializableDescriptor(
        'Z', Poly1DType, required=True, strict=DEFAULT_STRICT, docstring='The Z polynomial.')
    # TODO: a better description would be good here

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('X', 'Y', 'Z'), and are required.
        """
        super(XYZPolyType, self).__init__(**kwargs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyAttributeType(XYZPolyType):
    __fields = ('X', 'Y', 'Z', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    index = _IntegerDescriptor(
        'index', required=True, strict=DEFAULT_STRICT, docstring='The (array) index value.')

    def __init__(self, **kwargs):
        """The constructor. 
        :param dict kwargs: the valid keys are in ('X', 'Y', 'Z', 'index'), all required.
        """
        super(XYZPolyAttributeType, self).__init__(**kwargs)


class GainPhasePolyType(Serializable):
    """
    A container for the Gain and Phase Polygon definitions.
    """
    __fields = ('GainPoly', 'PhasePoly')
    __required = __fields
    # the descriptors
    GainPoly = _SerializableDescriptor(
        'GainPoly', Poly2DType, required=True, strict=DEFAULT_STRICT, docstring='The Gain (two variable) Polygon.')
    PhasePoly = _SerializableDescriptor(
        'GainPhasePoly', Poly2DType, required=True, strict=DEFAULT_STRICT, docstring='The Phase (two variable) Polygon.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are in ('GainPoly', 'PhasePoly'), all required.
        """
        super(GainPhasePolyType, self).__init__(**kwargs)


class LineType(Serializable):
    """
    A geographic line feature.
    """
    __fields = ('Endpoints', 'size')
    __required = ('Endpoints', )
    __child_tags = {'Endpoints': 'Endpoint'}
    __set_as_attribute = ('size', )
    # the descriptors
    Endpoints = _SerializableArrayDescriptor(
        'Endpoints', LatLonType, required=True, strict=DEFAULT_STRICT, tag='Endpoint',
        docstring="A list of elements of type LatLonType. This isn't directly part of the SICD standard, and just "
                  "represents an intermediate convenience object.")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key is 'Endpoints', and it is required.
        """
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
            logging.warning(
                'Required field Endpoints only has length {} for class {}'.format(self.size, self.__class__.__name__))
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

    # TODO: helper methods for functionality, again?

    def to_node(self,
                doc,  # type: minidom.Document
                tag=None,  # type: Union[None, str]
                par=None,  # type: Union[None, minidom.Element]
                strict=DEFAULT_STRICT,  # type: bool
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

        node = super(LineType, self).to_node(
            doc, tag=tag, par=par, strict=strict, exclude=exclude+('Endpoints', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Endpoints):
            entry.to_node(doc, par=node, tag='EndPoint', strict=strict, exclude=()).setAttribute('index', str(i))
        return node


class PolygonType(Serializable):
    """
    A geographic polygon element
    """
    __fields = ('Vertices', 'size')
    __required = ('Vertices', )
    __child_tags = {'Vertices': 'Vertex'}
    __set_as_attribute = ('size', )
    # child class definition

    class LatLonArrayElement(LatLonRestrictionType):
        """LatLon array element"""
        __tag = 'Vertex'
        __fields = ('Lat', 'Lon', 'index')
        __required = __fields
        __set_as_attribute = ('index', )
        index = _IntegerDescriptor('index', required=True, strict=True)

        def __init__(self, **kwargs):
            """The constructor.
            :param dict kwargs: the valid keys are in ('Lat', 'Lon', 'index'), all are required.
            """
            super(PolygonType.LatLonArrayElement, self).__init__(**kwargs)

    # descriptors
    Vertices = _SerializableArrayDescriptor(
        'Vertices', LatLonArrayElement, required=True, strict=DEFAULT_STRICT, tag='Vertex',
        docstring="A list of elements of type LatLonRestrictionType. This isn't directly part of the SICD standard, "
                  "and just represents an intermediate convenience object.")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys is 'Vertices', and it is required.
        """
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
            logging.warning(
                'Required field Vertices only has length {} for class {}'.format(self.size, self.__class__.__name__))
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

    # TODO: helper methods for functionality, again?


class ErrorDecorrFuncType(Serializable):
    """
    The Error Decorrelation Function?
    """
    __fields = ('CorrCoefZero', 'DecorrRate')
    __required = __fields
    __numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '0.8f'}  # TODO: desired precision?
    # descriptors
    CorrCoefZero = _FloatDescriptor(
        'CorrCoefZero', required=True, strict=DEFAULT_STRICT, docstring='The CorrCoefZero attribute.')
    DecorrRate = _FloatDescriptor(
        'DecorrRate', required=True, strict=DEFAULT_STRICT, docstring='The DecorrRate attribute.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['CorrCoefZero', 'DecorrRate'], all required.
        """
        super(ErrorDecorrFuncType, self).__init__(**kwargs)

    # TODO: HIGH - this is supposed to be a "function". We should implement the functionality here.


class RadarModeType(Serializable):
    """
    Radar mode type container class
    """
    __tag = 'RadarMode'
    __fields = ('ModeType', 'ModeId')
    __required = ('ModeType', )
    _MODE_TYPE_VALUES = ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP')
    # descriptors
    ModeId = _StringDescriptor('ModeId', required=False, docstring='The Mode Id.')
    ModeType = _StringEnumDescriptor(
        'ModeType', _MODE_TYPE_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring="The ModeType, which will be one of {}.".format(_MODE_TYPE_VALUES))

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['ModeType', 'ModeId'], and 'ModeType' is required.
        """
        super(RadarModeType, self).__init__(**kwargs)


class FullImageType(Serializable):
    """
    The full image attributes
    """
    __tag = 'FullImage'
    __fields = ('NumRows', 'NumCols')
    __required = __fields
    # descriptors
    NumRows = _IntegerDescriptor('NumRows', required=True, strict=DEFAULT_STRICT, docstring='The number of rows.')
    NumCols = _IntegerDescriptor('NumCols', required=True, strict=DEFAULT_STRICT, docstring='The number of columns.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['NumRows', 'NumCols'], all required.
        """
        super(FullImageType, self).__init__(**kwargs)


##########
# Direct building blocks for SICD


class CollectionInfoType(Serializable):
    """
    The collection information container.
    """
    __tag = 'CollectionInfo'
    __child_tags = {'Parameters': 'Parameter', 'CountryCode': 'CountryCode'}  # these list type ones are hand-jammed
    __fields = (
        'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
        'RadarMode', 'Classification', 'Parameters', 'CountryCodes')
    __required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')
    _COLLECT_TYPE_VALUES = ('MONOSTATIC', 'BISTATIC')
    # descriptors
    CollectorName = _StringDescriptor('CollectorName', required=True, strict=DEFAULT_STRICT, docstring='The Collector Name.')
    IlluminatorName = _StringDescriptor(
        'IlluminatorName', required=False, strict=DEFAULT_STRICT, docstring='The Illuminator Name.')
    CoreName = _StringDescriptor('CoreName', required=True, strict=DEFAULT_STRICT, docstring='The Core Name.')
    CollectType = _StringEnumDescriptor(
        'CollectType', _COLLECT_TYPE_VALUES, required=False,
        docstring="The collect type, one of {}".format(_COLLECT_TYPE_VALUES))
    RadarMode = _SerializableDescriptor(
        'RadarMode', RadarModeType, required=True, strict=DEFAULT_STRICT, docstring='The Radar Mode')
    Classification = _StringDescriptor('Classification', required=True, strict=DEFAULT_STRICT, docstring='The Classification.')
    # list type descriptors
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, required=False, strict=DEFAULT_STRICT, tag='Parameter', docstring='The parameters list.')
    CountryCodes = _StringListDescriptor(
        'CountryCodes', required=False, strict=DEFAULT_STRICT, docstring="The country code list.")

    def __init__(self, **kwargs):
        """
        The Constructor.
        :param dict kwargs: the valid keys are
            ['CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
             'RadarMode', 'Classification', 'CountryCodes', 'Parameters'].
        While the required fields are ['CollectorName', 'CoreName', 'RadarMode', 'Classification'].
        """
        super(CollectionInfoType, self).__init__(**kwargs)


class ImageCreationType(Serializable):
    """
    The image creation data container.
    """
    __fields = ('Application', 'DateTime', 'Site', 'Profile')
    __required = ()
    # descriptors
    Application = _StringDescriptor('Application', required=False, strict=DEFAULT_STRICT, docstring='The Application')
    DateTime = _DateTimeDescriptor(
        'DateTime', required=False, strict=DEFAULT_STRICT, docstring='The Date/Time', numpy_datetime_units='us')
    Site = _StringDescriptor('Site', required=False, strict=DEFAULT_STRICT, docstring='The Site')
    Profile = _StringDescriptor('Profile', required=False, strict=DEFAULT_STRICT, docstring='The Profile')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['Application', 'DateTime', 'Site', 'Profile'], and none are required
        """
        super(ImageCreationType, self).__init__(**kwargs)


class ImageDataType(Serializable):
    """
    The image data container.
    """
    __child_tags = {'AmpTable': 'Amplitude'}
    __fields = ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel',
                'ValidData')
    __required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    __numeric_format = {'AmpTable': '0.8f'}  # TODO: precision for AmpTable?
    _PIXEL_TYPE_VALUES = ("RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I")
    # child class definition

    class ValidDataType(Serializable):
        """
        The valid data definition for the SICD image data.
        """
        __child_tags = {'Vertices': 'Vertex'}  # to account for the list type descriptor
        __fields = ('Vertices', 'size')
        __required = ('Vertices',)
        __set_as_attribute = ('size',)
        # descriptors
        Vertices = _SerializableArrayDescriptor(
            'Vertices', RowColvertexType, required=True, strict=DEFAULT_STRICT, tag='Vertex',
            docstring="A list of elements of type RowColvertexType.")

        def __init__(self, **kwargs):
            """
            The constructor.
            :param dict kwargs: the only valid key is 'Vertices'.
            """
            super(ImageDataType.ValidDataType, self).__init__(**kwargs)

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
                logging.warning(
                    'Required field Vertices only has length {} for class {}'.format(self.size,
                                                                                     self.__class__.__name__))
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

        # TODO: helper methods for functionality, again?

    # descriptors
    PixelType = _StringEnumDescriptor(
        'PixelType', _PIXEL_TYPE_VALUES, required=True, strict=True,
        docstring="""
        The PixelType attribute which specifies the interpretation of the file data:
            * `"RE32F_IM32F"` - a pixel is specified as `(real, imaginary)` each 32 bit floating point.
            * `"RE16I_IM16I"` - a pixel is specified as `(real, imaginary)` each a 16 bit (short) integer.
            * `"AMP8I_PHS8I"` - a pixel is specified as `(amplitude, phase)` each an 8 bit unsigned integer. The
                `amplitude` actually specifies the index into the `AmpTable` attribute. The `angle` is properly
                interpreted (in radians) as `theta = 2*pi*angle/256`.
        """)
    NumRows = _IntegerDescriptor('NumRows', required=True, strict=DEFAULT_STRICT, docstring='The number of Rows')
    NumCols = _IntegerDescriptor('NumCols', required=True, strict=DEFAULT_STRICT, docstring='The number of Columns')
    FirstRow = _IntegerDescriptor('FirstRow', required=True, strict=DEFAULT_STRICT, docstring='The first row')
    FirstCol = _IntegerDescriptor('FirstCol', required=True, strict=DEFAULT_STRICT, docstring='The first column')
    FullImage = _SerializableDescriptor(
        'FullImage', FullImageType, required=True, strict=DEFAULT_STRICT, docstring='The full image')
    SCPPixel = _SerializableDescriptor(
        'SCPPixel', RowColType, required=True, strict=DEFAULT_STRICT, docstring='The SCP Pixel')
    # TODO: better explanation of this metadata
    ValidData = _SerializableDescriptor(
        'ValidData', ValidDataType, required=False, strict=DEFAULT_STRICT, docstring='The valid data area')
    AmpTable = _FloatArrayDescriptor(
        'AmpTable', required=False, strict=DEFAULT_STRICT, child_tag='Amplitude',
        docstring="The Amplitude lookup table. This must be defined if PixelType == 'AMP8I_PHS8I'")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are
            ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow',
            'FirstCol', 'FullImage', 'SCPPixel', 'ValidData')
        Required are ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel'),
        and 'AmpTable' must conditionally be defined if PixelType == 'AMP8I_PHS8I'.
        """
        self._AmpTable = None
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


class GeoInfoType(Serializable):
    """
    The GeoInfo container.
    """
    __tag = 'GeoInfo'
    __child_tags = {'Descriptions': 'Desc'}
    __fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon')
    __required = ('name', )
    __set_as_attribute = ('name', )
    # descriptors
    name = _StringDescriptor('name', required=True, strict=True, docstring='The name.')
    Descriptions = _SerializableArrayDescriptor(
        'Descriptions', ParameterType, required=False, strict=DEFAULT_STRICT, tag='Desc', docstring='The descriptions.')
    Point = _SerializableDescriptor(
        'Point', LatLonRestrictionType, required=False, strict=DEFAULT_STRICT, docstring='A point.')
    Line = _SerializableDescriptor(
        'Line', LineType, required=False, strict=DEFAULT_STRICT, docstring='A line.')
    Polygon = _SerializableDescriptor(
        'Polygon', PolygonType, required=False, strict=DEFAULT_STRICT, docstring='A polygon.')
    # TODO: is the standard really self-referential here? I find that confusing.

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ('name', 'Descriptions', 'Point', 'Line', 'Polygon'), where
            the only required key is 'name' , and only one of 'Point', 'Line', or 'Polygon' should be present.
        """
        feature_count = 0
        for typ in ['Point', 'Line', 'Polygon']:
            if kwargs[typ] is not None:
                feature_count += 1
        if feature_count > 1:
            raise ValueError("Only one of 'Point', 'Line', or 'Polygon' can be defined.")
        super(GeoInfoType, self). __init__(**kwargs)


class GeoDataType(Serializable):
    """Container specifying the image coverage area in geographic coordinates."""
    __fields = ('EarthModel', 'SCP', 'ImageCorners', 'ValidData', 'GeoInfos')
    __required = ('EarthModel', 'SCP', 'ImageCorners')
    _EARTH_MODEL_VALUES = ('WGS_84', )
    # child class definitions

    class SCPType(Serializable):
        """The SCP container"""
        __tag = 'SCP'
        __fields = ('ECF', 'LLH')
        __required = __fields  # isn't this redundant?
        ECF = _SerializableDescriptor(
            'ECF', XYZType, tag='ECF', required=True, strict=DEFAULT_STRICT, docstring='The ECF coordinates.')
        LLH = _SerializableDescriptor(
            'LLH', LatLonHAERestrictionType, tag='LLH', required=True, strict=DEFAULT_STRICT, docstring='The WGS 84 coordinates.'
        )

        def __init__(self, **kwargs):
            """
            The constructor.
            :param dict kwargs: the keys are ['ECF', 'LLH'], all required.
            """
            super(GeoDataType.SCPType, self).__init__(**kwargs)

    class ImageCornersType(Serializable):
        """The image corners container. """
        __tag = 'ImageCorners'
        __fields = ('ICPs', )
        __child_tags = {'ICPs': 'ICP'}
        __required = __fields
        ICPs = _SerializableArrayDescriptor(
            'ICPs', LatLonCornerStringType, tag='ICP', required=True, strict=DEFAULT_STRICT,
            docstring='The image corner points.')

        def __init__(self, **kwargs):
            """
            The constructor
            :param dict kwargs: the only (required) key is 'ICPs'
            """
            super(GeoDataType.ImageCornersType, self).__init__(**kwargs)
            if self.ICPs is None or len(self.ICPs) != 4:
                raise ValueError('There sre not 4 image corner points defined.')

        def is_valid(self, recursive=False):  # type: (bool) -> bool
            valid = super(GeoDataType.ImageCornersType, self).is_valid(recursive=recursive)
            if self.ICPs is not None:
                if len(self.ICPs) != 4:
                    logging.warning('There must be 4 image corner points defined')
                    valid = False
                else:
                    indices = []
                    for entry in self.ICPs:
                        indices.append(entry.index)
                    iset = set(indices)
                    if not iset.issubset(LatLonCornerStringType._CORNER_VALUES) or len(iset) != 4:
                        logging.warning(
                            'The image corner data indices collection should match {}, '
                            'but we have {}'.format(LatLonCornerStringType._CORNER_VALUES, indices))
                        valid = False
            return valid

    # descriptors
    EarthModel = _StringEnumDescriptor(
        'EarthModel', _EARTH_MODEL_VALUES, required=True, strict=True, default_value='WGS_84',
        docstring='The Earth Model, which taks values in {}.'.format(_EARTH_MODEL_VALUES))
    SCP = _SerializableDescriptor(
        'SCP', SCPType, required=True, strict=DEFAULT_STRICT, tag='SCP', docstring='The Scene Center Point.')
    ImageCorners = _SerializableDescriptor(
        'ImageCorners', ImageCornersType, required=True, strict=DEFAULT_STRICT, tag='ImageCorners',
        docstring='The geographic image corner points')
    ValidData = _SerializableDescriptor(
        'ValidData', PolygonType, required=False, strict=DEFAULT_STRICT, tag='ValidData',
        docstring='Indicates the full image includes both valid data and some zero filled pixels.')
    GeoInfos = _SerializableArrayDescriptor(
        'GeoInfos', GeoInfoType, required=False, strict=DEFAULT_STRICT, tag='GeoInfo',
        docstring='Relevant geographic features.')


class DirParamType(Serializable):
    """The direction parameters container"""
    __fields = (
        'UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly',
        'WgtType', 'WgtFunct')
    __required = ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2')
    __numeric_format = {
        'SS': '0.8f', 'ImpRespWid': '0.8f', 'Sgn': '+d', 'ImpRespBW': '0.8f', 'KCtr': '0.8f',
        'DeltaK1': '0.8f', 'DeltaK2': '0.8f'}
    __child_tags = {'WgtFunct': 'Wgt'}
    # child class definitions

    class WgtTypeType(Serializable):
        """The weight type parameters"""
        __fields = ('WindowName', 'Parameters')
        __required = ('WindowName', )
        __child_tags = {'Parameters': 'Parameter'}
        WindowName = _StringDescriptor('WindowName', required=True, strict=DEFAULT_STRICT, docstring='The window name')
        Parameters = _SerializableArrayDescriptor(
            'Parameters', ParameterType, tag='Parameter', required=False, strict=DEFAULT_STRICT,
            docstring='The parameters list')

        def __init__(self, **kwargs):
            """
            The constructor.
            :param dict kwargs: the valid keys are ('WindowName', 'Parameters'), and 'WindowName' is required.
            """
            super(DirParamType.WgtTypeType, self).__init__(**kwargs)

    # descriptors
    UVectECF = _SerializableDescriptor(
        'UVectECF', XYZType, tag='UVectECF', required=True, strict=DEFAULT_STRICT,
        docstring='Unit vector in the increasing (row/col) direction (ECF) at the SCP pixel.')
    SS = _FloatDescriptor(
        'SS', required=True, strict=DEFAULT_STRICT,
        docstring='Sample spacing in the increasing (row/col) direction. Precise spacing at the SCP.')
    ImpRespWid = _FloatDescriptor(
        'ImpRespWid', required=True, strict=DEFAULT_STRICT,
        docstring='Half power impulse response width in the increasing (row/col) direction. Measured at the SCP.')
    Sgn = _IntegerEnumDescriptor(
        'Sgn', values=(1, -1), required=True, strict=DEFAULT_STRICT,
        docstring='sign for exponent in the DFT to transform the (row/col) dimension to spatial frequency dimension.')
    ImpRespBW = _FloatDescriptor(
        'ImpRespBW', required=True, strict=DEFAULT_STRICT,
        docstring='Spatial bandwidth in (row/col) used to form the impulse response in the (row/col) direction. '
                  'Measured at the center of support for the SCP.')
    KCtr = _FloatDescriptor(
        'KCtr', required=True, strict=DEFAULT_STRICT,
        docstring='Center spatial frequency in the given dimension. '
                  'Corresponds to the zero frequency of the DFT in the given (row/col) direction.')
    DeltaK1 = _FloatDescriptor(
        'DeltaK1', required=True, strict=DEFAULT_STRICT,
        docstring='Minimum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaK2 = _FloatDescriptor(
        'DeltaK2', required=True, strict=DEFAULT_STRICT,
        docstring='Maximum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaKCOAPoly = _SerializableDescriptor(
        'DeltaKCOAPoly', Poly2DType, tag='DeltaKCOAPoly', required=False, strict=DEFAULT_STRICT,
        docstring='Offset from KCtr of the center of support in the given (row/col) spatial frequency. '
                  'The polynomial is a function of image given (row/col) coordinate (variable 1) and '
                  'column coordinate (variable 2).')
    WgtType = _SerializableDescriptor(
        'WgtType', WgtTypeType, tag='WgtType', required=False, strict=DEFAULT_STRICT,
        docstring='Parameters describing aperture weighting type applied in the spatial frequency domain '
                  'to yield the impulse response in the given(row/col) direction.')
    WgtFunct = _FloatArrayDescriptor(
        'WgtFunct', child_tag='Wgt', required=False, strict=DEFAULT_STRICT,
        docstring='Sampled aperture amplitude weighting function applied to form the SCP impulse response '
                  'in the given (row/col) direction. Attribute “size” equals the number of weights')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are
            ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr',
            'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly', 'WgtType', 'WgtFunct'). The only optional keys are
        """
        super(DirParamType, self).__init__(**kwargs)

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.

        :param bool recursive: should we recursively check that child are also valid?
        :return bool: condition for validity of this element

        .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
            that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then
            this will result in an infinite loop.
        """

        valid = super(DirParamType, self).is_valid(recursive=recursive)
        if self.WgtFunct is not None:
            if self.WgtFunct.size < 2:
                logging.warning('The WgtFunct array has been provided, but there are fewer than 2 entries.')
                valid = False
        return valid


class GridType(Serializable):
    """Collection grid details container"""
    __fields = ('ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col')
    __required = __fields
    _IMAGE_PLANE_VALUES = ('SLANT', 'GROUND', 'OTHER')
    _TYPE_VALUES = ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    # descriptors
    ImagePlane = _StringEnumDescriptor(
        'ImagePlane', _IMAGE_PLANE_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring="The image plane. Possible values are {}".format(_IMAGE_PLANE_VALUES))
    Type = _StringEnumDescriptor(
        'Type', _TYPE_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring="""The possible values and meanings:
* 'RGAZIM' - Range & azimuth relative to the ARP at a reference time.
* 'RGZERO' - Range from ARP trajectory at zero Doppler and azimuth aligned with the strip being imaged.
* 'XRGYCR' - Orthogonal slant plane grid oriented range and cross range relative to the ARP at a reference time.
* 'XCTYAT' - Orthogonal slant plane grid with X oriented cross track.
* 'PLANE'  - Uniformly sampled in an arbitrary plane along directions U & V.""")
    TimeCOAPoly = _SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, required=True, strict=DEFAULT_STRICT,
        docstring="*Time of Center Of Aperture* as a polynomial function of image coordinates. The polynomial is a "
                  "function of image row coordinate (variable 1) and column coordinate (variable 2).")
    Row = _SerializableDescriptor(
        'Row', DirParamType, required=True, strict=DEFAULT_STRICT,
        docstring="Row direction parameters of type DirParamType")
    Col = _SerializableDescriptor(
        'Col', DirParamType, required=True, strict=DEFAULT_STRICT,
        docstring="Column direction parameters of type DirParamType")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are .
        """
        super(GridType, self).__init__(**kwargs)


##############
# TimelineType section


class IPPType(Serializable):
    """The Inter-Pulse Period parameters."""
    __fields = ('Sets', 'size')
    __required = ('Sets', )
    __child_tags = {'Sets': 'Set'}
    # child class

    class SetType(Serializable):
        """"""
        __tag = 'Set'
        __fields = ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index')
        __required = __fields
        __set_as_attribute = ('index',)
        # descriptors
        TStart = _FloatDescriptor(
            'TStart', required=True, strict=DEFAULT_STRICT,
            docstring='IPP start time relative to collection start time, i.e. offsets in seconds.')
        TEnd = _FloatDescriptor(
            'TEnd', required=True, strict=DEFAULT_STRICT,
            docstring='IPP end time relative to collection start time, i.e. offsets in seconds.')
        IPPStart = _IntegerDescriptor(
            'IPPStart', required=True, strict=True, docstring='Starting IPP index for the period described.')
        IPPEnd = _IntegerDescriptor(
            'IPPEnd', required=True, strict=True, docstring='Ending IPP index for the period described.')
        IPPPoly = _SerializableDescriptor(
            'IPPPoly', Poly1DType, required=True, strict=DEFAULT_STRICT,
            docstring='IPP index polynomial coefficients yield IPP index as a function of time for TStart to TEnd.')
        index = _IntegerDescriptor('index', required=True, strict=DEFAULT_STRICT, docstring='The index')

        def __init__(self, **kwargs):
            """The constructor.
            :param dict kwargs: the valid keys are ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index'),
            all required.
            """
            super(IPPType.SetType, self).__init__(**kwargs)

    # descriptors
    Sets = _SerializableArrayDescriptor(
        'Sets', SetType, required=True, strict=DEFAULT_STRICT, tag='Set', docstring='The set container.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key is 'Sets', which must be defined.
        """
        super(IPPType, self).__init__(**kwargs)

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.

        :param bool recursive: should we recursively check that child are also valid?
        :return bool: condition for validity of this element

        .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
            that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then
            this will result in an infinite loop.
        """

        valid = super(IPPType, self).is_valid(recursive=recursive)
        if self.Sets is not None and len(self.Sets) == 0:
            valid = False
            logging.warning('IPPType has an empty collection of Set objects.')
        return valid

    @property
    def size(self):  # type: () -> int
        """
        The size attribute.
        :return int:
        """
        return 0 if self.Sets is None else len(self.Sets)


class TimelineType(Serializable):
    """The details for the imaging collection timeline."""
    __fields = ('CollectStart', 'CollectDuration', 'IPP')
    __required = ('CollectStart', 'CollectDuration', )
    # descriptors
    CollectStart = _DateTimeDescriptor(
        'CollectStart', required=True, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. This will be an instance of numpy.datetime64. The default precision will '
                  'be microseconds.')
    CollectDuration = _FloatDescriptor(
        'CollectDuration', required=True, strict=DEFAULT_STRICT, docstring='The duration of the collection in seconds.')
    IPP = _SerializableDescriptor(
        'IPP', IPPType, required=False, strict=DEFAULT_STRICT, docstring="The Inter-Pulse Period (IPP) parameters.")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('CollectStart', 'CollectDuration', 'IPP'), and
        ('CollectStart', 'CollectDuration') are required.
        """
        super(TimelineType, self).__init__(**kwargs)


###################
# PositionType child section

class RcvAPCType(Serializable):
    """The Receive Aperture Phase Center polynomials collection container"""
    __fields = ('RcvAPCPolys', 'size')
    __required = ('RcvAPCPolys', )
    __child_tags = {'RcvAPCPolys': 'RcvAPCPoly'}

    # descriptors
    RcvAPCPolys = _SerializableArrayDescriptor(
        'RcvAPCPolys', XYZPolyAttributeType, required=True, strict=DEFAULT_STRICT, tag='RcvAPCPoly',
        docstring='Receive Aperture Phase Center polynomials. Each polynomial has output in ECF, and represents a '
                  'function of elapsed seconds since start of collection.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key is 'RcvAPCPolys', which must be defined.
        """
        super(RcvAPCType, self).__init__(**kwargs)

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.

        :param bool recursive: should we recursively check that child are also valid?
        :return bool: condition for validity of this element

        .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
            that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then
            this will result in an infinite loop.
        """

        valid = super(RcvAPCType, self).is_valid(recursive=recursive)
        if self.RcvAPCPolys is not None and len(self.RcvAPCPolys) == 0:
            valid = False
            logging.warning('RcvAPC has an empty collection of RcvAPCPoly objects.')
        return valid

    @property
    def size(self):  # type: () -> int
        """
        The size attribute.
        :return int:
        """
        return 0 if self.RcvAPCPolys is None else len(self.RcvAPCPolys)


class PositionType(Serializable):
    """The details for platform and ground reference positions as a function of time since collection start."""
    __fields = ('ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC')
    __required = ('ARPPoly',)
    # descriptors
    ARPPoly = _SerializableDescriptor(
        'ARPPoly', XYZPolyType, required=('ARPPoly' in __required), strict=DEFAULT_STRICT,
        docstring='Aperture Reference Point (ARP) position polynomial in ECF as a function of elapsed seconds '
                  'since start of collection.')
    GRPPoly = _SerializableDescriptor(
        'GRPPoly', XYZPolyType, required=('GRPPoly' in __required), strict=DEFAULT_STRICT,
        docstring='Ground Reference Point (GRP) position polynomial in ECF as a function of elapsed seconds '
                  'since start of collection.')
    TxAPCPoly = _SerializableDescriptor(
        'TxAPCPoly', XYZPolyType, required=('TxAPCPoly' in __required), strict=DEFAULT_STRICT,
        docstring='Transmit Aperture Phase Center (APC) position polynomial in ECF as a function of elapsed seconds '
                  'since start of collection.')
    RcvAPC = _SerializableDescriptor(
        'RcvAPC', RcvAPCType, required=('RcvAPC' in __required), strict=DEFAULT_STRICT,
        docstring="The Receive Aperture Phase Center polynomials collection container.")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC'), and only '
        ARPPoly' is required.
        """
        super(PositionType, self).__init__(**kwargs)


##################
# RadarCollectionType child classes


class WaveformParametersType(Serializable):
    """Transmit and receive demodulation waveform parameters."""
    __fields = (
        'TxPulseLength', 'TxRFBandwidth', 'TxFreqStart', 'TxFMRate', 'RcvDemodType', 'RcvWindowLength',
        'ADCSampleRate', 'RcvIFBandwidth', 'RcvFreqStart', 'RcvFMRate')
    __required = ()
    _DEMOD_TYPE_VALUES = ('STRETCH', 'CHIRP')
    # descriptors
    TxPulseLength = _FloatDescriptor(
        'TxPulseLength', required=('TxPulseLength' in __required), strict=DEFAULT_STRICT,
        docstring='Transmit pulse length in seconds.')
    TxRFBandwidth = _FloatDescriptor(
        'TxRFBandwidth', required=('TxRFBandwidth' in __required), strict=DEFAULT_STRICT,
        docstring='Transmit RF bandwidth of the transmit pulse in Hz.')
    TxFreqStart = _FloatDescriptor(
        'TxFreqStart', required=('TxFreqStart' in __required), strict=DEFAULT_STRICT,
        docstring='Transmit Start frequency for Linear FM waveform in Hz, may be relative to reference frequency.')
    TxFMRate = _FloatDescriptor(
        'TxFMRate', required=('TxFMRate' in __required), strict=DEFAULT_STRICT,
        docstring='Transmit FM rate for Linear FM waveform in Hz/second.')
    RcvWindowLength = _FloatDescriptor(
        'RcvWindowLength', required=('RcvWindowLength' in __required), strict=DEFAULT_STRICT,
        docstring='Receive window duration in seconds.')
    ADCSampleRate = _FloatDescriptor(
        'ADCSampleRate', required=('ADCSampleRate' in __required), strict=DEFAULT_STRICT,
        docstring='Analog-to-Digital Converter sampling rate in samples/second.')
    RcvIFBandwidth = _FloatDescriptor(
        'RcvIFBandwidth', required=('RcvIFBandwidth' in __required), strict=DEFAULT_STRICT,
        docstring='Receive IF bandwidth in Hz.')
    RcvFreqStart = _FloatDescriptor(
        'RcvFreqStart', required=('RcvFreqStart' in __required), strict=DEFAULT_STRICT,
        docstring='Receive demodulation start frequency in Hz, may be relative to reference frequency.')
    RcvFMRate = _FloatDescriptor(
        'RcvFMRate', required=('RcvFMRate' in __required), strict=DEFAULT_STRICT,
        docstring='Receive FM rate. Should be 0 if RcvDemodType = "CHIRP".')
    RcvDemodType = _StringEnumDescriptor(
        'RcvDemodType', _DEMOD_TYPE_VALUES, required=('RcvFMRate' in __required), strict=DEFAULT_STRICT,
        docstring="""Receive demodulation used when Linear FM waveform is used on transmit.
* 'STRETCH' - De-ramp on Receive demodulation.
* 'CHIRP'   - No De-ramp On Receive demodulation""")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are , and required.
        """
        super(WaveformParametersType, self).__init__(**kwargs)
        if self.RcvDemodType == 'CHIRP':
            self.RcvFMRate = 0

    def is_valid(self, recursive=False):  # type: (bool) -> bool
        """
        Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.

        :param bool recursive: should we recursively check that child are also valid?
        :return bool: condition for validity of this element

        .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
            that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then
            this will result in an infinite loop.
        """

        valid = super(WaveformParametersType, self).is_valid(recursive=recursive)
        if self.RcvDemodType == 'CHIRP' and self.RcvFMRate != 0:
            logging.warning('In WaveformParameters, we have RcvDemodType == "CHIRP" and self.RcvFMRate non-zero.')

        return valid


class TxStepType(Serializable):
    """Transmit sequence step details"""
    __fields = ('WFIndex', 'TxPolarization', 'index')
    __required = ('index', )
    _POLARIZATION2_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER')
    # descriptors
    WFIndex = _IntegerDescriptor(
        'WFIndex', required=True, strict=DEFAULT_STRICT, docstring='The waveform number for this step.')
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION2_VALUES, required=False, strict=DEFAULT_STRICT,
        docstring='Transmit signal polarization for this step.')
    index = _IntegerDescriptor('index', required=True, strict=DEFAULT_STRICT, docstring='The step index')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('WFIndex', 'TxPolarization', 'index'), and 'index' is required.
        """
        super(TxStepType, self).__init__(**kwargs)


class ChanParametersType(Serializable):
    """Transmit sequence step details"""
    __fields = ('TxRcvPolarization', 'RcvAPCIndex', 'index')
    __required = ('TxRcvPolarization', 'index', )
    _DUAL_POLARIZATION_VALUES = (
        'V:V', 'V:H', 'H:V', 'H:H', 'RHC:RHC', 'RHC:LHC', 'LHC:RHC', 'LHC:LHC', 'OTHER', 'UNKNOWN')
    # descriptors
    TxRcvPolarization = _StringEnumDescriptor(
        'TxRcvPolarization', _DUAL_POLARIZATION_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring='Combined Transmit and Receive signal polarization for the channel.')

    RcvAPCIndex = _IntegerDescriptor(
        'RcvAPCIndex', required=True, strict=DEFAULT_STRICT,
        docstring='Index of the Receive Aperture Phase Center (Rcv APC). Only include if Receive APC position '
                  'polynomial(s) are included.')
    index = _IntegerDescriptor('index', required=True, strict=DEFAULT_STRICT, docstring='The parameter index')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('TxRcvPolarization', 'RcvAPCIndex', 'index'), and
        ('TxRcvPolarization', 'index') are required.
        """
        super(ChanParametersType, self).__init__(**kwargs)


# AreaType is a bit of a show
#   - CornerType
#   - PlaneType
#       - RefPtType
#       - XDirType
#       - YDirType
#       - SegmentListType (dumb)
#           - SegmentType


# class ShellType(Serializable):
#     """"""
#     __fields = ()
#     __required = ()
#
#     # child class definitions
#     # descriptors
#
#     def __init__(self, **kwargs):
#         """The constructor.
#         :param dict kwargs: the valid keys are , and required.
#         """
#         super(ShellType, self).__init__(**kwargs)
#
#     def is_valid(self, recursive=False):  # type: (bool) -> bool
#         """
#         Returns the validity of this object according to the schema. This is done by inspecting that all required
#         fields (i.e. entries of `__required`) are not `None`.
#
#         :param bool recursive: should we recursively check that child are also valid?
#         :return bool: condition for validity of this element
#
#         .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
#             that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then
#             this will result in an infinite loop.
#         """
#
#         valid = super(ShellType, self).is_valid(recursive=recursive)
#         return valid





# class ShellType(Serializable):
#     """"""
#     __fields = ()
#     __required = ()
#
#     # child class definitions
#     # descriptors
#
#     def __init__(self, **kwargs):
#         """The constructor.
#         :param dict kwargs: the valid keys are , and required.
#         """
#         super(ShellType, self).__init__(**kwargs)
#
#     def is_valid(self, recursive=False):  # type: (bool) -> bool
#         """
#         Returns the validity of this object according to the schema. This is done by inspecting that all required
#         fields (i.e. entries of `__required`) are not `None`.
#
#         :param bool recursive: should we recursively check that child are also valid?
#         :return bool: condition for validity of this element
#
#         .. Note: This DOES NOT recursively check if each attribute is itself valid, unless `recursive=True`. Note
#             that if a circular dependence is introduced at any point in the SICD standard (extremely unlikely) then
#             this will result in an infinite loop.
#         """
#
#         valid = super(ShellType, self).is_valid(recursive=recursive)
#         return valid
