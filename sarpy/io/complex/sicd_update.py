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
# descriptor (i.e. reusable properties) definition


class _StringDescriptor(object):
    """A descriptor for string type properties"""

    def __init__(self, name, docstring=None, required=False, strict=False):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "String type field"

    def __get__(self, instance, owner):  # type: (object, object) -> str
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

    def __set__(self, instance, value):  # type: (object, str) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        # Do we have to do anything to ensure instance is Hashable? Properly define __eq__ and __hash__ or something?
        if isinstance(value, str):
            self.data[instance] = value
        elif value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))


class _StringListDescriptor(object):
    """A descriptor for properties of a assumed to be an array type item for specified extension of Serializable"""

    def __init__(self, name, docstring=None, required=False, strict=False):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "list of strings type field"

    def __get__(self, instance, owner):  # type: (object, object) -> list
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return list: the return value
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
        :return None:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'Field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        elif isinstance(value, str):
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


class _StringEnumDescriptor(object):
    """A descriptor for enumerated string type properties"""

    def __init__(self, name, values, docstring=None, required=False, strict=False):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        self.values = values
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "Enumerated string type field"

    def __get__(self, instance, owner):  # type: (object, object) -> str
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

    def __set__(self, instance, value):  # type: (object, str) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

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
        elif value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = None
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))


class _IntegerDescriptor(object):
    """A descriptor for integer type properties"""

    def __init__(self, name, docstring=None, required=False, strict=False):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "Integer type field"

    def __get__(self, instance, owner):  # type: (object, object) -> int
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return int: the return value
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

    def __set__(self, instance, value):  # type: (object, int) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        else:
            self.data[instance] = int(value)


class _FloatDescriptor(object):
    """A descriptor for float type properties"""

    def __init__(self, name, docstring=None, required=False, strict=False):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "Float type field"

    def __get__(self, instance, owner):  # type: (object, object) -> float
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return float: the return value
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

    def __set__(self, instance, value):  # type: (object, float) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        else:
            self.data[instance] = float(value)


class _DateTimeDescriptor(object):
    """A descriptor for date time type properties"""

    def __init__(self, name, docstring=None, required=False, strict=False):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "Float type field"

    def __get__(self, instance, owner):  # type: (object, object) -> numpy.datetime64
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return numpy.datetime64: the return value
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

    def __set__(self, instance, value):  # type: (object, Union[date, datetime, str, int, numpy.datetime64]) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        elif isinstance(value, numpy.datetime64):
            self.data[instance] = value
        else:
            self.data[instance] = numpy.datetime64(value, 'us')  # let's default to microsecond precision


class _FloatModularDescriptor(object):
    """A descriptor for float type properties"""

    def __init__(self, name, limit, docstring=None, required=False, strict=False):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        self.limit = limit
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "Float type field"

    def __get__(self, instance, owner):  # type: (object, object) -> float
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return float: the return value
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

    def __set__(self, instance, value):  # type: (object, float) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        else:
            val = float(value)  # do modular arthimatic - so good
            val = (val % (2*self.limit))  # NB: % and * have same precedence, so it can be super dumb
            self.data[instance] = val if val <= self.limit else val - 2*self.limit


class _SerializableDescriptor(object):
    """A descriptor for properties of a specified type assumed to be an extension of Serializable"""

    def __init__(self, name, theType, docstring=None, required=False, strict=False, tag=None):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        self.theType = theType
        self.tag = tag
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "{} type field".format(theType.__class__.__name__)

    def __get__(self, instance, owner):  # type: (object, object) -> Serializable
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return Serializable: the return value
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
        :return None:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'Field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        elif isinstance(value, self.theType):
            self.data[instance] = value
        elif isinstance(value, dict):
            self.data[instance] = self.theType.from_dict(value).modify_tag(self.tag)
        elif isinstance(value, minidom.Element):
            self.data[instance] = self.theType.from_node(value).modify_tag(self.tag)
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _SerializableArrayDescriptor(object):
    """A descriptor for properties of a assumed to be an array type item for specified extension of Serializable"""

    def __init__(self, name, theType, hasIndex=False, docstring=None, required=False, strict=False, tag=None):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.strict = strict
        self.theType = theType
        self.hasIndex = hasIndex
        self.tag = tag
        if docstring is not None:
            self.__doc__ = docstring
        else:
            self.__doc__ = "{} type field".format(theType.__class__.__name__)

    def __get__(self, instance, owner):  # type: (object, object) -> list
        """
        The getter.
        :param instance: the calling class instance
        :param owner: the type of the class - that is, the actual object to which this descriptor is assigned
        :return list: the return value
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
        :return None:
        """

        if value is None:
            if self.strict:
                raise ValueError(
                    'Field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = value
        elif isinstance(value, self.theType):
            self.data[instance] = [value, ]
        elif isinstance(value, minidom.Element):
            self.data[instance] = [self.theType.from_node(value).modify_tag(self.tag), ]
        elif isinstance(value, minidom.NodeList):
            new_value = []
            if self.hasIndex:
                for node in value:
                    new_value.append((int(node.getAttribute('index')), self.theType.from_node(node).modify_tag(self.tag)))
                self.data[instance] = [entry for ind, entry in sorted(new_value, key=lambda x: x[0])]
            else:
                for node in value:
                    new_value.append(self.theType.from_node(node).modify_tag(self.tag))
                self.data[instance] = value
        elif isinstance(value, list):
            if len(value) == 0:
                self.data[instance] = value
            elif isinstance(value[0], self.theType):
                self.data[instance] = value
            elif isinstance(value[0], dict):
                # NB: charming errors are possible if something stupid has been done.
                self.data[instance] = [self.theType.from_dict(node).modify_tag(self.tag) for node in value]
            elif isinstance(value[0], minidom.Element):
                self.data[instance] = [self.theType.from_node(node).modify_tag(self.tag) for node in value]
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
    __tags = {}  # only needed for children where attribute_name != tag_name
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
        :return Serializable: returns a reference to
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
            if isinstance(val, Serializable):
                good = val.is_valid(recursive=recursive)
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

            tagname = cls.__tags[attribute] if attribute in cls.__tags else attribute
            if attribute in cls.__set_as_attribute:
                val = node.getAttribute(tagname)
                kwargs[attribute] = None if len(val) == 0 else val
            else:
                pnodes = [entry for entry in node.getElementsByTagName(tagname) if entry.parentNode == node]
                if len(pnodes) == 1:
                    kwargs[attribute] = pnodes[0]
                elif len(pnodes) > 1:
                    # this is for list type attributes. Probably should have an entry in __tags.
                    kwargs[attribute] = pnodes
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

        def serializeChild(value, attribute, tag, fmt):
            if isinstance(value, Serializable):
                value.to_node(doc, tag=tag, par=nod, strict=strict)
            elif isinstance(value, str):  # TODO: MEDIUM - unicode issues?
                _create_text_node(doc, tag, value, par=nod)
            elif isinstance(value, int) or isinstance(value, float):
                if fmt is None:
                    _create_text_node(doc, attribute, str(value), par=nod)
                else:
                    _create_text_node(doc, attribute, fmt.format(value), par=nod)
            elif isinstance(value, complex):
                raise NotImplementedError  # TODO: let's figure it out
            elif isinstance(value, date):
                _create_text_node(doc, attribute, value.isoformat(), par=nod)
            elif isinstance(value, datetime):
                _create_text_node(doc, attribute, value.isoformat(sep='T'), par=nod)
                # TODO: LOW - this can be error prone with the naive vs zoned issue.
                #   Discourage using this type?
            elif isinstance(value, numpy.datetime64):
                _create_text_node(doc, attribute, str(value), par=nod)
            elif isinstance(value, list) or isinstance(value, tuple):
                for val in value:
                    serializeChild(val, tag, fmt)
            elif isinstance(value, numpy.ndarray):
                raise NotImplementedError  # TODO: still need to figure out
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
            tag = self.__tags.get(attribute, attribute)
            fmt = self.__numeric_format(attribute, None)

            if attribute in self.__set_as_attribute:
                if fmt is None:
                    nod.setAttribute(tag, str(value))
                else:
                    nod.setAttribute(tag, fmt.format(value))
            else:
                serializeChild(value, attribute, tag, fmt)

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

class PlainValueType(Serializable):
    # This isn't actually part of the SICD standard, it's just a bottom level building block.

    __fields = ('value', )
    __required = __fields
    value = _StringDescriptor('value', required=True, strict=True, docstring='The value')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: one required key - 'value'
        :param kwargs:
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

        # we have to short-circuit the call here, because this is a really primitive element
        if tag is None:
            tag = self.__tag
        node = _create_text_node(doc, tag, self.value, par=par)
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
        """
        The constructor.
        :param dict kwargs: valid keys are ['name', 'value'], all required.
        :param kwargs:
        """
        super(ParameterType, self).__init__(**kwargs)


class XYZType(Serializable):
    __slots__ = ()  # prevent adhoc field definition, for speed and safety.
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    __numeric_format = {'X': '0.8f', 'Y': '0.8f', 'Z': '0.8f'}  # TODO: desired precision? This is usually meters?
    X = _FloatDescriptor(
        'X', required=True, strict=False, docstring='The X attribute. Assumed to ECF or other, similar coordinates.')
    Y = _FloatDescriptor(
        'Y', required=True, strict=False, docstring='The Y attribute. Assumed to ECF or other, similar coordinates.')
    Z = _FloatDescriptor(
        'Z', required=True, strict=False, docstring='The Z attribute. Assumed to ECF or other, similar coordinates.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['X', 'Y', 'Z'], all required.
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
    __slots__ = ()
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    Lat = _FloatDescriptor(
        'Lat', required=True, strict=False, docstring='The Latitude attribute. Assumed to be WGS 84 coordinates.')
    Lon = _FloatDescriptor(
        'Lon', required=True, strict=False, docstring='The Longitude attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon'], all required.
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
    __slots__ = ()
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, required=True, strict=False,
        docstring='The Latitude attribute. Assumed to be WGS 84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, required=True, strict=False,
        docstring='The Longitude attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon'], all required.
        """

        super(LatLonRestrictionType, self).__init__(**kwargs)


class LatLonHAEType(LatLonType):
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?
    HAE = _FloatDescriptor(
        'HAE', required=True, strict=False,
        docstring='The Height Above Ellipsoid (in meters) attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon', 'HAE'], all required.
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
        'Lat', 90.0, required=True, strict=False,
        docstring='The Latitude attribute. Assumed to be WGS 84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, required=True, strict=False,
        docstring='The Longitude attribute. Assumed to be WGS 84 coordinates.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys are ['Lat', 'Lon', 'HAE'], all required.
        """

        super(LatLonHAERestrictionType, self).__init__(**kwargs)


class RowColType(Serializable):
    __fields = ('Row', 'Col')
    __required = __fields
    Row = _IntegerDescriptor(
        'Row', required=True, strict=False, docstring='The (integer valued) Row attribute.')
    Col = _IntegerDescriptor(
        'Col', required=True, strict=False, docstring='The (integer valued) Column attribute.')

    def __init__(self, **kwargs):
        """
        The RowColType constructor.
        :param dict kwargs: valid keys are ['Row', 'Col'], all required.
        """
        super(RowColType, self).__init__(**kwargs)


class RowColvertexType(RowColType):
    __fields = ('Row', 'Col', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    index = _IntegerDescriptor(
        'index', required=True, strict=False, docstring='The (integer valued) index attribute.')

    def __init__(self, **kwargs):
        """
        The RowColvertexType constructor.
        :param dict kwargs: valid keys are ['Row', 'Col', 'index'], all required.
        """
        super(RowColvertexType, self).__init__(**kwargs)


class PolyCoef1DType(Serializable):
    __fields = ('value', 'exponent1')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    __set_as_attribute = ('exponent1', )
    value = _FloatDescriptor(
        'value', required=True, strict=False, docstring='The (coefficient) value attribute.')
    exponent1 = _IntegerDescriptor(
        'exponent1', required=True, strict=False, docstring='The (power) exponent1 attribute.')

    def __init__(self, **kwargs):
        """
        This represents a single monomial term of the form - `value * x^{exponent1}`
        :param dict kwargs: valid keys are ['value', 'exponent1'], all required.
        """
        super(PolyCoef1DType, self).__init__(**kwargs)

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

    __fields = ('value', 'exponent1', 'exponent2')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    __set_as_attribute = ('exponent1', 'exponent2')
    value = _FloatDescriptor(
        'value', required=True, strict=False, docstring='The (coefficient) value attribute.')
    exponent1 = _IntegerDescriptor(
        'exponent1', required=True, strict=False, docstring='The (power) exponent1 attribute.')
    exponent2 = _IntegerDescriptor(
        'exponent2', required=True, strict=False, docstring='The (power) exponent2 attribute.')

    def __init__(self, **kwargs):
        """
        The constructor. This class represents the monomial - `value * x^{exponent1} * y^{exponent2}`.
        :param dict kwargs: valid keys are ['value', 'exponent1', 'exponent2'], all required.
        """
        super(PolyCoef2DType, self).__init__(**kwargs)

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
    __fields = ('coefs', 'order1')
    __required = ('coefs', )
    __tags = {'coefs': 'Coef'}
    __set_as_attribute = ('order1', )
    coefs = _SerializableArrayDescriptor(
        'coefs', PolyCoef1DType, hasIndex=False, required=True, strict=False, tag='Coef',
        docstring='the collection of PolyCoef1DType monomial terms.')

    def __init__(self, **kwargs):
        """
        The constructor. Represents a one-variable polynomial, defined as the sum of monomial terms given in `coefs`.
        :param dict kwargs: valid key is 'coefs'.
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
    __fields = ('Coefs', 'order1', 'order2')
    __required = ('Coefs', )
    __tags = {'Coefs': 'Coef'}
    __set_as_attribute = ('order1', 'order2')
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef2DType, hasIndex=False, required=True, strict=False, tag='Coef',
        docstring='the collection of PolyCoef2DType monomial terms.')

    def __init__(self, **kwargs):
        """
        The constructor. Represents a two variable polynomial, defined as the sum of monomial terms given in `coefs`.
        :param dict kwargs: valid key is 'coefs'.
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
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    X = _SerializableDescriptor(
        'X', Poly1DType, required=True, strict=False, docstring='The X polynomial.')
    Y = _SerializableDescriptor(
        'Y', Poly1DType, required=True, strict=False, docstring='The Y polynomial.')
    Z = _SerializableDescriptor(
        'Z', Poly1DType, required=True, strict=False, docstring='The Z polynomial.')
    # TODO: a better description would be good here

    def __init__(self, **kwargs):
        """
        The constructor. Provides single variable polynomials for each of `X`, `Y`, and `Z`.
        :param dict kwargs: valid keys are ['X', 'Y', 'Z'], all required.
        """
        super(XYZPolyType, self).__init__(**kwargs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyAttributeType(XYZPolyType):
    __fields = ('X', 'Y', 'Z', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    index = _IntegerDescriptor(
        'index', required=True, strict=False, docstring='The (array) index value.')

    def __init__(self, **kwargs):
        """
        The constructor. This class extension enables making an array/list of XYZPolyType instances.
        :param dict kwargs: valid keys are in ['X', 'Y', 'Z', 'index'], all required.
        """
        super(XYZPolyAttributeType, self).__init__(**kwargs)


class GainPhasePolyType(Serializable):
    __fields = ('GainPoly', 'PhasePoly')
    __required = __fields
    GainPoly = _SerializableDescriptor(
        'GainPoly', Poly2DType, required=True, strict=False,
        docstring='The Gain (two variable) Polygon.')
    PhasePoly = _SerializableDescriptor(
        'GainPhasePoly', Poly2DType, required=True, strict=False,
        docstring='The Phase (two variable) Polygon.')
    # TODO: improve these docstrings

    def __init__(self, **kwargs):
        """
        The constructor. The class is a container for the Gain and Phase Polygon definitions.
        :param dict kwargs: valid keys are ['GainPoly', 'PhasePoly'], all required.
        """
        super(GainPhasePolyType, self).__init__(**kwargs)


class LineType(Serializable):
    __fields = ('Endpoints', 'size')
    __required = ('Endpoints', )
    __tags = {'Endpoints': 'Endpoint'}
    __set_as_attribute = ('size', )
    Endpoints = _SerializableArrayDescriptor(
        'Endpoints', LatLonType, hasIndex=True, required=True, strict=False, tag='Endpoint',
        docstring="A list of elements of type LatLonType. This isn't directly part of the SICD standard, and just "
                  "represents an intermediate convenience object.")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the only valid key is 'Endpoints'.
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

        node = super(LineType, self).to_node(
            doc, tag=tag, par=par, strict=strict, exclude=exclude+('Endpoints', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Endpoints):
            entry.to_node(doc, par=node, tag='EndPoint', strict=strict, exclude=()).setAttribute('index', str(i))
        return node


class PolygonType(Serializable):
    __fields = ('Vertices', 'size')
    __required = ('Vertices', )
    __tags = {'Vertices': 'Vertex'}
    Vertices = _SerializableArrayDescriptor(
        'Vertices', LatLonRestrictionType, hasIndex=True, required=True, strict=False, tag='Vertex',
        docstring="A list of elements of type LatLonRestrictionType. This isn't directly part of the SICD standard, "
                  "and just represents an intermediate convenience object.")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the only valid key is 'Vertices'.
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

        node = super(PolygonType, self).to_node(
            doc, tag=tag, par=par, strict=strict, exclude=exclude+('Vertices', 'size'))
        node.setAttribute('size', str(self.size))
        for i, entry in enumerate(self.Vertices):
            entry.to_node(doc, par=node, tag='Vertex', strict=strict, exclude=())\
                .setAttribute('index', str(i))
        return node


class ErrorDecorrFuncType(Serializable):
    __fields = ('CorrCoefZero', 'DecorrRate')
    __required = __fields
    __numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '0.8f'}  # TODO: desired precision?
    CorrCoefZero = _FloatDescriptor(
        'CorrCoefZero', required=True, strict=False, docstring='The CorrCoefZero attribute.')
    DecorrRate = _FloatDescriptor(
        'DecorrRate', required=True, strict=False, docstring='The DecorrRate attribute.')
    # TODO: Improve the description here.

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['CorrCoefZero', 'DecorrRate'], all required.
        """
        super(ErrorDecorrFuncType, self).__init__(**kwargs)

    # TODO: HIGH - this is supposed to be a "function". We should implement the functionality here.


class RadarModeType(Serializable):
    # Really just subordinate to CollectionInfo
    __tag = 'RadarMode'
    __fields = ('ModeType', 'ModeId')
    __required = ('ModeType', )
    ModeId = _StringDescriptor('ModeId', required=False, docstring='The Mode Id.')
    ModeType = _StringEnumDescriptor(
        'ModeType', ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP'), required=True, strict=False,
        docstring="The ModeType, which will be one of ['SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP'].")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['ModeType', 'ModeId'], and 'ModeType' is required.
        """
        super(RadarModeType, self).__init__(**kwargs)


class FullImageType(Serializable):
    __tag = 'FullImage'
    __fields = ('NumRows', 'NumCols')
    __required = __fields
    NumRows = _IntegerDescriptor('NumRows', required=True, strict=False, docstring='The number of rows.')
    NumCols = _IntegerDescriptor('NumCols', required=True, strict=False, docstring='The number of columns.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['NumRows', 'NumCols'], all required.
        """
        super(FullImageType, self).__init__(**kwargs)


class ValidDataType(Serializable):
    __tags = {'Vertices': 'Vertex'}  # to account for the list type descriptor
    __fields = ('Vertices', 'size')
    __required = ('Vertices', )
    __set_as_attribute = ('size', )
    Vertices = _SerializableArrayDescriptor(
        'Vertices', RowColvertexType, hasIndex=True, required=True, strict=False, tag='Vertex',
        docstring="A list of elements of type RowColvertexType.")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the only valid key is 'Vertices'.
        """
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


# TODO: corner type mumbo-jumbo


# direct building blocks for SICD


class CollectionInfoType(Serializable):
    __tag = 'CollectionInfo'
    __tags = {'Parameters': 'Parameter', 'CountryCode': 'CountryCode'}  # these list type ones are hand-jammed
    __fields = (
        'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
        'RadarMode', 'Classification', 'Parameters', 'CountryCodes')
    __required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')
    # the descriptors
    CollectorName = _StringDescriptor('CollectorName', required=True, strict=False, docstring='The Collector Name.')
    IlluminatorName = _StringDescriptor(
        'IlluminatorName', required=False, strict=False, docstring='The Illuminator Name.')
    CoreName = _StringDescriptor('CoreName', required=True, strict=False, docstring='The Core Name.')
    CollectType = _StringEnumDescriptor(
        'CollectType', ('MONOSTATIC', 'BISTATIC'), required=False,
        docstring="The collect type, one of ('MONOSTATIC', 'BISTATIC')")
    RadarMode = _SerializableDescriptor(
        'RadarMode', RadarModeType, required=True, strict=False, docstring='The Radar Mode')
    Classification = _StringDescriptor('Classification', required=True, strict=False, docstring='The Classification.')
    # list type descriptors
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, required=False, strict=False, tag='Parameter', docstring='The parameters list.')
    CountryCodes = _StringListDescriptor(
        'CountryCodes', )

    def __init__(self, **kwargs):
        """
        The Constructor.
        :param dict kwargs: the valid keys are [
            'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
            'RadarMode', 'Classification', 'CountryCodes', 'Parameters'].
            While the required fields are ['CollectorName', 'CoreName', 'RadarMode', 'Classification'].
        """
        super(CollectionInfoType, self).__init__(**kwargs)


class ImageCreationType(Serializable):
    __fields = ('Application', 'DateTime', 'Site', 'Profile')
    __required = ()
    Application = _StringDescriptor('Application', required=False, strict=False, docstring='The Application')
    DateTime = _DateTimeDescriptor('DateTime', required=False, strict=False, docstring='The Date/Time')
    Site = _StringDescriptor('Site', required=False, strict=False, docstring='The Site')
    Profile = _StringDescriptor('Profile', required=False, strict=False, docstring='The Profile')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['Application', 'DateTime', 'Site', 'Profile'], and none are required
        """
        super(ImageCreationType, self).__init__(**kwargs)


class ImageDataType(Serializable):
    __slots__ = ('_AmpTable', )
    __fields = ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel',
                'ValidData')
    __required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    __numeric_format = {'AmpTable': '0.8f'}  # TODO: precision for AmpTable?
    PixelType = _StringEnumDescriptor(
        'PixelType', ("RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I"), required=True, strict=True,
        docstring="""
        The PixelType attribute which specifies the interpretation of the file data:
            * `"RE32F_IM32F"` - a pixel is specified as `(real, imaginary)` each 32 bit floating point.
            * `"RE16I_IM16I"` - a pixel is specified as `(real, imaginary)` each a 16 bit (short) integer.
            * `"AMP8I_PHS8I"` - a pixel is specified as `(amplitude, phase)` each an 8 bit unsigned integer. The
                `amplitude` actually specifies the index into the `AmpTable` attribute. The `angle` is properly
                interpreted (in radians) as `theta = 2*pi*angle/256`.
        """)
    NumRows = _IntegerDescriptor('NumRows', required=True, strict=False, docstring='The number of Rows')
    NumCols = _IntegerDescriptor('NumCols', required=True, strict=False, docstring='The number of Columns')
    FirstRow = _IntegerDescriptor('FirstRow', required=True, strict=False, docstring='The first row')
    FirstCol = _IntegerDescriptor('FirstCol', required=True, strict=False, docstring='The first column')
    FullImage = _SerializableDescriptor('FullImage', FullImageType, required=True, strict=False, docstring='The full image')
    SCPPixel = _SerializableDescriptor('SCPPixel', RowColType, required=True, strict=False, docstring='The SCP Pixel')
    ValidData = _SerializableDescriptor('ValidData', ValidDataType, required=False, strict=False, docstring='The valid data area')
    # TODO: better explanation of this metadata

    def __init__(self, **kwargs):
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

        # TODO: BROKEN!!! set up the Descriptor for this? If we need to? Setup serialization otherwise.
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

