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

        # NOTE: This is intended to handle this case for every extension of this class. Hence the boolean return,
        # which extensions SHOULD NOT implement. This is merely to enable following DRY principles.
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
            # from user or json deserialization
            self.data[instance] = value
        elif isinstance(value, minidom.Element):
            # from XML deserialization
            self.data[instance] = _get_node_value(value)
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))


class _StringListDescriptor(_BasicDescriptor):
    """A descriptor for properties of a assumed to be an array type item for specified extension of Serializable"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, minimum_length=0, maximum_length=2**32, docstring=None):
        super(_StringListDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Field {} of class {} is a string list of size {}, and must have length at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Field {} of class {} is a string list of size {}, and must have length nu greater than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            self.data[instance] = new_value

        if super(_StringListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, str):
            set_value([value, ])
        elif isinstance(value, minidom.Element):
            set_value([_get_node_value(value), ])
        elif isinstance(value, minidom.NodeList):
            set_value([_get_node_value(nod) for nod in value])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], str):
                set_value(value)
            elif isinstance(value[0], minidom.Element):
                set_value([_get_node_value(nod) for nod in value])
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
        elif isinstance(value, minidom.Element):
            val = _get_node_value(value).upper()
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))

        if val in self.values:
            self.data[instance] = val
        else:
            if self.strict:
                raise ValueError(
                    'Received value {} for field {} of class {}, but values ARE REQUIRED to be one of {}'.format(
                        value, self.name, instance.__class__.__name__, self.values))
            else:
                logging.warning(
                    'Received value {} for field {} of class {}, but values ARE REQUIRED to be one of {}. '.format(
                        value, self.name, instance.__class__.__name__, self.values))
            self.data[instance] = val


class _BooleanDescriptor(_BasicDescriptor):
    """A descriptor for integer type properties"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, docstring=None):
        super(_BooleanDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):  # type: (object, int) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_BooleanDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
            # from XML deserialization
            bv = self.parse_string(instance, _get_node_value(value))
        elif isinstance(value, bool):
            bv = value
        elif isinstance(value, int):
            bv = bool(value)
        elif isinstance(value, str):
            bv = self.parse_string(instance, value)
        else:
            raise ValueError('Boolean field {} of class {} cannot assign from type {}.'.format(
                self.name, instance.__class__.__name__, type(value)))
        self.data[instance] = bv

    def parse_string(self, instance, value):  # type: (object, str) -> bool
        if value.lower() in ['0', 'false']:
            return False
        elif value.lower() in ['1', 'true']:
            return True
        else:
            raise ValueError(
                'Boolean field {} of class {} cannot assign from string value {}. It must be one of '
                '["0", "false", "1", "true"]'.format(self.name, instance.__class__.__name__, value))


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

        if isinstance(value, minidom.Element):
            # from XML deserialization
            iv = int(_get_node_value(value))
        else:
            # user or json deserialization
            iv = int(value)

        if self.bounds is None or (self.bounds[0] <= iv <= self.bounds[1]):
            self.data[instance] = iv
        else:
            msg = 'Field {} of class {} is required by standard to take value between {}.'.format(
                self.name, instance.__class__.__name__, self.bounds)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)
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

        if isinstance(value, minidom.Element):
            # from XML deserialization
            iv = int(_get_node_value(value))
        else:
            # user or json deserialization
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


class _IntegerListDescriptor(_BasicDescriptor):
    """A descriptor for float array type properties"""

    def __init__(self, name, tag_entry, required=False, strict=DEFAULT_STRICT, minimum_length=0, maximum_length=2**32, docstring=None):
        super(_IntegerListDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.child_tag = tag_entry['child_tag']
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Field {} of class {} is an integer list of size {}, and must have size at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Field {} of class {} is an integer list of size {}, and must have size no larger than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            self.data[instance] = new_value

        if super(_IntegerListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, int):
            set_value([value, ])
        elif isinstance(value, minidom.Element):
            set_value([int(_get_node_value(value)), ])
        elif isinstance(value, minidom.NodeList):
            set_value([int(_get_node_value(nod)) for nod in value])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], int):
                set_value(value)
            elif isinstance(value[0], minidom.Element):
                set_value([int(_get_node_value(nod)) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


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

        if isinstance(value, minidom.Element):
            # from XML deserialization
            self.data[instance] = float(_get_node_value(value))
        else:
            # from user of json deserialization
            self.data[instance] = float(value)


class _ComplexDescriptor(_BasicDescriptor):
    """A descriptor for float type properties"""

    def __init__(self, name, required=False, strict=DEFAULT_STRICT, docstring=None):
        super(_ComplexDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):  # type: (object, float) -> None
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_ComplexDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
            # from XML deserialization
            rnode = value.getElementsByTagName('Real')
            inode = value.getElementsByTagName('Imag')
            if len(rnode) != 1:
                raise ValueError('There must be exactly one Real component of a complex type node defined.')
            if len(inode) != 1:
                raise ValueError('There must be exactly one Imag component of a complex type node defined.')
            real = float(_get_node_value(rnode))
            imag = float(_get_node_value(inode))
            self.data[instance] = complex(re=real, im=imag)
        else:
            # from user or json deserialization
            self.data[instance] = complex(value)


class _FloatArrayDescriptor(_BasicDescriptor):
    """A descriptor for float array type properties"""

    def __init__(self, name, tag_entry, required=False, strict=DEFAULT_STRICT, minimum_length=0, maximum_length=2**32, docstring=None):
        super(_FloatArrayDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.child_tag = tag_entry['child_tag']
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Field {} of class {} is a double array of size {}, and must have size at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Field {} of class {} is a double array of size {}, and must have size no larger than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            self.data[instance] = new_value

        if super(_FloatArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, numpy.ndarray):
            if not (len(value) == 1) and (numpy.dtype == numpy.float64):
                raise ValueError('Only one-dimensional ndarrays of dtype float64 are supported here.')
            set_value(value)
        elif isinstance(value, minidom.Element):
            size = int(value.getAttribute('size'))
            child_nodes = [entry for entry in value.getElementsByTagName(self.child_tag) if entry.parentNode == value]
            if len(child_nodes) != size:
                raise ValueError(
                    'Field {} of double array type functionality belonging to class {} got a minidom element with size '
                    'attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size, ), dtype=numpy.float64)
            for i, node in enumerate(new_value):
                new_value[i] = float(_get_node_value(node))
            set_value(new_value)
        elif isinstance(value, list):
            set_value(numpy.array(value, dtype=numpy.float64))
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
        if isinstance(value, minidom.Element):
            # from XML deserialization
            self.data[instance] = numpy.datetime64(_get_node_value(value), self.units)
        elif isinstance(value, numpy.datetime64):
            # from user
            self.data[instance] = value
        else:
            # from user or json deserialization
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

        if isinstance(value, minidom.Element):
            # from XML deserialization
            val = float(_get_node_value(value))
        else:
            # from user of json deserialization
            val = float(value)

        # do modular arithmatic manipulations
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

    def __init__(self, name, child_type, tags, required=False, strict=DEFAULT_STRICT,
                 minimum_length=0, maximum_length=2**32, docstring=None):
        super(_SerializableArrayDescriptor, self).__init__(name, required=required, strict=strict, docstring=docstring)
        self.child_type = child_type
        self.array = tags.get('array', False)
        self.child_tag = tags['child_tag']
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length

    def __actual_set(self, instance, value):
        if len(value) < self.minimum_length:
            msg = 'Field {} of class {} is of size {}, and must have size at least ' \
                  '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        if len(value) > self.maximum_length:
            msg = 'Field {} of class {} is of size {}, and must have size no greater than ' \
                  '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        self.data[instance] = value

    def __array_set(self, instance, value):
        if isinstance(value, numpy.ndarray):
            if value.dtype != numpy.object:
                raise ValueError(
                    'Field {} of array type functionality belonging to class {} got a ndarray of dtype {},'
                    'but contains objects, so requires dtype=numpy.object.'.format(
                        self.name, instance.__class__.__name__, value.dtype))
            elif len(value.shape) != 1:
                raise ValueError(
                    'Field {} of array type functionality belonging to class {} got a ndarray of shape {},'
                    'but requires a one dimensional array.'.format(
                        self.name, instance.__class__.__name__, value.shape))
            elif not isinstance(value[0], self.child_type):
                raise TypeError(
                    'Field {} of array type functionality belonging to class {} got a ndarray containing '
                    'first element of incompaticble type {}.'.format(
                        self.name, instance.__class__.__name__, type(value[0])))
            self.__actual_set(instance, value)
        elif isinstance(value, minidom.Element):
            # this is the parent node from XML deserialization
            size = int(value.getAttribute('size'))
            # extract child nodes at top level
            child_nodes = [entry for entry in value.getElementsByTagName(self.child_tag) if entry.parentNode == value]
            if len(child_nodes) != size:
                raise ValueError(
                    'Field {} of array type functionality belonging to class {} got a minidom element with size '
                    'attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size, ), dtype=numpy.object)
            for i, entry in enumerate(child_nodes):
                new_value[i] = self.child_type.from_node(entry).modify_tag(self.child_tag)
            self.__actual_set(instance, new_value)
        elif isinstance(value, list):
            # this would arrive from users or json deserialization
            if len(value) == 0:
                self.__actual_set(instance, numpy.empty((0, ), dtype=numpy.object))
            elif isinstance(value[0], self.child_type):
                self.__actual_set(instance, numpy.array(value, dtype=numpy.object))
            elif isinstance(value[0], dict):
                # NB: charming errors are possible here if something stupid has been done.
                self.__actual_set(instance, numpy.array(
                    [self.child_type.from_dict(node).modify_tag(self.child_tag) for node in value], dtype=numpy.object))
            else:
                raise TypeError(
                    'Field {} of array type functionality belonging to class {} got a list containing first element of '
                    'incompatible type {}.'.format(self.name, instance.__class__.__name__, type(value[0])))
        else:
            raise TypeError(
                'Field {} of array type functionality belonging to class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))

    def __list_set(self, instance, value):
        if isinstance(value, self.child_type):
            # this is the child element
            self.__actual_set(instance, [value, ])
        elif isinstance(value, minidom.Element):
            # this is the child
            self.__actual_set(instance, [self.child_type.from_node(value).modify_tag(self.child_tag), ])
        elif isinstance(value, minidom.NodeList):
            new_value = []
            for node in value:  # NB: I am ignoring the index attribute (if it exists) and just leaving it in doc order
                new_value.append(self.child_type.from_node(node).modify_tag(self.child_tag))
            self.__actual_set(instance, value)
        elif isinstance(value, list) or isinstance(value[0], self.child_type):
            if len(value) == 0:
                self.__actual_set(instance, value)
            elif isinstance(value[0], dict):
                # NB: charming errors are possible if something stupid has been done.
                self.__actual_set(instance, [self.child_type.from_dict(node).modify_tag(self.child_tag) for node in value])
            elif isinstance(value[0], minidom.Element):
                self.__actual_set(instance, [self.child_type.from_node(node).modify_tag(self.child_tag) for node in value])
            else:
                raise TypeError(
                    'Field {} of list type functionality belonging to class {} got a list containing first element of '
                    'incompatible type {}.'.format(self.name, instance.__class__.__name__, type(value[0])))
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))

    def __set__(self, instance, value):
        """
        The setter method.
        :param instance: the calling class instance
        :param value: the input value
        :return None:
        """

        if super(_SerializableArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if self.array:
            self.__array_set(instance, value)
        else:
            self.__list_set(instance, value)


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
    :param None|minidom.Element par: the parent element for the new element. Defaults to the document root
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
    :param None|minidom.Element par: parent element. Defaults to the root element if unspecified.
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

    __collections_tags = {}  # only needed for list/array type objects
    # Possible entry form:

    # - {'array': True, 'child_tag': <child_name>} represents an array object, which will have int attribute 'size'.
    #   It has #size# children with tag=<child_name>, each of which has an attribute 'index', which is not always an
    #   integer. Even when it is an integer, it apparently sometimes follows the matlab convention (1 based), and
    #   sometimes the standard convention (0 based). In this case, I will deserialize as though the objects are
    #   properly ordered, and the deserialized objects will have the 'index' property from the xml, but it will not
    #   be used to determine array order - which will be simply carried over from file order.

    # - {'array': False, 'child_tag': <child_name>} represents a collection of things with tag=<child_name>.
    #       This entries are not directly below one coherent container tag, but just dumped into an object.
    #       For example of such usage search for "Parameter" in the SICD standard.
    #
    #       In this case, I've have to create an ephemeral variable in the class that doesn't exist in the standard,
    #       and it's not clear what the intent is for this unstructured collection, so used a list object.
    #       For example, I have a variable called 'Parameters' in CollectionInfoType, whose job it to contain the
    #       parameters objects.

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

    def _get_formatter(self, attribute):
        """
        Return a formatting function for the given attribute. This will default to str if no other
        option is presented.

        :param str attribute: the given attribute name
        :return:
        """
        entry = self.__numeric_format.get(attribute, None)
        if isinstance(entry, str):
            fmt_str = '{0:' + entry + '}'
            return fmt_str.format
        elif callable(entry):
            return entry
        else:
            return str

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
            if isinstance(value, Serializable):
                good = value.is_valid(recursive=recursive)
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
            elif isinstance(val, list) or (isinstance(val, numpy.ndarray) and val.dtype == numpy.object):
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
        :param None|dict kwargs: None or dictionary of previously serialized attributes. For use in
            inheritance call, when certain attributes require specific deserialization.
        :return Serializable: corresponding class instance
        """

        def handle_attribute(the_tag):
            val = node.getAttribute(the_tag)  # this returns an empty string if the_tag doesn't exist
            if len(val) > 0:
                kwargs[the_tag] = val

        def handle_single(the_tag):
            pnodes = [entry for entry in node.getElementsByTagName(the_tag) if entry.parentNode == node]
            if len(pnodes) > 0:
                kwargs[the_tag] = pnodes[0]

        def handle_list(attribute, child_tag):
            pnodes = [entry for entry in node.getElementsByTagName(child_tag) if entry.parentNode == node]
            if len(pnodes) > 0:
                kwargs[attribute] = pnodes

        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError(
                "Named input argument kwargs for class {} must be dictionary instance".format(cls.__class__.__name__))

        for attribute in cls.__fields:
            if attribute in kwargs:
                continue

            kwargs[attribute] = None
            # This value will be replaced if tags are present
            # Note that we want to try explicitly setting to None to trigger descriptor behavior
            # for required fields (warning or error)

            if attribute in cls.__set_as_attribute:
                handle_attribute(attribute)
            elif attribute in cls.__collections_tags:
                # it's an array type of a list type parameter
                array_tag = cls.__collections_tags[attribute]
                array = array_tag.get('array', False)
                child_tag = array_tag.get('child_tag', None)
                if array:
                    handle_single(attribute)
                elif child_tag is not None:
                    # it's a list
                    handle_list(attribute, child_tag)
                else:
                    # the metadata is broken
                    raise ValueError(
                        'Attribute {} in class {} is listed in the __collections_tags dictionary, but the '
                        '"child_tags" value is either missing or None.'.format(attribute, cls.__class__.__name__))
            else:
                # it's a regular property
                handle_single(attribute)
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
        :param None|str tag: the tag name. The class name is unspecified.
        :param None|minidom.Element par: parent element. The document root element will be used if unspecified.
        :param bool strict: whether to raise an Exception if the structure is not valid.
        :param tuple exclude: property names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, but still use this super method.
        :return minidom.Element: the constructed dom element, already assigned to the parent element.
        """

        def set_attribute(node, field, value, fmt_func):
            node.setAttribute(field, fmt_func(value))

        def serialize_array(node, the_tag, child_tag, value, fmt_func):
            if not isinstance(value, numpy.ndarray):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # __array_tag or the descriptor, probably at runtime
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be an array based on '
                    'the metadata in the __arrays_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(value)))
            if not len(value.shape) == 1:
                # again, I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a one-dimensional numpy.ndarray, but it has shape {}'.format(
                        attribute, self.__class__.__name__, value.shape))
            if value.size == 0:
                return  # serializing an empty array is dumb

            if value.dtype == numpy.float64:
                anode = _create_new_node(doc, the_tag, par=node)
                anode.setAttribute('size', str(value.size))
                for i, val in enumerate(value):
                    vnode = _create_text_node(doc, child_tag, fmt_func(val), par=anode)
                    vnode.setAttribute('index', str(i))  # I think that this is reliable
            elif value.dtype == numpy.object:
                anode = _create_new_node(doc, the_tag, par=node)
                anode.setAttribute('size', str(value.size))
                for i, entry in enumerate(value):
                    if not isinstance(entry, Serializable):
                        raise TypeError(
                            'The value associated with attribute {} is an instance of class {} should be an object '
                            'array based on the standard, but entry {} is of type {} and not an instance of '
                            'Serializable'.format(attribute, self.__class__.__name__, i, type(entry)))
                    serialize_plain(anode, child_tag, entry, fmt_func)
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a numpy.ndarray of dtype float64 or object, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, value.dtype))

        def serialize_list(node, child_tag, value, fmt_func):
            if not isinstance(value, list):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # __collections_tags or the descriptor?
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be a list based on '
                    'the metadata in the __arrays_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(value)))
            if len(value) == 0:
                return  # serializing an empty list is dumb
            else:
                for entry in value:
                    serialize_plain(node, child_tag, entry, fmt_func)

        def serialize_plain(node, field, value, fmt_func):
            # may be called not at top level - if object array or list is present
            if isinstance(value, Serializable):
                value.to_node(doc, tag=field, par=node, strict=strict)
            elif isinstance(value, str):
                # TODO: MEDIUM - unicode issues?
                _create_text_node(doc, field, value, par=node)
            elif isinstance(value, int) or isinstance(value, float):
                _create_text_node(doc, field, fmt_func(value), par=node)
            elif isinstance(value, bool):
                # fmt_func here? note that str doesn't work, so...
                _create_text_node(doc, field, '1' if value else '0', par=node)
            elif isinstance(value, date):
                _create_text_node(doc, field, value.isoformat(), par=node)
            elif isinstance(value, datetime):
                _create_text_node(doc, field, value.isoformat(sep='T'), par=node)
            elif isinstance(value, numpy.datetime64):
                _create_text_node(doc, field, str(value), par=node)
            elif isinstance(value, complex):
                cnode = _create_new_node(doc, field, par=node)
                _create_text_node(doc, 'Real', fmt_func(value.real), par=cnode)
                _create_text_node(doc, 'Imag', fmt_func(value.imag), par=cnode)
            else:
                raise ValueError(
                    'a entry for class {} using tag {} is of type {}, and serialization has not '
                    'been implemented'.format(self.__class__.__name__, field, type(value)))

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

            fmt_func = self._get_formatter(attribute)
            array_tag = self.__collections_tags.get(attribute, None)
            if attribute in self.__set_as_attribute:
                set_attribute(nod, attribute, value, fmt_func)
            elif array_tag is not None:
                array = array_tag.get('array', False)
                child_tag = array_tag.get('child_tag', None)
                if array:
                    # this will be an numpy.ndarray
                    serialize_array(nod, attribute, child_tag, value, fmt_func)
                elif child_tag is not None:
                    # this will be a list
                    serialize_list(nod, child_tag, value, fmt_func)
                else:
                    # the metadata is broken
                    raise ValueError(
                        'Attribute {} in class {} is listed in the __collections_tags dictionary, but the '
                        '"child_tags" value is either missing or None.'.format(attribute, self.__class__.__name__))
            else:
                serialize_plain(nod, attribute, value, fmt_func)
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
    """This is a basic xml building block element, and not actually specified in the SICD standard"""
    __fields = ('value', )
    __required = __fields
    # descriptor
    value = _StringDescriptor('value', required=True, strict=True, docstring=':str: The value')

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
        :param None|dict kwargs: None or dictionary of previously serialized attributes. For use in
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
        :param None|str tag: the tag name. The class name is unspecified.
        :param None|minidom.Element par: parent element. The document root element will be used if unspecified.
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
    """This is a basic xml building block element, and not actually specified in the SICD standard"""
    __fields = ('value', )
    __required = __fields
    # descriptor
    value = _FloatDescriptor('value', required=True, strict=True, docstring=':float: The value')

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
        :param None|dict kwargs: None or dictionary of previously serialized attributes. For use in
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
        :param None|str tag: the tag name. The class name is unspecified.
        :param None|minidom.Element par: parent element. The document root element will be used if unspecified.
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


class ComplexType(PlainValueType):
    """A complex number"""
    __fields = ('Real', 'Imag')
    __required = __fields
    # descriptor
    Real = _FloatDescriptor('Real', required=True, strict=True, docstring=':float: The real component.')
    Imag = _FloatDescriptor('Imag', required=True, strict=True, docstring=':float: The imaginary  component.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key are ('Real', 'Imag'), all required.
        """
        super(ComplexType, self).__init__(**kwargs)


class ParameterType(PlainValueType):
    """A Parameter structure - just a name attribute and associated value"""
    __tag = 'Parameter'
    __fields = ('name', 'value')
    __required = __fields
    __set_as_attribute = ('name', )
    # descriptor
    name = _StringDescriptor('name', required=True, strict=True, docstring=':str: The name')
    value = _StringDescriptor('value', required=True, strict=True, docstring=':str: The value')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid key are ('name', 'value'), all required.
        """
        super(ParameterType, self).__init__(**kwargs)


class XYZType(Serializable):
    """A spatial point, probably in ECF coordinates."""
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    __numeric_format = {'X': '0.8f', 'Y': '0.8f', 'Z': '0.8f'}  # TODO: desired precision? This is usually meters?
    # descriptors
    X = _FloatDescriptor(
        'X', required=True, strict=DEFAULT_STRICT,
        docstring=':float: The X attribute. Assumed to ECF or other, similar coordinates.')
    Y = _FloatDescriptor(
        'Y', required=True, strict=DEFAULT_STRICT,
        docstring=':float: The Y attribute. Assumed to ECF or other, similar coordinates.')
    Z = _FloatDescriptor(
        'Z', required=True, strict=DEFAULT_STRICT,
        docstring=':float: The Z attribute. Assumed to ECF or other, similar coordinates.')

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
    """A two-dimensional geographic point in WGS-84 coordinates."""
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', required=True, strict=DEFAULT_STRICT,
        docstring=':float: The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatDescriptor(
        'Lon', required=True, strict=DEFAULT_STRICT,
        docstring=':float: The longitude attribute. Assumed to be WGS-84 coordinates.')

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


class LatLonArrayElementType(LatLonType):
    """An geographic point in an array"""
    __fields = ('Lat', 'Lon', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    index = _IntegerDescriptor('index', required=True, strict=False, docstring=":int: The array index")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'index'), all required.
        """
        super(LatLonArrayElementType, self).__init__(**kwargs)


class LatLonRestrictionType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates."""
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    # descriptors
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, required=True, strict=DEFAULT_STRICT,
        docstring=':float: The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, required=True, strict=DEFAULT_STRICT,
        docstring=':float: The longitude attribute. Assumed to be WGS-84 coordinates.')

    def __init__(self, **kwargs):
        super(LatLonRestrictionType, self).__init__(**kwargs)


class LatLonHAEType(LatLonType):
    """A three-dimensional geographic point in WGS-84 coordinates."""
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?
    # descriptors
    HAE = _FloatDescriptor(
        'HAE', required=True, strict=DEFAULT_STRICT,
        docstring=':float: The Height Above Ellipsoid (in meters) attribute. Assumed to be WGS-84 coordinates.')

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
    """A three-dimensional geographic point in WGS-84 coordinates."""
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, required=True, strict=DEFAULT_STRICT,
        docstring=':float: The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, required=True, strict=DEFAULT_STRICT,
        docstring=':float: The longitude attribute. Assumed to be WGS-84 coordinates.')

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
        docstring=':int: The integer index in 1-4. This represents a clockwise enumeration of the rectangle vertices '
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
    # other specific class variable
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, required=True, strict=True,
        docstring=":str: The string index in {}".format(_CORNER_VALUES))

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
        docstring=':int: The integer index in 1-4. This represents a clockwise enumeration of the rectangle vertices '
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
        'index', _CORNER_VALUES, required=True, strict=True, docstring=":str: The string index in {}".format(_CORNER_VALUES))

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Lat', 'Lon', 'HAE', 'index'), all required.
        """
        super(LatLonHAECornerStringType, self).__init__(**kwargs)


class RowColType(Serializable):
    __fields = ('Row', 'Col')
    __required = __fields
    Row = _IntegerDescriptor(
        'Row', required=True, strict=DEFAULT_STRICT, docstring=':int: The Row attribute.')
    Col = _IntegerDescriptor(
        'Col', required=True, strict=DEFAULT_STRICT, docstring=':int: The Column attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Row', 'Col'), both required.
        """
        super(RowColType, self).__init__(**kwargs)


class RowColArrayElement(RowColType):
    """"""
    # Note - in the SICD standard this type is listed as RowColvertexType. This is not a descriptive name
    # and has an inconsistency in camel case
    __fields = ('Row', 'Col', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', required=True, strict=DEFAULT_STRICT, docstring=':int: The index attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Row', 'Col', 'index'), all required.
        """
        super(RowColArrayElement, self).__init__(**kwargs)


class PolyCoef1DType(FloatValueType):
    """
    Represents a monomial term of the form `value * x^{exponent1}`.
    """
    __fields = ('value', 'exponent1')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    __set_as_attribute = ('exponent1', )
    # descriptors
    exponent1 = _IntegerDescriptor(
        'exponent1', required=True, strict=DEFAULT_STRICT, docstring=':int: The exponent1 attribute.')

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
    # descriptors
    exponent1 = _IntegerDescriptor(
        'exponent1', required=True, strict=DEFAULT_STRICT, docstring=':int: The exponent1 attribute.')
    exponent2 = _IntegerDescriptor(
        'exponent2', required=True, strict=DEFAULT_STRICT, docstring=':int: The exponent2 attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('value', 'exponent1', 'exponent2'), both required.
        """
        super(PolyCoef2DType, self).__init__(**kwargs)


class Poly1DType(Serializable):
    """Represents a one-variable polynomial, defined as the sum of the given monomial terms."""
    __fields = ('Coefs', 'order1')
    __required = ('Coefs', )
    __collections_tags = {'Coefs': {'array': False, 'child_tag': 'Coef'}}
    __set_as_attribute = ('order1', )
    # descriptos
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef1DType, __collections_tags['Coefs'], required=True, strict=DEFAULT_STRICT,
        docstring=':PolyCoef1DType: The **list** of monomial terms.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the only valid key is 'Coefs', and is required.
        """
        super(Poly1DType, self).__init__(**kwargs)

    @property
    def order1(self):  # type: () -> int
        """
        :int: The order1 attribute [READ ONLY]  - that is, largest exponent presented in the monomial terms of coefs.
        """

        return 0 if self.Coefs is None else max(entry.exponent1 for entry in self.Coefs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class Poly2DType(Serializable):
    """Represents a one-variable polynomial, defined as the sum of the given monomial terms."""
    __fields = ('Coefs', 'order1', 'order2')
    __required = ('Coefs', )
    __collections_tags = {'Coefs': {'array': False, 'child_tag': 'Coef'}}
    __set_as_attribute = ('order1', 'order2')
    # descriptors
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef2DType, __collections_tags['Coefs'], required=True, strict=DEFAULT_STRICT,
        docstring=':PolyCoef2DType: The **list** of monomial terms.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the only valid key is 'Coefs', and is required.
        """
        super(Poly2DType, self).__init__(**kwargs)

    @property
    def order1(self):  # type: () -> int
        """
        :int: The order1 attribute [READ ONLY]  - that is, the largest exponent1 presented in the monomial terms of coefs.
        """

        return 0 if self.Coefs is None else max(entry.exponent1 for entry in self.Coefs)

    @property
    def order2(self):  # type: () -> int
        """
        :int: The order2 attribute [READ ONLY]  - that is, the largest exponent2 presented in the monomial terms of coefs.
        """

        return 0 if self.Coefs is None else max(entry.exponent2 for entry in self.Coefs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyType(Serializable):
    """Represents a single variable polynomial for each of `X`, `Y`, and `Z`."""
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    # descriptors
    X = _SerializableDescriptor(
        'X', Poly1DType, required=True, strict=DEFAULT_STRICT, docstring=':Poly1DType: The X polynomial.')
    Y = _SerializableDescriptor(
        'Y', Poly1DType, required=True, strict=DEFAULT_STRICT, docstring=':Poly1DType: The Y polynomial.')
    Z = _SerializableDescriptor(
        'Z', Poly1DType, required=True, strict=DEFAULT_STRICT, docstring=':Poly1DType: The Z polynomial.')
    # TODO: a better description would be good here

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('X', 'Y', 'Z'), and are required.
        """
        super(XYZPolyType, self).__init__(**kwargs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyAttributeType(XYZPolyType):
    """
    An array element of X, Y, Z polynomials. The output of these polynomials are expected to spatial variables in
    the ECF coordinate system.
    """
    __fields = ('X', 'Y', 'Z', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', required=True, strict=DEFAULT_STRICT, docstring=':int: The array index value.')

    def __init__(self, **kwargs):
        """The constructor. 
        :param dict kwargs: the valid keys are in ('X', 'Y', 'Z', 'index'), all required.
        """
        super(XYZPolyAttributeType, self).__init__(**kwargs)


class GainPhasePolyType(Serializable):
    """A container for the Gain and Phase Polygon definitions."""
    __fields = ('GainPoly', 'PhasePoly')
    __required = __fields
    # descriptors
    GainPoly = _SerializableDescriptor(
        'GainPoly', Poly2DType, required=True, strict=DEFAULT_STRICT, docstring=':Poly2DType: The Gain Polygon.')
    PhasePoly = _SerializableDescriptor(
        'GainPhasePoly', Poly2DType, required=True, strict=DEFAULT_STRICT, docstring=':Poly2DType: The Phase Polygon.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are in ('GainPoly', 'PhasePoly'), all required.
        """
        super(GainPhasePolyType, self).__init__(**kwargs)


class ErrorDecorrFuncType(Serializable):
    """
    The Error Decorrelation Function?
    """
    __fields = ('CorrCoefZero', 'DecorrRate')
    __required = __fields
    __numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '0.8f'}  # TODO: desired precision?
    # descriptors
    CorrCoefZero = _FloatDescriptor(
        'CorrCoefZero', required=True, strict=DEFAULT_STRICT, docstring=':float: The CorrCoefZero attribute.')
    DecorrRate = _FloatDescriptor(
        'DecorrRate', required=True, strict=DEFAULT_STRICT, docstring=':float: The DecorrRate attribute.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['CorrCoefZero', 'DecorrRate'], all required.
        """
        super(ErrorDecorrFuncType, self).__init__(**kwargs)

    # TODO: HIGH - this is supposed to be a "function". We should implement the functionality here.


class RadarModeType(Serializable):
    """Radar mode type container class"""
    __tag = 'RadarMode'
    __fields = ('ModeType', 'ModeId')
    __required = ('ModeType', )
    # other class variable
    _MODE_TYPE_VALUES = ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP')
    # descriptors
    ModeId = _StringDescriptor('ModeId', required=False, docstring=':str: The (optional) Mode Id.')
    ModeType = _StringEnumDescriptor(
        'ModeType', _MODE_TYPE_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring=":str: The mode type, which will be one of {}.".format(_MODE_TYPE_VALUES))

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['ModeType', 'ModeId'], and 'ModeType' is required.
        """
        super(RadarModeType, self).__init__(**kwargs)


class FullImageType(Serializable):
    """The full image attributes"""
    __tag = 'FullImage'
    __fields = ('NumRows', 'NumCols')
    __required = __fields
    # descriptors
    NumRows = _IntegerDescriptor('NumRows', required=True, strict=DEFAULT_STRICT, docstring=':int: The number of rows.')
    NumCols = _IntegerDescriptor('NumCols', required=True, strict=DEFAULT_STRICT, docstring=':int: The number of columns.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: valid keys in ['NumRows', 'NumCols'], all required.
        """
        super(FullImageType, self).__init__(**kwargs)


####################################################################
# Direct building blocks for SICD
#############
# CollectionInfoType section

class CollectionInfoType(Serializable):
    """The collection information container."""
    __tag = 'CollectionInfo'
    __collections_tags = {
        'Parameters': {'array': False, 'child_tag': 'Parameter'},
        'CountryCode': {'array': False, 'child_tag': 'CountryCode'},
    }
    __fields = (
        'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
        'RadarMode', 'Classification', 'Parameters', 'CountryCodes')
    __required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')
    # other class variable
    _COLLECT_TYPE_VALUES = ('MONOSTATIC', 'BISTATIC')
    # descriptors
    CollectorName = _StringDescriptor(
        'CollectorName', required=True, strict=DEFAULT_STRICT, docstring=':str: The collector name.')
    IlluminatorName = _StringDescriptor(
        'IlluminatorName', required=False, strict=DEFAULT_STRICT, docstring=':str: The (optional) illuminator name.')
    CoreName = _StringDescriptor(
        'CoreName', required=True, strict=DEFAULT_STRICT, docstring=':str: The core name.')
    CollectType = _StringEnumDescriptor(
        'CollectType', _COLLECT_TYPE_VALUES, required=False,
        docstring=":str: The collect type, one of {}".format(_COLLECT_TYPE_VALUES))
    RadarMode = _SerializableDescriptor(
        'RadarMode', RadarModeType, required=True, strict=DEFAULT_STRICT, docstring=':RadarModeType: The radar mode')
    Classification = _StringDescriptor(
        'Classification', required=True, strict=DEFAULT_STRICT, docstring=':str: The classification.')
    # list type descriptors
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags['Parameters'], required=False, strict=DEFAULT_STRICT,
        docstring=':ParameterType: The (optional) parameters objects **list**.')
    CountryCodes = _StringListDescriptor(
        'CountryCodes', required=False, strict=DEFAULT_STRICT, docstring=":str: The (optional) country code **list**.")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are
            ('CollectorName', 'IlluminatorName', 'CoreName', 'CollectType', 'RadarMode', 'Classification',
            'CountryCodes', 'Parameters').
            While the required fields are ('CollectorName', 'CoreName', 'RadarMode', 'Classification').
        """
        super(CollectionInfoType, self).__init__(**kwargs)

###############
# ImageCreation section

class ImageCreationType(Serializable):
    """
    The image creation data container.
    """
    __fields = ('Application', 'DateTime', 'Site', 'Profile')
    __required = ()
    # descriptors
    Application = _StringDescriptor('Application', required=False, strict=DEFAULT_STRICT, docstring='The Application')
    DateTime = _DateTimeDescriptor(
        'DateTime', required=False, strict=DEFAULT_STRICT, docstring=':numpy.datetime64: The Date/Time', numpy_datetime_units='us')
    Site = _StringDescriptor('Site', required=False, strict=DEFAULT_STRICT, docstring=':str: The Site')
    Profile = _StringDescriptor('Profile', required=False, strict=DEFAULT_STRICT, docstring=':str: The Profile')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ['Application', 'DateTime', 'Site', 'Profile'], and none are required
        """
        super(ImageCreationType, self).__init__(**kwargs)

###############
# ImageData section

class ImageDataType(Serializable):
    """
    The image data container.
    """
    __collections_tags = {
        'AmpTable': {'array': True, 'child_tag': 'Amplitude'},
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
    }
    __fields = ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel',
                'ValidData')
    __required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    __numeric_format = {'AmpTable': '0.8f'}  # TODO: precision for AmpTable?
    _PIXEL_TYPE_VALUES = ("RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I")
    # descriptors
    PixelType = _StringEnumDescriptor(
        'PixelType', _PIXEL_TYPE_VALUES, required=True, strict=True,
        docstring="""
        :str: The PixelType attribute which specifies the interpretation of the file data:
            * `"RE32F_IM32F"` - a pixel is specified as `(real, imaginary)` each 32 bit floating point.
            * `"RE16I_IM16I"` - a pixel is specified as `(real, imaginary)` each a 16 bit (short) integer.
            * `"AMP8I_PHS8I"` - a pixel is specified as `(amplitude, phase)` each an 8 bit unsigned integer. The
                `amplitude` actually specifies the index into the `AmpTable` attribute. The `angle` is properly
                interpreted (in radians) as `theta = 2*pi*angle/256`.
        """)
    NumRows = _IntegerDescriptor('NumRows', required=True, strict=DEFAULT_STRICT, docstring=':int: The number of Rows')
    NumCols = _IntegerDescriptor('NumCols', required=True, strict=DEFAULT_STRICT, docstring=':int: The number of Columns')
    FirstRow = _IntegerDescriptor('FirstRow', required=True, strict=DEFAULT_STRICT, docstring=':int: The first row')
    FirstCol = _IntegerDescriptor('FirstCol', required=True, strict=DEFAULT_STRICT, docstring=':int: The first column')
    FullImage = _SerializableDescriptor(
        'FullImage', FullImageType, required=True, strict=DEFAULT_STRICT, docstring=':FullImageType: The full image')
    SCPPixel = _SerializableDescriptor(
        'SCPPixel', RowColType, required=True, strict=DEFAULT_STRICT, docstring=':RowColType: The SCP Pixel')
    ValidData = _SerializableArrayDescriptor(
        'ValidData', RowColArrayElement, __collections_tags['ValidData'], required=False, strict=DEFAULT_STRICT,
        minimum_length=3, docstring=':RowColArrayElement: The valid data area **array**')
    AmpTable = _FloatArrayDescriptor(
        'AmpTable', __collections_tags['AmpTable'], required=False, strict=DEFAULT_STRICT,
        minimum_length=256, maximum_length=256,
        docstring=":float: The 256 element amplitude look-up table **array**. This must be defined if PixelType == 'AMP8I_PHS8I'")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are
            ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel', 'ValidData')
            Required are ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel'),
            and 'AmpTable' must conditionally be defined if `PixelType == 'AMP8I_PHS8I'`.
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

###############
# GeoInfo section


class GeoInfoType(Serializable):
    """The GeoInfo container."""
    __tag = 'GeoInfo'
    __collections_tags = {
        'Descriptions': {'array': False, 'child_tag': 'Desc'},
        'Line': {'array': True, 'child_tag': 'Endpoint'},
        'Polygon': {'array': True, 'child_tag': 'Vertex'}, }
    __fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon')
    __required = ('name', )
    __set_as_attribute = ('name', )
    # descriptors
    name = _StringDescriptor('name', required=True, strict=True, docstring=':str: The name.')
    Descriptions = _SerializableArrayDescriptor(
        'Descriptions', ParameterType, __collections_tags['Descriptions'], required=False, strict=DEFAULT_STRICT,
        docstring=':ParameterType: The descriptions **list**.')
    Point = _SerializableDescriptor(
        'Point', LatLonRestrictionType, required=False, strict=DEFAULT_STRICT,
        docstring=':LatLonRestrictionType: A geographic point with WGS-84 coordinates.')
    Line = _SerializableArrayDescriptor(
        'Line', LatLonArrayElementType, __collections_tags['Line'], required=False, strict=DEFAULT_STRICT, minimum_length=2,
        docstring=':LatLonRestrictionType: A geographic line (**array**) with WGS-84 coordinates.')
    Polygon = _SerializableArrayDescriptor(
        'Polygon', LatLonArrayElementType, __collections_tags['Polygon'], required=False, strict=DEFAULT_STRICT, minimum_length=3,
        docstring=':LatLonRestrictionType: A geographic polygon (**array**) with WGS-84 coordinates.')
    # TODO: is the standard really self-referential here? I find that confusing.

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ('name', 'Descriptions', 'Point', 'Line', 'Polygon'), where
            the only required key is 'name' , and only one of 'Point', 'Line', or 'Polygon' should be present.
        """
        super(GeoInfoType, self).__init__(**kwargs)


###############
# GeoData section


class SCPType(Serializable):
    """The Scene Center Point container"""
    __tag = 'SCP'
    __fields = ('ECF', 'LLH')
    __required = __fields  # isn't this redundant?
    ECF = _SerializableDescriptor(
        'ECF', XYZType, required=True, strict=DEFAULT_STRICT, docstring=':XYZType: The ECF coordinates.')
    LLH = _SerializableDescriptor(
        'LLH', LatLonHAERestrictionType, required=True, strict=DEFAULT_STRICT,
        docstring=':LatLonHAERestrictionType: The WGS 84 coordinates.'
    )

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the keys are ('ECF', 'LLH'), all required.
        """
        super(SCPType, self).__init__(**kwargs)


class GeoDataType(Serializable):
    """Container specifying the image coverage area in geographic coordinates."""
    __fields = ('EarthModel', 'SCP', 'ImageCorners', 'ValidData', 'GeoInfos')
    __required = ('EarthModel', 'SCP', 'ImageCorners')
    __collections_tags = {
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
        'ImageCorners': {'array': True, 'child_tag': 'ICP'},
        'GeoInfos': {'array': False, 'child_tag': 'GeoInfo'},
    }
    # other class variables
    _EARTH_MODEL_VALUES = ('WGS_84', )
    # descriptors
    EarthModel = _StringEnumDescriptor(
        'EarthModel', _EARTH_MODEL_VALUES, required=True, strict=True, default_value='WGS_84',
        docstring=':str: The Earth Model, which taks values in {}.'.format(_EARTH_MODEL_VALUES))
    SCP = _SerializableDescriptor(
        'SCP', SCPType, required=True, strict=DEFAULT_STRICT, tag='SCP', docstring=':SCPType: The Scene Center Point.')
    ImageCorners = _SerializableArrayDescriptor(
        'ImageCorners', LatLonCornerStringType, __collections_tags['ImageCorners'], required=True, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring=':LatLonCornerStringType: The (4) geographic image corner points **array**.')
    ValidData = _SerializableArrayDescriptor(
        'ValidData', LatLonArrayElementType, __collections_tags['ValidData'], required=False,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring=':LatLonArrayElementType: The full image **array** includes both valid data and some zero filled pixels.')
    GeoInfos = _SerializableArrayDescriptor(
        'GeoInfos', GeoInfoType, __collections_tags['GeoInfos'], required=False, strict=DEFAULT_STRICT,
        docstring=':GeoInfoType: Relevant geographic features **list**.')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ('EarthModel', 'SCP', 'ImageCorners', 'ValidData', 'GeoInfos'), and
            ('EarthModel', 'SCP', 'ImageCorners') are required.
        """
        super(GeoDataType, self).__init__(**kwargs)


################
# DirParam section


class WgtTypeType(Serializable):
    """The weight type parameters of the direction parameters"""
    __fields = ('WindowName', 'Parameters')
    __required = ('WindowName', )
    __collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    WindowName = _StringDescriptor('WindowName', required=True, strict=DEFAULT_STRICT, docstring='The window name')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags['Parameters'], required=False, strict=DEFAULT_STRICT,
        docstring=':ParameterType: The parameters **list**')

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are ('WindowName', 'Parameters'), and 'WindowName' is required.
        """
        super(WgtTypeType, self).__init__(**kwargs)


class DirParamType(Serializable):
    """The direction parameters container"""
    __fields = (
        'UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly',
        'WgtType', 'WgtFunct')
    __required = ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2')
    __numeric_format = {
        'SS': '0.8f', 'ImpRespWid': '0.8f', 'Sgn': '+d', 'ImpRespBW': '0.8f', 'KCtr': '0.8f',
        'DeltaK1': '0.8f', 'DeltaK2': '0.8f'}
    __collections_tags = {'WgtFunct': {'array': True, 'child_tag': 'Wgt'}}
    # descriptors
    UVectECF = _SerializableDescriptor(
        'UVectECF', XYZType, tag='UVectECF', required=True, strict=DEFAULT_STRICT,
        docstring=':XYZType: Unit vector in the increasing (row/col) direction (ECF) at the SCP pixel.')
    SS = _FloatDescriptor(
        'SS', required=True, strict=DEFAULT_STRICT,
        docstring=':float: Sample spacing in the increasing (row/col) direction. Precise spacing at the SCP.')
    ImpRespWid = _FloatDescriptor(
        'ImpRespWid', required=True, strict=DEFAULT_STRICT,
        docstring=':float: Half power impulse response width in the increasing (row/col) direction. Measured at the SCP.')
    Sgn = _IntegerEnumDescriptor(
        'Sgn', values=(1, -1), required=True, strict=DEFAULT_STRICT,
        docstring=':int: Sign (+1/-1) for exponent in the DFT to transform the (row/col) dimension to '
                  'spatial frequency dimension.')
    ImpRespBW = _FloatDescriptor(
        'ImpRespBW', required=True, strict=DEFAULT_STRICT,
        docstring=':float: Spatial bandwidth in (row/col) used to form the impulse response in the (row/col) direction. '
                  'Measured at the center of support for the SCP.')
    KCtr = _FloatDescriptor(
        'KCtr', required=True, strict=DEFAULT_STRICT,
        docstring=':float: Center spatial frequency in the given dimension. '
                  'Corresponds to the zero frequency of the DFT in the given (row/col) direction.')
    DeltaK1 = _FloatDescriptor(
        'DeltaK1', required=True, strict=DEFAULT_STRICT,
        docstring=':float: Minimum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaK2 = _FloatDescriptor(
        'DeltaK2', required=True, strict=DEFAULT_STRICT,
        docstring=':float: Maximum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaKCOAPoly = _SerializableDescriptor(
        'DeltaKCOAPoly', Poly2DType, tag='DeltaKCOAPoly', required=False, strict=DEFAULT_STRICT,
        docstring=':Poly2DType: Offset from KCtr of the center of support in the given (row/col) spatial frequency. '
                  'The polynomial is a function of image given (row/col) coordinate (variable 1) and '
                  'column coordinate (variable 2).')
    WgtType = _SerializableDescriptor(
        'WgtType', WgtTypeType, tag='WgtType', required=False, strict=DEFAULT_STRICT,
        docstring=':WgtTypeType: Parameters describing aperture weighting type applied in the spatial frequency domain '
                  'to yield the impulse response in the given(row/col) direction.')
    WgtFunct = _FloatArrayDescriptor(
        'WgtFunct', __collections_tags['WgtFunct'], required=False, strict=DEFAULT_STRICT, minimum_length=2,
        docstring=':float: Sampled aperture amplitude weighting function (**array**) applied to form the SCP impulse '
                  'response in the given (row/col) direction. If present, the size of the array is required to be at least 2..')

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

###############
# GridType section

class GridType(Serializable):
    """Collection grid details container"""
    __fields = ('ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col')
    __required = __fields
    _IMAGE_PLANE_VALUES = ('SLANT', 'GROUND', 'OTHER')
    _TYPE_VALUES = ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    # descriptors
    ImagePlane = _StringEnumDescriptor(
        'ImagePlane', _IMAGE_PLANE_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring=":str: The image plane. Possible values are {}".format(_IMAGE_PLANE_VALUES))
    Type = _StringEnumDescriptor(
        'Type', _TYPE_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring=""":str: The possible values and meanings:
* 'RGAZIM' - Range & azimuth relative to the ARP at a reference time.
* 'RGZERO' - Range from ARP trajectory at zero Doppler and azimuth aligned with the strip being imaged.
* 'XRGYCR' - Orthogonal slant plane grid oriented range and cross range relative to the ARP at a reference time.
* 'XCTYAT' - Orthogonal slant plane grid with X oriented cross track.
* 'PLANE'  - Uniformly sampled in an arbitrary plane along directions U & V.""")
    TimeCOAPoly = _SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, required=True, strict=DEFAULT_STRICT,
        docstring=":Poly2DType: *Time of Center Of Aperture* as a polynomial function of image coordinates. The polynomial is a "
                  "function of image row coordinate (variable 1) and column coordinate (variable 2).")
    Row = _SerializableDescriptor(
        'Row', DirParamType, required=True, strict=DEFAULT_STRICT,
        docstring=":DirParamType: Row direction parameters.")
    Col = _SerializableDescriptor(
        'Col', DirParamType, required=True, strict=DEFAULT_STRICT,
        docstring=":DirParamType: Column direction parameters.")

    def __init__(self, **kwargs):
        """
        The constructor.
        :param dict kwargs: the valid keys are .
        """
        super(GridType, self).__init__(**kwargs)


##############
# TimelineType section


class IPPSetType(Serializable):
    """The Inter-Pulse Parameter array element container."""
    # NOTE that this is simply defined as a child class ("Set") of the TimelineType in the SICD standard
    #   Defining it at root level clarifies the documentation, and giving it a more descriptive name is
    #   appropriate.
    __tag = 'Set'
    __fields = ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    TStart = _FloatDescriptor(
        'TStart', required=True, strict=DEFAULT_STRICT,
        docstring=':float: IPP start time relative to collection start time, i.e. offsets in seconds.')
    TEnd = _FloatDescriptor(
        'TEnd', required=True, strict=DEFAULT_STRICT,
        docstring=':float: IPP end time relative to collection start time, i.e. offsets in seconds.')
    IPPStart = _IntegerDescriptor(
        'IPPStart', required=True, strict=True, docstring=':int: Starting IPP index for the period described.')
    IPPEnd = _IntegerDescriptor(
        'IPPEnd', required=True, strict=True, docstring=':int: Ending IPP index for the period described.')
    IPPPoly = _SerializableDescriptor(
        'IPPPoly', Poly1DType, required=True, strict=DEFAULT_STRICT,
        docstring=':Poly1DType: IPP index polynomial coefficients yield IPP index as a function of time for TStart to TEnd.')
    index = _IntegerDescriptor(
        'index', required=True, strict=DEFAULT_STRICT, docstring=':int: The element array index.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index'),
            all required.
        """
        super(IPPSetType, self).__init__(**kwargs)


class TimelineType(Serializable):
    """The details for the imaging collection timeline."""
    __fields = ('CollectStart', 'CollectDuration', 'IPP')
    __required = ('CollectStart', 'CollectDuration', )
    __collections_tags = {'IPP': {'array': True, 'child_tag': 'Set'}}
    # descriptors
    CollectStart = _DateTimeDescriptor(
        'CollectStart', required=True, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring=':numpy.datetime64: The collection start time. The default precision will be microseconds.')
    CollectDuration = _FloatDescriptor(
        'CollectDuration', required=True, strict=DEFAULT_STRICT,
        docstring=':float: The duration of the collection in seconds.')
    IPP = _SerializableArrayDescriptor(
        'IPP', IPPSetType, __collections_tags['IPP'], required=False, strict=DEFAULT_STRICT, minimum_length=1,
        docstring=":IPPSetType: The Inter-Pulse Period (IPP) parameters **array**.")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('CollectStart', 'CollectDuration', 'IPP'), and
            ('CollectStart', 'CollectDuration') are required.
        """
        super(TimelineType, self).__init__(**kwargs)


###################
# PositionType section


class PositionType(Serializable):
    """The details for platform and ground reference positions as a function of time since collection start."""
    __fields = ('ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC')
    __required = ('ARPPoly',)
    __collections_tags = {'RcvAPC': {'array': True, 'child_tag': 'RcvAPCPoly'}}

    # descriptors
    ARPPoly = _SerializableDescriptor(
        'ARPPoly', XYZPolyType, required=('ARPPoly' in __required), strict=DEFAULT_STRICT,
        docstring=':XYZPolyType: Aperture Reference Point (ARP) position polynomial in ECF as a function of elapsed seconds '
                  'since start of collection.')
    GRPPoly = _SerializableDescriptor(
        'GRPPoly', XYZPolyType, required=('GRPPoly' in __required), strict=DEFAULT_STRICT,
        docstring=':XYZPolyType: Ground Reference Point (GRP) position polynomial in ECF as a function of elapsed seconds '
                  'since start of collection.')
    TxAPCPoly = _SerializableDescriptor(
        'TxAPCPoly', XYZPolyType, required=('TxAPCPoly' in __required), strict=DEFAULT_STRICT,
        docstring=':XYZPolyType: Transmit Aperture Phase Center (APC) position polynomial in ECF as a function of elapsed seconds '
                  'since start of collection.')
    RcvAPC = _SerializableArrayDescriptor(
        'RcvAPC', XYZPolyAttributeType, __collections_tags['RcvAPC'], required=('RcvAPC' in __required), strict=DEFAULT_STRICT,
        docstring=':XYZPolyAttributeType: Receive Aperture Phase Center polynomials **array**. Each polynomial has output '
                  'in ECF, and represents a function of elapsed seconds since start of collection.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC'), and only '
            ARPPoly' is required.
        """
        super(PositionType, self).__init__(**kwargs)


##################
# RadarCollectionType section


class TxFrequencyType(Serializable):
    """The transmit frequency range"""
    __tag = 'TxFrequency'
    __fields = ('Min', 'Max')
    __required = __fields
    # descriptors
    Min = _FloatDescriptor(
        'Min', required=('Min' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The transmit minimum frequency in Hz.')
    Max = _FloatDescriptor(
        'Max', required=('Max' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The transmit maximum frequency in Hz.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Min', 'Max'), all required.
        """
        super(TxFrequencyType, self).__init__(**kwargs)


class WaveformParametersType(Serializable):
    """Transmit and receive demodulation waveform parameters."""
    __fields = (
        'TxPulseLength', 'TxRFBandwidth', 'TxFreqStart', 'TxFMRate', 'RcvDemodType', 'RcvWindowLength',
        'ADCSampleRate', 'RcvIFBandwidth', 'RcvFreqStart', 'RcvFMRate')
    __required = ()
    # other class variables
    _DEMOD_TYPE_VALUES = ('STRETCH', 'CHIRP')
    # descriptors
    TxPulseLength = _FloatDescriptor(
        'TxPulseLength', required=('TxPulseLength' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Transmit pulse length in seconds.')
    TxRFBandwidth = _FloatDescriptor(
        'TxRFBandwidth', required=('TxRFBandwidth' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Transmit RF bandwidth of the transmit pulse in Hz.')
    TxFreqStart = _FloatDescriptor(
        'TxFreqStart', required=('TxFreqStart' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Transmit Start frequency for Linear FM waveform in Hz, may be relative to reference frequency.')
    TxFMRate = _FloatDescriptor(
        'TxFMRate', required=('TxFMRate' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Transmit FM rate for Linear FM waveform in Hz/second.')
    RcvWindowLength = _FloatDescriptor(
        'RcvWindowLength', required=('RcvWindowLength' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Receive window duration in seconds.')
    ADCSampleRate = _FloatDescriptor(
        'ADCSampleRate', required=('ADCSampleRate' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Analog-to-Digital Converter sampling rate in samples/second.')
    RcvIFBandwidth = _FloatDescriptor(
        'RcvIFBandwidth', required=('RcvIFBandwidth' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Receive IF bandwidth in Hz.')
    RcvFreqStart = _FloatDescriptor(
        'RcvFreqStart', required=('RcvFreqStart' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Receive demodulation start frequency in Hz, may be relative to reference frequency.')
    RcvFMRate = _FloatDescriptor(
        'RcvFMRate', required=('RcvFMRate' in __required), strict=DEFAULT_STRICT,
        docstring=':float: Receive FM rate. Should be 0 if RcvDemodType = "CHIRP".')
    RcvDemodType = _StringEnumDescriptor(
        'RcvDemodType', _DEMOD_TYPE_VALUES, required=('RcvFMRate' in __required), strict=DEFAULT_STRICT,
        docstring=""":str: Receive demodulation used when Linear FM waveform is used on transmit.
* 'STRETCH' - De-ramp on Receive demodulation.
* 'CHIRP'   - No De-ramp On Receive demodulation""")

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('TxPulseLength', 'TxRFBandwidth', 'TxFreqStart', 'TxFMRate',
            'RcvDemodType', 'RcvWindowLength', 'ADCSampleRate', 'RcvIFBandwidth', 'RcvFreqStart', 'RcvFMRate'), and
            none are required.
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
    # other class variables
    _POLARIZATION2_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER')
    # descriptors
    WFIndex = _IntegerDescriptor(
        'WFIndex', required=True, strict=DEFAULT_STRICT, docstring=':int: The waveform number for this step.')
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION2_VALUES, required=False, strict=DEFAULT_STRICT,
        docstring=':str: Transmit signal polarization for this step.')
    index = _IntegerDescriptor('index', required=True, strict=DEFAULT_STRICT, docstring=':int: The step index')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('WFIndex', 'TxPolarization', 'index'), and 'index' is required.
        """
        super(TxStepType, self).__init__(**kwargs)


class ChanParametersType(Serializable):
    """Transmit receive sequence step details"""
    __fields = ('TxRcvPolarization', 'RcvAPCIndex', 'index')
    __required = ('TxRcvPolarization', 'index', )
    # other class variables
    _DUAL_POLARIZATION_VALUES = (
        'V:V', 'V:H', 'H:V', 'H:H', 'RHC:RHC', 'RHC:LHC', 'LHC:RHC', 'LHC:LHC', 'OTHER', 'UNKNOWN')
    # descriptors
    TxRcvPolarization = _StringEnumDescriptor(
        'TxRcvPolarization', _DUAL_POLARIZATION_VALUES, required=True, strict=DEFAULT_STRICT,
        docstring=':str: Combined Transmit and Receive signal polarization for the channel.')
    RcvAPCIndex = _IntegerDescriptor(
        'RcvAPCIndex', required=True, strict=DEFAULT_STRICT,
        docstring=':int: Index of the Receive Aperture Phase Center (Rcv APC). Only include if Receive APC position '
                  'polynomial(s) are included.')
    index = _IntegerDescriptor(
        'index', required=True, strict=DEFAULT_STRICT, docstring=':int: The parameter index')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('TxRcvPolarization', 'RcvAPCIndex', 'index'), and
            ('TxRcvPolarization', 'index') are required.
        """
        super(ChanParametersType, self).__init__(**kwargs)


class ReferencePointType(Serializable):
    """The reference point definition"""
    __fields = ('ECF', 'Line', 'Sample', 'name')
    __required = __fields
    __set_as_attribute = ('name', )
    # descriptors
    ECF = _SerializableDescriptor(
        'ECF', XYZType, required=('ECF' in __required), strict=DEFAULT_STRICT,
        docstring=':XYZType: The geographical coordinates for the reference point')
    Line = _FloatDescriptor(
        'Line', required=('Line' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The Line?')  # TODO: what is this?
    Sample = _FloatDescriptor(
        'Sample', required=('Sample' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The Sample?')
    name = _StringDescriptor(
        'name', required=('name' in __required), strict=DEFAULT_STRICT,
        docstring=':str: The reference point name')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('ECF', 'Line', 'Sample', 'name'), all required.
        """
        super(ReferencePointType, self).__init__(**kwargs)


class XDirectionType(Serializable):
    """The X direction of the collect"""
    __fields = ('UVectECF', 'LineSpacing', 'NumLines', 'FirstLine')
    __required = __fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType, required=('UVectECF' in __required), strict=DEFAULT_STRICT,
        docstring=':XYZType: The unit vector')
    LineSpacing = _FloatDescriptor(
        'LineSpacing', required=('LineSpacing' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The collection line spacing in meters.')
    NumLines = _IntegerDescriptor(
        'NumLines', required=('NumLines' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The number of lines')
    FirstLine = _IntegerDescriptor(
        'FirstLine', required=('FirstLine' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The first line')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('UVectECF', 'LineSpacing', 'NumLines', 'FirstLine'), all required.
        """
        super(XDirectionType, self).__init__(**kwargs)


class YDirectionType(Serializable):
    """The Y direction of the collect"""
    __fields = ('UVectECF', 'LineSpacing', 'NumSamples', 'FirstSample')
    __required = __fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType, required=('UVectECF' in __required), strict=DEFAULT_STRICT,
        docstring=':XYZType: The unit vector')
    LineSpacing = _FloatDescriptor(
        'LineSpacing', required=('LineSpacing' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The collection line spacing in meters.')
    NumSamples = _IntegerDescriptor(
        'NumSamples', required=('NumSamples' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The number of samples')
    FirstSample = _IntegerDescriptor(
        'FirstSample', required=('FirstSample' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The first sample')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('UVectECF', 'LineSpacing', 'NumSamples', 'FirstSample'), all required.
        """
        super(YDirectionType, self).__init__(**kwargs)


class SegmentArrayElement(Serializable):
    """The reference point definition"""
    __fields = ('StartLine', 'StartSample', 'EndLine', 'EndSample', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    StartLine = _IntegerDescriptor(
        'StartLine', required=('StartLine' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The starting line number.')
    StartSample = _IntegerDescriptor(
        'StartSample', required=('StartSample' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The starting sample number.')
    EndLine = _IntegerDescriptor(
        'EndLine', required=('EndLine' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The ending line number.')
    EndSample = _IntegerDescriptor(
        'EndSample', required=('EndSample' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The ending sample number.')
    index = _IntegerDescriptor(
        'index', required=('index' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The array index.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('ECF', 'Line', 'Sample', 'name'), all required.
        """
        super(SegmentArrayElement, self).__init__(**kwargs)


class ReferencePlaneType(Serializable):
    """The reference plane"""
    __fields = ('RefPt', 'XDir', 'YDir', 'SegmentList', 'Orientation')
    __required = ('RefPt', 'XDir', 'YDir')
    __collections_tags = {'SegmentList': {'array': True, 'child_tag': 'SegmentList'}}
    # other class variable
    _ORIENTATION_VALUES = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'ARBITRARY')
    # descriptors
    RefPt = _SerializableDescriptor(
        'RefPt', ReferencePointType, required=('RefPt' in __required), strict=DEFAULT_STRICT,
        docstring=':ReferencePointType: The reference point.')
    XDir = _SerializableDescriptor(
        'XDir', XDirectionType, required=('XDir' in __required), strict=DEFAULT_STRICT,
        docstring=':XDirectionType: The X direction collection plane parameters.')
    YDir = _SerializableDescriptor(
        'YDir', YDirectionType, required=('YDir' in __required), strict=DEFAULT_STRICT,
        docstring=':YDirectionType: The Y direction collection plane parameters.')
    SegmentList = _SerializableArrayDescriptor(
        'SegmentList', SegmentArrayElement, __collections_tags['SegmentList'],
        required=('SegmentList' in __required), strict=DEFAULT_STRICT,
        docstring=':SegmentArrayElement: The segment **list**.')
    Orientation = _StringEnumDescriptor(
        'Orientation', _ORIENTATION_VALUES, required=('Orientation' in __required), strict=DEFAULT_STRICT,
        docstring=':str: The orientation enum, which takes values in {}.'.format(_ORIENTATION_VALUES))

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('RefPt', 'XDir', 'YDir', 'SegmentList', 'Orientation'), and
            ('RefPt', 'XDir', 'YDir') are required.
        """
        super(ReferencePlaneType, self).__init__(**kwargs)


class AreaType(Serializable):
    """The collection area"""
    __fields = ('Corner', 'Plane')
    __required = ('Corner', )
    __collections_tags = {
        'Corner': {'array': False, 'child_tag': 'ACP'},}
    # descriptors
    Corner = _SerializableArrayDescriptor(
        'Corner', LatLonHAECornerRestrictionType, __collections_tags['Corner'],
        required=('Corner' in __required), strict=DEFAULT_STRICT, minimum_length=4, maximum_length=4,
        docstring=':LatLonHAECornerRestrictionType: The collection area corner point definition array.')
    Plane = _SerializableDescriptor(
        'Plane', ReferencePlaneType, required=('Plane' in __required), strict=DEFAULT_STRICT,
        docstring=':ReferencePlaneType: The collection area reference plane.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Corner', 'Plane'), and 'Corner' is required.
        """
        super(AreaType, self).__init__(**kwargs)


class RadarCollectionType(Serializable):
    """"""
    __tag = 'RadarCollection'
    __fields = ('TxFrequency', 'RefFreqIndex', 'Waveform', 'TxPolarization', 'TxSequence', 'RcvChannels',
                'Area', 'Parameters')
    __required = ('TxFrequency', 'TxPolarization', 'RcvChannels')
    __collections_tags = {
        'Waveform': {'array': True, 'child_tag': 'WFParameters'},
        'TxSequence': {'array': True, 'child_tag': 'TxStep'},
        'RcvChannels': {'array': True, 'child_tag': 'RcvChannels'},
        'Parameters': {'array': False, 'child_tag': 'Parameters'}}
    # other class variables
    _POLARIZATION1_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER', 'UNKNOWN', 'SEQUENCE')
    # descriptors
    TxFrequencyType = _SerializableDescriptor(
        'TxFrequency', TxFrequencyType, required=('TxFrequency' in __required), strict=DEFAULT_STRICT,
        docstring=':TxFrequencyType: The transmit frequency range')
    RefFreqIndex = _IntegerDescriptor(
        'RefFreqIndex', required=('RefFreqIndex' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The reference frequency index, if applicable.')
    Waveform = _SerializableArrayDescriptor(
        'Waveform', WaveformParametersType, __collections_tags['Waveform'], required=('Waveform' in __required),
        strict=DEFAULT_STRICT, minimum_length=1,
        docstring=':WaveformParametersType: The waveform parameters **array**.')
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION1_VALUES, required=('TxPolarization' in __required), strict=DEFAULT_STRICT,
        docstring=':str: The transmit polarization')  # TODO: iff SEQUENCE, then TxSequence is defined?
    TxSequence = _SerializableArrayDescriptor(
        'TxSequence', TxStepType, __collections_tags['TxSequence'], required=('TxSequence' in __required),
        strict=DEFAULT_STRICT, minimum_length=1,
        docstring=':TxStepType: The transmit sequence parameters **array**.')
    RcvChannels = _SerializableArrayDescriptor(
        'RcvChannels', ChanParametersType, __collections_tags['RcvChannels'],
        required=('RcvChannels' in __required), strict=DEFAULT_STRICT, minimum_length=1,
        docstring=':ChanParametersType: Transmit receive sequence step details **array**')
    Area = _SerializableDescriptor(
        'Area', AreaType, required=('Area' in __required), strict=DEFAULT_STRICT,
        docstring='The collection area')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags['Parameters'],
        required=('Parameters' in __required), strict=DEFAULT_STRICT,
        docstring=':ParameterType: A parameters **list**')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are
            ('TxFrequency', 'RefFreqIndex', 'Waveform', 'TxPolarization', 'TxSequence', 'RcvChannels', 'Area',
            'Parameters'), and ('TxFrequency', 'TxPolarization', 'RcvChannels') are required.
        """
        super(RadarCollectionType, self).__init__(**kwargs)


###############
# ImageFormationType section


class RcvChanProcType(Serializable):
    """"""
    __fields = ('NumChanProc', 'PRFScaleFactor', 'ChanIndices')
    __required = ('NumChanProc', 'ChanIndices')  # TODO: make proper descriptor
    __collections_tags = {
        'ChanIndices': {'array': False, 'child_tag': 'ChanIndex'}}
    # descriptors
    NumChanProc = _IntegerDescriptor(
        'NumChanProc', required=('NumChanProc' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The NumChanProc')
    PRFScaleFactor = _FloatDescriptor(
        'PRFScaleFactor', required=('PRFScaleFactor' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The (optional) PRF scale factor.')
    ChanIndices = _IntegerListDescriptor(
        'ChanIndices', __collections_tags['ChanIndices'], required=('ChanIndices' in __required), strict=DEFAULT_STRICT,
        docstring=':int: The channel index **list**.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('NumChanProc', 'PRFScaleFactor', 'ChanIndices'), and
            ('NumChanProc', 'ChanIndices') are required.
        """
        super(RcvChanProcType, self).__init__(**kwargs)


class TxFrequencyProcType(Serializable):
    """The transmit frequency range"""
    __tag = 'TxFrequencyProc'
    __fields = ('MinProc', 'MaxProc')
    __required = __fields
    # descriptors
    MinProc = _FloatDescriptor(
        'MinProc', required=('MinProc' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The transmit minimum frequency in Hz.')
    MaxProc = _FloatDescriptor(
        'MaxProc', required=('MaxProc' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The transmit maximum frequency in Hz.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('MinProc', 'MaxProc'), all required.
        """
        super(TxFrequencyProcType, self).__init__(**kwargs)


class ProcessingType(Serializable):
    """The transmit frequency range"""
    __tag = 'Processing'
    __fields = ('Type', 'Applied', 'Parameters')
    __required = ('Type', 'Applied')
    __collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Type = _StringDescriptor(
        'Type', required=('Type' in __required), strict=DEFAULT_STRICT, docstring=':str: The type string.')
    Applied = _BooleanDescriptor(
        'Applied', required=('Applied' in __required), strict=DEFAULT_STRICT,
        docstring=':bool: Whether the given type has been applied')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags['Parameters'],
        required=('Parameters' in __required), strict=DEFAULT_STRICT,
        docstring=':ParameterType: The parameters **list**.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('Type', 'Applied', 'Parameters'), and ('Type', 'Applied') required.
        """
        super(ProcessingType, self).__init__(**kwargs)


class DistortionType(Serializable):
    """Distortion"""
    __fields = (
        'CalibrationDate', 'A', 'F1', 'F2', 'Q1', 'Q2', 'Q3', 'Q4',
        'GainErrorA', 'GainErrorF1', 'GainErrorF2', 'PhaseErrorF1', 'PhaseErrorF2')
    __required = ('A', 'F1', 'Q1', 'Q2', 'F2', 'Q3', 'Q4')
    # descriptors
    CalibrationDate = _DateTimeDescriptor(
        'CalibrationDate', required=('CalibrationDate' in __required), strict=DEFAULT_STRICT,
        docstring=':numpy.datetime64: The (optional) calibration date.')
    A = _FloatDescriptor(
        'A', required=('A' in __required), strict=DEFAULT_STRICT, docstring=':float: The A attribute.')
    F1 = _ComplexDescriptor(
        'F1', required=('F1' in __required), strict=DEFAULT_STRICT, docstring=':complex: The F1 attribute.')
    F2 = _ComplexDescriptor(
        'F2', required=('F2' in __required), strict=DEFAULT_STRICT, docstring=':complex: The F2 attribute.')
    Q1 = _ComplexDescriptor(
        'Q1', required=('Q1' in __required), strict=DEFAULT_STRICT, docstring=':complex: The Q1 attribute.')
    Q2 = _ComplexDescriptor(
        'Q2', required=('Q2' in __required), strict=DEFAULT_STRICT, docstring=':complex: The Q2 attribute.')
    Q3 = _ComplexDescriptor(
        'Q3', required=('Q3' in __required), strict=DEFAULT_STRICT, docstring=':complex: The Q3 attribute.')
    Q4 = _ComplexDescriptor(
        'Q4', required=('Q4' in __required), strict=DEFAULT_STRICT, docstring=':complex: The Q4 attribute.')
    GainErrorA = _FloatDescriptor(
        'GainErrorA', required=('GainErrorA' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The GainErrorA attribute.')
    GainErrorF1 = _FloatDescriptor(
        'GainErrorF1', required=('GainErrorF1' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The GainErrorF1 attribute.')
    GainErrorF2 = _FloatDescriptor(
        'GainErrorF2', required=('GainErrorF2' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The GainErrorF2 attribute.')
    PhaseErrorF1 = _FloatDescriptor(
        'PhaseErrorF1', required=('PhaseErrorF1' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The PhaseErrorF1 attribute.')
    PhaseErrorF2 = _FloatDescriptor(
        'PhaseErrorF2', required=('PhaseErrorF2' in __required), strict=DEFAULT_STRICT,
        docstring=':float: The PhaseErrorF2 attribute.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('CalibrationDate', 'A', 'F1', 'Q1', 'Q2', 'F2', 'Q3', 'Q4',
            'GainErrorA', 'GainErrorF1', 'GainErrorF2', 'PhaseErrorF1', 'PhaseErrorF2'), and
            ('A', 'F1', 'Q1', 'Q2', 'F2', 'Q3', 'Q4') are required.
        """
        super(DistortionType, self).__init__(**kwargs)

class PolarizationCalibrationType(Serializable):
    """The polarization calibration"""
    __fields = ('DistortCorrectApplied', 'Distortion')
    __required = __fields
    # descriptors
    DistortCorrectApplied = _BooleanDescriptor(
        'DistortCorrectApplied', required=('DistortCorrectApplied' in __required), strict=DEFAULT_STRICT,
        docstring=':bool: Whether the distortion correction has been applied')
    Distortion = _SerializableDescriptor(
        'Distortion', DistortionType, required=('Distortion' in __required), strict=DEFAULT_STRICT,
        docstring=':DistortionType: The distortion parameters.')

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('DistortCorrectApplied', 'Distortion'), and both required.
        """
        super(PolarizationCalibrationType, self).__init__(**kwargs)


class ImageFormationType(Serializable):
    """The image formation parameters type"""
    __fields = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc', 'SegmentIdentifier',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus', 'Processings',
        'PolarizationCalibration')
    __required = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus')
    __collections_tags = {'Processings': {'array': False, 'child_tag': 'Processing'}}
    # descriptors
    # TODO: finish this here

    def __init__(self, **kwargs):
        """The constructor.
        :param dict kwargs: the valid keys are ('RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc',
            'TxFrequencyProc', 'SegmentIdentifier', 'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus',
            'RgAutofocus', 'Processings', 'PolarizationCalibration'), and all EXCEPT
            ('SegmentIdentifier', 'Processings', 'PolarizationCalibration')required.
        """
        super(ImageFormationType, self).__init__(**kwargs)


###############
# SCPCOAType section
###############
# RadiometricType section
###############
# AntParamType section
###############
# AntennaType section
###############
# ErrorStatisticsType section
###############
# MatchInfoType section
###############
# RgAzCompType section
###############
# PFAType section
###############
# RMAType section
####################################################################
# the SICD object


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
