"""
This module contains the base objects for use in base xml/serializable functionality.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from xml.etree import ElementTree
import json
from datetime import date, datetime
from collections import OrderedDict
import copy
import re
from io import StringIO
from typing import Dict

import numpy

from sarpy.compliance import bytes_to_string

try:
    from lxml import etree
except ImportError:
    etree = None


logger = logging.getLogger(__name__)
valid_logger = logging.getLogger('validation')

DEFAULT_STRICT = False


#################
# dom helper functions


def get_node_value(nod):
    """
    XML parsing helper for extracting text value from an ElementTree Element. No error checking performed.

    Parameters
    ----------
    nod : ElementTree.Element
        the xml dom element

    Returns
    -------
    str
        the string value of the node.
    """

    if nod.text is None:
        return None

    val = nod.text.strip()
    if len(val) == 0:
        return None
    else:
        return val


def create_new_node(doc, tag, parent=None):
    """
    XML ElementTree node creation helper function.

    Parameters
    ----------
    doc : ElementTree.ElementTree
        The xml Document object.
    tag : str
        Name/tag for new xml element.
    parent : None|ElementTree.Element
        The parent element for the new element. Defaults to the document root element if unspecified.
    Returns
    -------
    ElementTree.Element
        The new element populated as a child of `parent`.
    """

    if parent is None:
        parent = doc.getroot()  # what if there is no root?
    if parent is None:
        element = ElementTree.Element(tag)
        # noinspection PyProtectedMember
        doc._setroot(element)
        return element
    else:
        return ElementTree.SubElement(parent, tag)


def create_text_node(doc, tag, value, parent=None):
    """
    XML ElementTree text node creation helper function

    Parameters
    ----------
    doc : ElementTree.ElementTree
        The xml Document object.
    tag : str
        Name/tag for new xml element.
    value : str
        The value for the new element.
    parent : None|ElementTree.Element
        The parent element for the new element. Defaults to the document root element if unspecified.

    Returns
    -------
    ElementTree.Element
        The new element populated as a child of `parent`.
    """

    node = create_new_node(doc, tag, parent=parent)
    node.text = value
    return node


def find_first_child(node, tag, xml_ns, ns_key):
    """
    Finds the first child node

    Parameters
    ----------
    node : ElementTree.Element
    tag : str
    xml_ns : None|dict
    ns_key : None|str
    """

    if xml_ns is None:
        return node.find(tag)
    elif ns_key is None:
        return node.find('default:{}'.format(tag), xml_ns)
    else:
        return node.find('{}:{}'.format(ns_key, tag), xml_ns)


def find_children(node, tag, xml_ns, ns_key):
    """
    Finds the collection of children nodes

    Parameters
    ----------
    node : ElementTree.Element
    tag : str
    xml_ns : None|dict
    ns_key : None|str
    """

    if xml_ns is None:
        return node.findall(tag)
    elif ns_key is None:
        return node.findall('default:{}'.format(tag), xml_ns)
    else:
        return node.findall('{}:{}'.format(ns_key, tag), xml_ns)


def parse_xml_from_string(xml_string):
    """
    Parse the ElementTree root node and xml namespace dict from an xml string.

    Parameters
    ----------
    xml_string : str|bytes

    Returns
    -------
    root_node: ElementTree.Element
    xml_ns: Dict[str, str]
    """

    xml_string = bytes_to_string(xml_string, encoding='utf-8')

    root_node = ElementTree.fromstring(xml_string)
    # define the namespace dictionary
    xml_ns = dict([node for _, node in ElementTree.iterparse(StringIO(xml_string), events=('start-ns',))])
    if len(xml_ns.keys()) == 0:
        xml_ns = None
    elif '' in xml_ns:
        xml_ns['default'] = xml_ns['']
    else:
        # default will be the namespace for the root node
        namespace_match = re.match(r'\{.*\}', root_node.tag)
        if namespace_match is None:
            raise ValueError('Trouble finding the default namespace for tag {}'.format(root_node.tag))
        xml_ns['default'] = namespace_match[0][1:-1]
    return root_node, xml_ns


def parse_xml_from_file(xml_file_path):
    """
    Parse the ElementTree root node and xml namespace dict from an xml file.

    Parameters
    ----------
    xml_file_path : str

    Returns
    -------
    root_node: ElementTree.Element
    xml_ns: Dict[str, str]
    """

    with open(xml_file_path, 'rb') as fi:
        xml_bytes = fi.read()
    return parse_xml_from_string(xml_bytes)


def validate_xml_from_string(xml_string, xsd_path, output_logger=None):
    """
    Validate an xml string against a given xsd document.

    Parameters
    ----------
    xml_string : str|bytes
    xsd_path : str
        The path to the relevant xsd document.
    output_logger
        A desired output logger.

    Returns
    -------
    bool
        `True` if valid, `False` otherwise. Failure reasons will be
        logged at `'error'` level by the module.
    """

    if etree is None:
        raise ImportError(
            'The lxml package was not successfully imported,\n\t'
            'and this xml validation requires lxml.')

    xml_doc = etree.fromstring(xml_string)
    xml_schema = etree.XMLSchema(file=xsd_path)
    validity = xml_schema.validate(xml_doc)
    if not validity:
        for entry in xml_schema.error_log:
            msg = 'XML validation error on line {}\n\t{}'.format(
                entry.line, entry.message.encode('utf-8'))
            if output_logger is None:
                logger.error(msg)
            else:
                output_logger.error(msg)
    return validity


def validate_xml_from_file(xml_path, xsd_path, output_logger=None):
    """
    Validate an xml string against a given xsd document.

    Parameters
    ----------
    xml_path : str
        The path to the relevant xml file
    xsd_path : str
        The path to the relevant xsd document.
    output_logger
        A desired output logger.

    Returns
    -------
    bool
        `True` if valid, `False` otherwise. Failure reasons will be
        logged at `'error'` level by the module.
    """

    with open(xml_path, 'rb') as fi:
        xml_bytes = fi.read()

    return validate_xml_from_string(xml_bytes, xsd_path, output_logger=output_logger)


###
# parsing functions - for reusable functionality in descriptors or other property definitions


def parse_str(value, name, instance):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, ElementTree.Element):
        node_value = get_node_value(value)
        return "" if node_value is None else node_value
    else:
        raise TypeError(
            'field {} of class {} requires a string value.'.format(name, instance.__class__.__name__))


def parse_bool(value, name, instance):
    def parse_string(val):
        if val.lower() in ['0', 'false']:
            return False
        elif val.lower() in ['1', 'true']:
            return True
        else:
            raise ValueError(
                'Boolean field {} of class {} cannot assign from string value {}. '
                'It must be one of ["0", "false", "1", "true"]'.format(name, instance.__class__.__name__, val))

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    elif isinstance(value, int) or isinstance(value, numpy.bool_):
        return bool(value)
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization
        return parse_string(get_node_value(value))
    elif isinstance(value, str):
        return parse_string(value)
    else:
        raise ValueError('Boolean field {} of class {} cannot assign from type {}.'.format(
            name, instance.__class__.__name__, type(value)))


def parse_int(value, name, instance):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization
        return parse_int(get_node_value(value), name, instance)
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError as e:
            logger.warning(
                'Got non-integer value {}\n\t'
                'for integer valued field {} of class {}'.format(
                    value, name, instance.__class__.__name__))
            # noinspection PyBroadException
            try:
                return int(float(value))
            except Exception:
                raise e
    else:
        # user or json deserialization
        return int(value)


# noinspection PyUnusedLocal
def parse_float(value, name, instance):
    if value is None:
        return None
    if isinstance(value, float):
        return value
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization
        return float(get_node_value(value))
    else:
        # user or json deserialization
        return float(value)


def parse_complex(value, name, instance):
    if value is None:
        return None
    if isinstance(value, complex):
        return value
    elif isinstance(value, ElementTree.Element):
        xml_ns = getattr(instance, '_xml_ns', None)
        # noinspection PyProtectedMember
        if hasattr(instance, '_child_xml_ns_key') and name in instance._child_xml_ns_key:
            # noinspection PyProtectedMember
            xml_ns_key = instance._child_xml_ns_key[name]
        else:
            xml_ns_key = getattr(instance, '_xml_ns_key', None)
        # from XML deserialization
        rnode = find_children(value, 'Real', xml_ns, xml_ns_key)
        inode = find_children(value, 'Imag', xml_ns, xml_ns_key)

        if len(rnode) != 1:
            raise ValueError(
                'There must be exactly one Real component of a complex type node '
                'defined for field {} of class {}.'.format(name, instance.__class__.__name__))
        if len(inode) != 1:
            raise ValueError(
                'There must be exactly one Imag component of a complex type node '
                'defined for field {} of class {}.'.format(name, instance.__class__.__name__))
        real = float(get_node_value(rnode[0]))
        imag = float(get_node_value(inode[0]))
        return complex(real, imag)
    elif isinstance(value, dict):
        # from json deserialization
        real = None
        for key in ['re', 'real', 'Real']:
            real = value.get(key, real)
        imag = None
        for key in ['im', 'imag', 'Imag']:
            imag = value.get(key, imag)
        if real is None or imag is None:
            raise ValueError(
                'Cannot convert dict {} to a complex number for field {} of '
                'class {}.'.format(value, name, instance.__class__.__name__))
        return complex(real, imag)
    else:
        # from user - I can't imagine that this would ever work
        return complex(value)


def parse_datetime(value, name, instance, units='us'):
    if value is None:
        return None
    if isinstance(value, numpy.datetime64):
        return value
    elif isinstance(value, str):
        # handle Z timezone identifier explicitly - any timezone identifier is deprecated
        if value[-1] == 'Z':
            return numpy.datetime64(value[:-1], units)
        else:
            return numpy.datetime64(value, units)
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization - extract the string
        return parse_datetime(get_node_value(value), name, instance, units=units)
    elif isinstance(value, (date, datetime, numpy.int64, numpy.float64)):
        return numpy.datetime64(value, units)
    elif isinstance(value, int):
        # this is less safe, because the units are unknown...
        return numpy.datetime64(value, units)
    else:
        raise TypeError(
            'Field {} for class {} expects datetime convertible input, and '
            'got {}'.format(name, instance.__class__.__name__, type(value)))


def parse_serializable(value, name, instance, the_type):
    if value is None:
        return None
    if isinstance(value, the_type):
        return value
    elif isinstance(value, dict):
        return the_type.from_dict(value)
    elif isinstance(value, ElementTree.Element):
        xml_ns = getattr(instance, '_xml_ns', None)
        if hasattr(instance, '_child_xml_ns_key'):
            # noinspection PyProtectedMember
            xml_ns_key = instance._child_xml_ns_key.get(name, getattr(instance, '_xml_ns_key', None))
        else:
            xml_ns_key = getattr(instance, '_xml_ns_key', None)
        return the_type.from_node(value, xml_ns, ns_key=xml_ns_key)
    elif isinstance(value, (numpy.ndarray, list, tuple)):
        if issubclass(the_type, Arrayable):
            return the_type.from_array(value)
        else:
            raise TypeError(
                'Field {} of class {} is of type {} (not a subclass of Arrayable) and '
                'got an argument of type {}.'.format(name, instance.__class__.__name__, the_type, type(value)))
    else:
        raise TypeError(
            'Field {} of class {} is expecting type {}, but got an instance of incompatible '
            'type {}.'.format(name, instance.__class__.__name__, the_type, type(value)))


def parse_serializable_array(value, name, instance, child_type, child_tag):
    if value is None:
        return None
    if isinstance(value, child_type):
        # this is the child element
        return numpy.array([value, ], dtype='object')
    elif isinstance(value, numpy.ndarray):
        if value.dtype.name != 'object':
            if issubclass(child_type, Arrayable):
                return numpy.array([child_type.from_array(array) for array in value], dtype='object')
            else:
                raise ValueError(
                    'Attribute {} of array type functionality belonging to class {} got an ndarray of dtype {},'
                    'and child type is not a subclass of Arrayable.'.format(
                        name, instance.__class__.__name__, value.dtype))
        elif len(value.shape) != 1:
            raise ValueError(
                'Attribute {} of array type functionality belonging to class {} got an ndarray of shape {},'
                'but requires a one dimensional array.'.format(
                    name, instance.__class__.__name__, value.shape))
        elif not isinstance(value[0], child_type):
            raise TypeError(
                'Attribute {} of array type functionality belonging to class {} got an ndarray containing '
                'first element of incompatible type {}.'.format(
                    name, instance.__class__.__name__, type(value[0])))
        return value
    elif isinstance(value, ElementTree.Element):
        xml_ns = getattr(instance, '_xml_ns', None)
        if hasattr(instance, '_child_xml_ns_key'):
            # noinspection PyProtectedMember
            xml_ns_key = instance._child_xml_ns_key.get(name, getattr(instance, '_xml_ns_key', None))
        else:
            xml_ns_key = getattr(instance, '_xml_ns_key', None)
        # this is the parent node from XML deserialization
        size = int(value.attrib.get('size', -1))  # NB: Corner Point arrays don't have
        # extract child nodes at top level
        child_nodes = find_children(value, child_tag, xml_ns, xml_ns_key)

        if size == -1:  # fill in, if it's missing
            size = len(child_nodes)
        if len(child_nodes) != size:
            raise ValueError(
                'Attribute {} of array type functionality belonging to class {} got a ElementTree element '
                'with size attribute {}, but has {} child nodes with tag {}.'.format(
                    name, instance.__class__.__name__, size, len(child_nodes), child_tag))
        new_value = numpy.empty((size, ), dtype='object')
        for i, entry in enumerate(child_nodes):
            new_value[i] = child_type.from_node(entry, xml_ns, ns_key=xml_ns_key)
        return new_value
    elif isinstance(value, (list, tuple)):
        # this would arrive from users or json deserialization
        if len(value) == 0:
            return numpy.empty((0,), dtype='object')
        elif isinstance(value[0], child_type):
            return numpy.array(value, dtype='object')
        elif isinstance(value[0], dict):
            # NB: charming errors are possible here if something stupid has been done.
            return numpy.array([child_type.from_dict(node) for node in value], dtype='object')
        elif isinstance(value[0], (numpy.ndarray, list, tuple)):
            if issubclass(child_type, Arrayable):
                return numpy.array([child_type.from_array(array) for array in value], dtype='object')
            elif hasattr(child_type, 'Coefs'):
                return numpy.array([child_type(Coefs=array) for array in value], dtype='object')
            else:
                raise ValueError(
                    'Attribute {} of array type functionality belonging to class {} got an list '
                    'containing elements type {} and construction failed.'.format(
                        name, instance.__class__.__name__, type(value[0])))
        else:
            raise TypeError(
                'Attribute {} of array type functionality belonging to class {} got a list containing first '
                'element of incompatible type {}.'.format(name, instance.__class__.__name__, type(value[0])))
    else:
        raise TypeError(
            'Attribute {} of array type functionality belonging to class {} got incompatible type {}.'.format(
                name, instance.__class__.__name__, type(value)))


def parse_serializable_list(value, name, instance, child_type):
    if value is None:
        return None
    if isinstance(value, child_type):
        # this is the child element
        return [value, ]

    xml_ns = getattr(instance, '_xml_ns', None)
    if hasattr(instance, '_child_xml_ns_key'):
        # noinspection PyProtectedMember
        xml_ns_key = instance._child_xml_ns_key.get(name, getattr(instance, '_xml_ns_key', None))
    else:
        xml_ns_key = getattr(instance, '_xml_ns_key', None)
    if isinstance(value, ElementTree.Element):
        # this is the child
        return [child_type.from_node(value, xml_ns, ns_key=xml_ns_key), ]
    elif isinstance(value, list) or isinstance(value[0], child_type):
        if len(value) == 0:
            return value
        elif isinstance(value[0], child_type):
            return [entry for entry in value]
        elif isinstance(value[0], dict):
            # NB: charming errors are possible if something stupid has been done.
            return [child_type.from_dict(node) for node in value]
        elif isinstance(value[0], ElementTree.Element):
            return [child_type.from_node(node, xml_ns, ns_key=xml_ns_key) for node in value]
        else:
            raise TypeError(
                'Field {} of list type functionality belonging to class {} got a '
                'list containing first element of incompatible type '
                '{}.'.format(name, instance.__class__.__name__, type(value[0])))
    else:
        raise TypeError(
            'Field {} of class {} got incompatible type {}.'.format(
                name, instance.__class__.__name__, type(value)))


def parse_parameters_collection(value, name, instance):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    elif isinstance(value, list):
        out = OrderedDict()
        if len(value) == 0:
            return out
        if isinstance(value[0], ElementTree.Element):
            for entry in value:
                out[entry.attrib['name']] = get_node_value(entry)
            return out
        else:
            raise TypeError(
                'Field {} of list type functionality belonging to class {} got a '
                'list containing first element of incompatible type '
                '{}.'.format(name, instance.__class__.__name__, type(value[0])))
    else:
        raise TypeError(
            'Field {} of class {} got incompatible type {}.'.format(
                name, instance.__class__.__name__, type(value)))


##################
# Main class defining structure

class Serializable(object):
    """
    Basic abstract class specifying the serialization pattern. There are no clearly defined Python conventions
    for this issue. Every effort has been made to select sensible choices, but this is an individual effort.

    Notes
    -----
        All fields MUST BE LISTED in the `_fields` tuple. Everything listed in `_required` tuple will be checked
        for inclusion in `_fields` tuple. Note that special care must be taken to ensure compatibility of `_fields`
        tuple, if inheriting from an extension of this class.
    """
    _fields = ()
    """collection of field names"""
    _required = ()
    """subset of `_fields` defining the required (for the given object, according to the sicd standard) fields"""
    _tag_overide = {}
    """On occasion, the xml tag and the corresponding variable name may need to differ. 
    This dictionary should be populated as `{<variable name> : <tag name>}`."""

    _collections_tags = {}
    """
    Entries only appropriate for list/array type objects. Entry formatting:

    * `{'array': True, 'child_tag': <child_name>}` represents an array object, which will have int attribute `size`.
      It has *size* children with tag=<child_name>, each of which has an attribute `index`, which is not always an
      integer. Even when it is an integer, it apparently sometimes follows the matlab convention (1 based), and
      sometimes the standard convention (0 based). In this case, I will deserialize as though the objects are
      properly ordered, and the deserialized objects will have the `index` property from the xml, but it will not
      be used to determine array order - which will be simply carried over from file order.

    * `{'array': False, 'child_tag': <child_name>}` represents a collection of things with tag=<child_name>.
      This entries are not directly below one coherent container tag, but just dumped into an object.
      For example of such usage search for "Parameter" in the SICD standard.

      In this case, I've have to create an ephemeral variable in the class that doesn't exist in the standard,
      and it's not clear what the intent is for this unstructured collection, so used a list object.
      For example, I have a variable called `Parameters` in `CollectionInfoType`, whose job it to contain the
      parameters objects.
    """

    _numeric_format = {}
    """define dict entries of numeric formatting for serialization"""
    _set_as_attribute = ()
    """serialize these fields as xml attributes"""
    _choice = ()
    """
    Entries appropriate for choice selection between attributes. Entry formatting:

    * `{'required': True, 'collection': <tuple of attribute names>}` - indicates that EXACTLY only one of the 
      attributes should be populated.

    * `{'required': False, 'collection': <tuple of attribute names>}` - indicates that no more than one of the 
      attributes should be populated.
    """
    _child_xml_ns_key = {}
    """
    The expected namespace key for attributes. No entry indicates the default namespace. 
    This is important for SIDD handling, but not required for SICD handling.
    """

    # NB: it may be good practice to use __slots__ to further control class functionality?

    def __init__(self, **kwargs):
        """
        The default constructor. For each attribute name in `self._fields`, fetches the value (or None) from
        the `kwargs` dict, and sets the class instance attribute value. The details for attribute value validation,
        present for virtually every attribute, will be implemented specifically as descriptors.

        Parameters
        ----------
        **kwargs :
            the keyword arguments dictionary - the possible entries match the attributes.
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        unexpected_args = [key for key in kwargs if key not in self._fields and key[0] != '_']
        if len(unexpected_args) > 0:
            raise ValueError(
                'Received unexpected construction argument {} for attribute '
                'collection {}'.format(unexpected_args, self._fields))

        for attribute in self._fields:
            if attribute in kwargs:
                try:
                    setattr(self, attribute, kwargs.get(attribute, None))
                except AttributeError:
                    # NB: this is included to allow for read only properties without breaking the paradigm
                    #   Silently catching errors can potentially cover up REAL issues.
                    pass

    def __str__(self):
        return '{}(**{})'.format(self.__class__.__name__, json.dumps(self.to_dict(check_validity=False), indent=1))

    def __repr__(self):
        return '{}(**{})'.format(self.__class__.__name__, self.to_dict(check_validity=False))

    def __setattr__(self, key, value):
        if not (key.startswith('_') or (key in self._fields) or hasattr(self.__class__, key) or hasattr(self, key)):
            # not expected attribute - descriptors, properties, etc
            logger.warning(
                'Class {} instance receiving unexpected attribute {}.\n\t'
                'Ensure that this is not a typo of an expected field name.'.format(self.__class__.__name__, key))
        object.__setattr__(self, key, value)

    def __getstate__(self):
        """
        Method for allowing copying and/or pickling of state.

        Returns
        -------
        dict
            The dict representation for the object.
        """

        return self.to_dict(check_validity=False, strict=False)

    def __setstate__(self, the_dict):
        """
        Method for reconstructing from the serialized state.
        """

        return self.__init__(**the_dict)

    def set_numeric_format(self, attribute, format_string):
        """Sets the numeric format string for the given attribute.

        Parameters
        ----------
        attribute : str
            attribute for which the format applies - must be in `_fields`.
        format_string : str
            format string to be applied
        Returns
        -------
        None
        """
        # Extend this to include format function capabilities. Maybe numeric_format is not the right name?
        if attribute not in self._fields:
            raise ValueError('attribute {} is not permitted for class {}'.format(attribute, self.__class__.__name__))
        self._numeric_format[attribute] = format_string

    def _get_formatter(self, attribute):
        """Return a formatting function for the given attribute. This will default to `str` if no other
        option is presented.

        Parameters
        ----------
        attribute : str
            the given attribute name

        Returns
        -------
        Callable
            format function
        """

        entry = self._numeric_format.get(attribute, None)
        if isinstance(entry, str):
            fmt_str = '{0:' + entry + '}'
            return fmt_str.format
        elif callable(entry):
            return entry
        else:
            return str

    def log_validity_error(self, msg):
        """
        Log a validity check error message.

        Parameters
        ----------
        msg : str
        """

        valid_logger.error('{}: {}'.format(self.__class__.__name__, msg))

    def log_validity_warning(self, msg):
        """
        Log a validity check warning message.

        Parameters
        ----------
        msg : str
        """

        valid_logger.warning('{}: {}'.format(self.__class__.__name__, msg))

    def log_validity_info(self, msg):
        """
        Log a validation info message.

        Parameters
        ----------
        msg : str
        """

        valid_logger.info('{}: {}'.format(self.__class__.__name__, msg))

    def is_valid(self, recursive=False, stack=False):
        """Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `_required`) are not `None`.

        Parameters
        ----------
        recursive : bool
            True if we recursively check that child are also valid. This may result in verbose (i.e. noisy) logging.
        stack : bool
            Print a recursive error message?

        Returns
        -------
        bool
            condition for validity of this element
        """

        all_required = self._basic_validity_check()
        if not recursive:
            return all_required

        valid_children = self._recursive_validity_check(stack=stack)
        return all_required & valid_children

    def _basic_validity_check(self):
        """
        Perform the basic validity check on the direct attributes with no recursive checking.

        Returns
        -------
        bool
             True if all requirements *AT THIS LEVEL* are satisfied, otherwise False.
        """

        all_required = True
        for attribute in self._required:
            present = (getattr(self, attribute) is not None)
            if not present:
                self.log_validity_error("Missing required attribute {}".format(attribute))
            all_required &= present

        choices = True
        for entry in self._choice:
            required = entry.get('required', False)
            collect = entry['collection']
            # verify that no more than one of the entries in collect is set.
            present = []
            for attribute in collect:
                if getattr(self, attribute) is not None:
                    present.append(attribute)
            if len(present) == 0 and required:
                self.log_validity_error(
                    "Exactly one of the attributes {} should be set, but none are set".format(collect))
                choices = False
            elif len(present) > 1:
                self.log_validity_error(
                    "Exactly one of the attributes {} should be set, but multiple ({}) are set".format(collect,
                                                                                                       present))
                choices = False

        return all_required and choices

    def _recursive_validity_check(self, stack=False):
        """
        Perform a recursive validity check on all present attributes.

        Parameters
        ----------
        stack : bool
            Print a recursive error message?

        Returns
        -------
        bool
             True if requirements are recursively satisfied *BELOW THIS LEVEL*, otherwise False.
        """

        def check_item(value):
            if isinstance(value, (Serializable, SerializableArray)):
                return value.is_valid(recursive=True, stack=stack)
            return True

        valid_children = True
        for attribute in self._fields:
            val = getattr(self, attribute)
            good = True
            if isinstance(val, (Serializable, SerializableArray)):
                good = check_item(val)
            elif isinstance(val, list):
                for entry in val:
                    good &= check_item(entry)
            # any issues will be logged as discovered, but should we help with the "stack"?
            if not good and stack:
                self.log_validity_error(
                    "Issue discovered with attribute {} of type {}.".format(attribute, type(val)))
            valid_children &= good
        return valid_children

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        """For XML deserialization.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance
        xml_ns : None|dict
            The xml namespace dictionary.
        ns_key : None|str
            The xml namespace key. If `xml_ns` is None, then this is ignored. If `None` and `xml_ns` is not None,
            then the string `default` will be used. This will be recursively passed down,
            unless overridden by an entry of the cls._child_xml_ns_key dictionary.
        kwargs : None|dict
            `None` or dictionary of previously serialized attributes. For use in inheritance call, when certain
            attributes require specific deserialization.
        Returns
        -------
            Corresponding class instance
        """

        if len(node) == 0 and len(node.attrib) == 0:
            logger.warning(
                'There are no children or attributes associated\n\t'
                'with node {}\n\t'
                'for class {}.'.format(node, cls))
            # return None

        def handle_attribute(the_attribute, the_tag, the_xml_ns_key):
            if the_xml_ns_key is not None:  # handle namespace, if necessary
                fetch_tag = '{' + xml_ns[the_xml_ns_key] + '}' + the_tag
            else:
                fetch_tag = the_tag
            kwargs[the_attribute] = node.attrib.get(fetch_tag, None)

        def handle_single(the_attribute, the_tag, the_xml_ns_key):
            kwargs[the_attribute] = find_first_child(node, the_tag, xml_ns, the_xml_ns_key)

        def handle_list(attrib, ch_tag, the_xml_ns_key):
            cnodes = find_children(node, ch_tag, xml_ns, the_xml_ns_key)
            if len(cnodes) > 0:
                kwargs[attrib] = cnodes

        if kwargs is None:
            kwargs = {}
        kwargs['_xml_ns'] = xml_ns
        kwargs['_xml_ns_key'] = ns_key

        if not isinstance(kwargs, dict):
            raise ValueError(
                "Named input argument kwargs for class {} must be dictionary instance".format(cls))

        for attribute in cls._fields:
            if attribute in kwargs:
                continue

            kwargs[attribute] = None
            # This value will be replaced if tags are present
            # Note that we want to try explicitly setting to None to trigger descriptor behavior
            # for required fields (warning or error)

            base_tag_name = cls._tag_overide.get(attribute, attribute)

            # determine any expected xml namespace for the given entry
            if attribute in cls._child_xml_ns_key:
                xml_ns_key = cls._child_xml_ns_key[attribute]
            else:
                xml_ns_key = ns_key
            # verify that the xml namespace will work
            if xml_ns_key is not None:
                if xml_ns is None:
                    raise ValueError('Attribute {} in class {} expects an xml namespace entry of {}, '
                                     'but xml_ns is None.'.format(attribute, cls, xml_ns_key))
                elif xml_ns_key not in xml_ns:
                    raise ValueError('Attribute {} in class {} expects an xml namespace entry of {}, '
                                     'but xml_ns does not contain this key.'.format(attribute, cls, xml_ns_key))

            if attribute in cls._set_as_attribute:
                xml_ns_key = cls._child_xml_ns_key.get(attribute, None)
                handle_attribute(attribute, base_tag_name, xml_ns_key)
            elif attribute in cls._collections_tags:
                # it's a collection type parameter
                array_tag = cls._collections_tags[attribute]
                array = array_tag.get('array', False)
                child_tag = array_tag.get('child_tag', None)
                if array:
                    handle_single(attribute, base_tag_name, xml_ns_key)
                elif child_tag is not None:
                    handle_list(attribute, child_tag, xml_ns_key)
                else:
                    # the metadata is broken
                    raise ValueError(
                        'Attribute {} in class {} is listed in the _collections_tags dictionary, but the '
                        '`child_tag` value is either not populated or None.'.format(attribute, cls))
            else:
                # it's a regular property
                handle_single(attribute, base_tag_name, xml_ns_key)
        return cls.from_dict(kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        """For XML serialization, to a dom element.

        Parameters
        ----------
        doc : ElementTree.ElementTree
            The xml Document
        tag : None|str
            The tag name. Defaults to the value of `self._tag` and then the class name if unspecified.
        ns_key : None|str
            The namespace prefix. This will be recursively passed down, unless overridden by an entry in the
            _child_xml_ns_key dictionary.
        parent : None|ElementTree.Element
            The parent element. Defaults to the document root element if unspecified.
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        ElementTree.Element
            The constructed dom element, already assigned to the parent element.
        """

        def serialize_attribute(node, the_tag, val, format_function, the_xml_ns_key):
            if the_xml_ns_key is None or the_xml_ns_key == 'default':
                node.attrib[the_tag] = format_function(val)
            else:
                node.attrib['{}:{}'.format(the_xml_ns_key, the_tag)] = format_function(val)

        def serialize_array(node, the_tag, ch_tag, val, format_function, size_attrib, the_xml_ns_key):
            if not isinstance(val, numpy.ndarray):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # _collections_tag or the descriptor at runtime
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be an array based on '
                    'the metadata in the _collections_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(val)))
            if not len(val.shape) == 1:
                # again, I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a one-dimensional numpy.ndarray, but it has shape {}'.format(
                        attribute, self.__class__.__name__, val.shape))
            if val.size == 0:
                return  # serializing an empty array is dumb

            if val.dtype.name == 'float64':
                if the_xml_ns_key is None:
                    anode = create_new_node(doc, the_tag, parent=node)
                else:
                    anode = create_new_node(doc, '{}:{}'.format(the_xml_ns_key, the_tag), parent=node)
                anode.attrib[size_attrib] = str(val.size)
                for i, val in enumerate(val):
                    vnode = create_text_node(doc, ch_tag, format_function(val), parent=anode)
                    vnode.attrib['index'] = str(i) if ch_tag == 'Amplitude' else str(i + 1)
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a numpy.ndarray of dtype float64 or object, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, val.dtype))

        def serialize_list(node, ch_tag, val, format_function, the_xml_ns_key):
            if not isinstance(val, list):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # _collections_tags or the descriptor?
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be a list based on '
                    'the metadata in the _collections_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(val)))
            if len(val) == 0:
                return  # serializing an empty list is dumb
            else:
                for entry in val:
                    if entry is not None:
                        serialize_plain(node, ch_tag, entry, format_function, the_xml_ns_key)

        def serialize_plain(node, the_tag, val, format_function, the_xml_ns_key):
            # may be called not at top level - if object array or list is present
            prim_tag = '{}:{}'.format(the_xml_ns_key, the_tag) if the_xml_ns_key is not None else the_tag
            if isinstance(val, (Serializable, SerializableArray)):
                val.to_node(doc, the_tag, ns_key=the_xml_ns_key, parent=node,
                            check_validity=check_validity, strict=strict)
            elif isinstance(val, ParametersCollection):
                val.to_node(doc, ns_key=the_xml_ns_key, parent=node, check_validity=check_validity, strict=strict)
            elif isinstance(val, bool):  # this must come before int, where it would evaluate as true
                create_text_node(doc, prim_tag, 'true' if val else 'false', parent=node)
            elif isinstance(val, str):
                create_text_node(doc, prim_tag, val, parent=node)
            elif isinstance(val, int):
                create_text_node(doc, prim_tag, format_function(val), parent=node)
            elif isinstance(val, float):
                create_text_node(doc, prim_tag, format_function(val), parent=node)
            elif isinstance(val, numpy.datetime64):
                out2 = str(val)
                out2 = out2 + 'Z' if out2[-1] != 'Z' else out2
                create_text_node(doc, prim_tag, out2, parent=node)
            elif isinstance(val, complex):
                cnode = create_new_node(doc, prim_tag, parent=node)
                if the_xml_ns_key is None:
                    create_text_node(doc, 'Real', format_function(val.real), parent=cnode)
                    create_text_node(doc, 'Imag', format_function(val.imag), parent=cnode)
                else:
                    create_text_node(doc, '{}:Real'.format(the_xml_ns_key), format_function(val.real), parent=cnode)
                    create_text_node(doc, '{}:Imag'.format(the_xml_ns_key), format_function(val.imag), parent=cnode)
            elif isinstance(val, date):  # should never exist at present
                create_text_node(doc, prim_tag, val.isoformat(), parent=node)
            elif isinstance(val, datetime):  # should never exist at present
                create_text_node(doc, prim_tag, val.isoformat(sep='T'), parent=node)
            else:
                raise ValueError(
                    'An entry for class {} using tag {} is of type {},\n'
                    'and serialization has not been implemented'.format(self.__class__.__name__, the_tag, type(val)))

        if check_validity:
            if not self.is_valid(stack=False):
                msg = "{} is not valid,\n\t" \
                      "and cannot be SAFELY serialized to XML according to the " \
                      "SICD standard.".format(self.__class__.__name__)
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
        # create the main node
        if (ns_key is not None and ns_key != 'default') and not tag.startswith(ns_key + ':'):
            nod = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)
        else:
            nod = create_new_node(doc, tag, parent=parent)

        # serialize the attributes
        for attribute in self._fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue
            fmt_func = self._get_formatter(attribute)
            base_tag_name = self._tag_overide.get(attribute, attribute)
            if attribute in self._set_as_attribute:
                xml_ns_key = self._child_xml_ns_key.get(attribute, ns_key)
                serialize_attribute(nod, base_tag_name, value, fmt_func, xml_ns_key)
            else:
                # should we be using some namespace?
                if attribute in self._child_xml_ns_key:
                    xml_ns_key = self._child_xml_ns_key[attribute]
                else:
                    xml_ns_key = getattr(self, '_xml_ns_key', ns_key)
                    if xml_ns_key == 'default':
                        xml_ns_key = None

                if isinstance(value, (numpy.ndarray, list)):
                    array_tag = self._collections_tags.get(attribute, None)
                    if array_tag is None:
                        raise AttributeError(
                            'The value associated with attribute {} in an instance of class {} is of type {}, '
                            'but nothing is populated in the _collection_tags dictionary.'.format(
                                attribute, self.__class__.__name__, type(value)))
                    child_tag = array_tag.get('child_tag', None)
                    if child_tag is None:
                        raise AttributeError(
                            'The value associated with attribute {} in an instance of class {} is of type {}, '
                            'but `child_tag` is not populated in the _collection_tags dictionary.'.format(
                                attribute, self.__class__.__name__, type(value)))
                    size_attribute = array_tag.get('size_attribute', 'size')
                    if isinstance(value, numpy.ndarray):
                        serialize_array(nod, base_tag_name, child_tag, value, fmt_func, size_attribute, xml_ns_key)
                    else:
                        serialize_list(nod, child_tag, value, fmt_func, xml_ns_key)
                else:
                    serialize_plain(nod, base_tag_name, value, fmt_func, xml_ns_key)
        return nod

    @classmethod
    def from_dict(cls, input_dict):
        """For json deserialization, from dict instance.

        Parameters
        ----------
        input_dict : dict
            Appropriate parameters dict instance for deserialization

        Returns
        -------
            Corresponding class instance
        """

        return cls(**input_dict)

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        """
        For json serialization.

        Parameters
        ----------
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        OrderedDict
            dict representation of class instance appropriate for direct json serialization.
        """

        # noinspection PyUnusedLocal
        def serialize_array(ch_tag, val):
            if not len(val.shape) == 1:
                # again, I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a one-dimensional numpy.ndarray, but it has shape {}'.format(
                        attribute, self.__class__.__name__, val.shape))

            if val.size == 0:
                return []

            if val.dtype.name == 'float64':
                return [float(el) for el in val]
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}. This is expected to be'
                    'a numpy.ndarray of dtype float64, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, val.dtype))

        def serialize_list(ch_tag, val):
            if len(val) == 0:
                return []
            else:
                return [serialize_plain(ch_tag, entry) for entry in val if entry is not None]

        def serialize_plain(field, val):
            # may be called not at top level - if object array or list is present
            if isinstance(val, Serializable):
                return val.to_dict(check_validity=check_validity, strict=strict)
            elif isinstance(val, SerializableArray):
                return val.to_json_list(check_validity=check_validity, strict=strict)
            elif isinstance(val, ParametersCollection):
                return val.to_dict()
            elif isinstance(val, int) or isinstance(val, str) or isinstance(val, float):
                return val
            elif isinstance(val, numpy.datetime64):
                out2 = str(val)
                return out2 + 'Z' if out2[-1] != 'Z' else out2
            elif isinstance(val, complex):
                return {'Real': val.real, 'Imag': val.imag}
            elif isinstance(val, date):  # probably never present
                return val.isoformat()
            elif isinstance(val, datetime):  # probably never present
                return val.isoformat(sep='T')
            else:
                raise ValueError(
                    'a entry for class {} using tag {} is of type {}, and serialization has not '
                    'been implemented'.format(self.__class__.__name__, field, type(val)))

        if check_validity:
            if not self.is_valid(stack=False):
                msg = "{} is not valid,\n\t" \
                      "and cannot be SAFELY serialized to a dictionary valid in " \
                      "the SICD standard.".format(self.__class__.__name__)
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)

        out = OrderedDict()

        for attribute in self._fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue

            if isinstance(value, (numpy.ndarray, list)):
                array_tag = self._collections_tags.get(attribute, None)
                if array_tag is None:
                    raise AttributeError(
                        'The value associated with attribute {} in an instance of class {} is of type {}, '
                        'but nothing is populated in the _collection_tags dictionary.'.format(
                            attribute, self.__class__.__name__, type(value)))
                child_tag = array_tag.get('child_tag', None)
                if child_tag is None:
                    raise AttributeError(
                        'The value associated with attribute {} in an instance of class {} is of type {}, '
                        'but `child_tag` is not populated in the _collection_tags dictionary.'.format(
                            attribute, self.__class__.__name__, type(value)))
                if isinstance(value, numpy.ndarray):
                    out[attribute] = serialize_array(child_tag, value)
                else:
                    out[attribute] = serialize_list(child_tag, value)
            else:
                out[attribute] = serialize_plain(attribute, value)
        return out

    def copy(self):
        """
        Create a deep copy.

        Returns
        -------

        """

        return self.__class__.from_dict(copy.deepcopy(self.to_dict(check_validity=False)))

    def to_xml_bytes(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        """
        Gets a bytes array, which corresponds to the xml string in utf-8 encoding,
        identified as using the namespace given by `urn` (if given).

        Parameters
        ----------
        urn : None|str|dict
            The xml namespace string or dictionary describing the xml namespace.
        tag : None|str
            The root node tag to use. If not given, then the class name will be used.
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.

        Returns
        -------
        bytes
            bytes array from :func:`ElementTree.tostring()` call.
        """
        if tag is None:
            tag = self.__class__.__name__
        the_etree = ElementTree.ElementTree()
        node = self.to_node(the_etree, tag, ns_key=getattr(self, '_xml_ns_key', None),
                            check_validity=check_validity, strict=strict)

        if urn is None:
            pass
        elif isinstance(urn, str):
            node.attrib['xmlns'] = urn
        elif isinstance(urn, dict):
            for key in urn:
                node.attrib[key] = urn[key]
        else:
            raise TypeError('Expected string or dictionary of string for urn, got type {}'.format(type(urn)))
        return ElementTree.tostring(node, encoding='utf-8', method='xml')

    def to_xml_string(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        """
        Gets an xml string with utf-8 encoding, identified as using the namespace
        given by `urn` (if given).

        Parameters
        ----------
        urn : None|str|dict
            The xml namespace or dictionary describing the xml namespace.
        tag : None|str
            The root node tag to use. If not given, then the class name will be used.
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.

        Returns
        -------
        str
            xml string from :func:`ElementTree.tostring()` call.
        """

        return self.to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict).decode('utf-8')


class Arrayable(object):
    """Abstract class specifying basic functionality for assigning from/to an array"""

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type object.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple

        Returns
        -------
        Arrayable
        """

        raise NotImplementedError

    def get_array(self, dtype=numpy.float64):
        """Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
        """
        raise NotImplementedError

    def __getitem__(self, item):
        return self.get_array()[item]


class SerializableArray(object):
    __slots__ = (
        '_child_tag', '_child_type', '_array', '_name', '_minimum_length',
        '_maximum_length', '_xml_ns', '_xml_ns_key')
    _default_minimum_length = 0
    _default_maximum_length = 2**32
    _set_size = True
    _size_var_name = 'size'
    _set_index = True
    _index_var_name = 'index'

    def __init__(self, coords=None, name=None, child_tag=None, child_type=None,
                 minimum_length=None, maximum_length=None, _xml_ns=None, _xml_ns_key=None):
        self._xml_ns = _xml_ns
        self._xml_ns_key = _xml_ns_key
        self._array = None
        if name is None:
            raise ValueError('The name parameter is required.')
        if not isinstance(name, str):
            raise TypeError(
                'The name parameter is required to be an instance of str, got {}'.format(type(name)))
        self._name = name

        if child_tag is None:
            raise ValueError('The child_tag parameter is required.')
        if not isinstance(child_tag, str):
            raise TypeError(
                'The child_tag parameter is required to be an instance of str, got {}'.format(type(child_tag)))
        self._child_tag = child_tag

        if child_type is None:
            raise ValueError('The child_type parameter is required.')
        if not issubclass(child_type, Serializable):
            raise TypeError('The child_type is required to be a subclass of Serializable.')
        self._child_type = child_type

        if minimum_length is None:
            self._minimum_length = self._default_minimum_length
        else:
            self._minimum_length = max(int(minimum_length), 0)

        if maximum_length is None:
            self._maximum_length = max(self._default_maximum_length, self._minimum_length)
        else:
            self._maximum_length = max(int(maximum_length), self._minimum_length)

        self.set_array(coords)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self._array[index]

    def __setitem__(self, index, value):
        if value is None:
            raise TypeError('Elements of {} must be of type {}, not None'.format(self._name, self._child_type))
        self._array[index] = parse_serializable(value, self._name, self, self._child_type)

    def log_validity_error(self, msg):
        """
        Log a validation error message.

        Parameters
        ----------
        msg : str
        """

        valid_logger.error('{}:{} {}'.format(self.__class__.__name__, self._name, msg))

    def log_validity_warning(self, msg):
        """
        Log a validation warning message.

        Parameters
        ----------
        msg : str
        """

        valid_logger.warning('{}:{} {}'.format(self.__class__.__name__, self._name, msg))

    def log_validity_info(self, msg):
        """
        Log a validation info message.

        Parameters
        ----------
        msg : str
        """

        valid_logger.info('{}:{} {}'.format(self.__class__.__name__, self._name, msg))

    def is_valid(self, recursive=False, stack=False):
        """Returns the validity of this object according to the schema. This is done by inspecting that the
        array is populated.

        Parameters
        ----------
        recursive : bool
            True if we recursively check that children are also valid. This may result in verbose (i.e. noisy) logging.
        stack : bool
            Should we print error messages recursively, for a stack type situation?
        Returns
        -------
        bool
            condition for validity of this element
        """
        if self._array is None:
            self.log_validity_error("Unpopulated array")
            return False
        if not recursive:
            return True
        valid_children = True
        for i, entry in enumerate(self._array):
            good = entry.is_valid(recursive=True, stack=stack)
            if not good and stack:
                self.log_validity_error("Issue discovered with entry {}".format(i))
            valid_children &= good
        return valid_children

    @property
    def size(self):  # type: () -> int
        """
        int: the size of the array.
        """

        if self._array is None:
            return 0
        else:
            return self._array.size

    def get_array(self, dtype='object', **kwargs):
        """Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return.
        kwargs : keyword arguments for calls of the form child.get_array(**kwargs)

        Returns
        -------
        numpy.ndarray
            * If `dtype` == 'object'`, then the literal array of
              child objects is returned. *Note: Beware of mutating the elements.*
            * If `dtype` has any other value, then the return value will be tried
              as `numpy.array([child.get_array(dtype=dtype, **kwargs) for child in array]`.
            * If there is any error, then `None` is returned.
        """

        if dtype in ['object', numpy.dtype('object')]:
            return self._array
        else:
            # noinspection PyBroadException
            try:
                return numpy.array(
                    [child.get_array(dtype=dtype, **kwargs) for child in self._array], dtype=dtype)
            except Exception:
                return None

    def set_array(self, coords):
        """
        Sets the underlying array.

        Parameters
        ----------
        coords : numpy.ndarray|list|tuple

        Returns
        -------
        None
        """

        if coords is None:
            self._array = None
            return
        array = parse_serializable_array(
            coords, 'coords', self, self._child_type, self._child_tag)
        if not (self._minimum_length <= array.size <= self._maximum_length):
            raise ValueError(
                'Field {} is required to be an array with {} <= length <= {}, and input of length {} '
                'was received'.format(self._name, self._minimum_length, self._maximum_length, array.size))

        self._array = array
        self._check_indices()

    def _check_indices(self):
        if not self._set_index:
            return
        for i, entry in enumerate(self._array):
            try:
                setattr(entry, self._index_var_name, i+1)
            except (AttributeError, ValueError, TypeError):
                continue

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT):
        if self.size == 0:
            return None  # nothing to be done
        if ns_key is None:
            anode = create_new_node(doc, tag, parent=parent)
        else:
            anode = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)
        if self._set_size:
            anode.attrib[self._size_var_name] = str(self.size)
        for i, entry in enumerate(self._array):
            entry.to_node(doc, self._child_tag, ns_key=ns_key, parent=anode,
                          check_validity=check_validity, strict=strict)
        return anode

    @classmethod
    def from_node(cls, node, name, child_tag, child_type, **kwargs):
        return cls(coords=node, name=name, child_tag=child_tag, child_type=child_type, **kwargs)

    def to_json_list(self, check_validity=False, strict=DEFAULT_STRICT):
        """
        For json serialization.
        Parameters
        ----------
        check_validity : bool
            passed through to child_type.to_dict() method.
        strict : bool
            passed through to child_type.to_dict() method.

        Returns
        -------
        List[dict]
        """

        if self.size == 0:
            return []
        return [entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._array]


class ParametersCollection(object):
    __slots__ = ('_name', '_child_tag', '_dict', '_xml_ns', '_xml_ns_key')

    def __init__(self, collection=None, name=None, child_tag='Parameters', _xml_ns=None, _xml_ns_key=None):
        self._dict = None
        self._xml_ns = _xml_ns
        self._xml_ns_key = _xml_ns_key
        if name is None:
            raise ValueError('The name parameter is required.')
        if not isinstance(name, str):
            raise TypeError(
                'The name parameter is required to be an instance of str, got {}'.format(type(name)))
        self._name = name

        if child_tag is None:
            raise ValueError('The child_tag parameter is required.')
        if not isinstance(child_tag, str):
            raise TypeError(
                'The child_tag parameter is required to be an instance of str, got {}'.format(type(child_tag)))
        self._child_tag = child_tag

        self.set_collection(collection)

    def __delitem__(self, key):
        if self._dict is not None:
            del self._dict[key]

    def __getitem__(self, key):
        if self._dict is not None:
            return self._dict[key]
        raise KeyError('Dictionary does not contain key {}'.format(key))

    def __setitem__(self, name, value):
        if not isinstance(name, str):
            raise ValueError('Parameter name must be of type str, got {}'.format(type(name)))
        if not isinstance(value, str):
            raise ValueError('Parameter name must be of type str, got {}'.format(type(value)))

        if self._dict is None:
            self._dict = OrderedDict()
        self._dict[name] = value

    def get(self, key, default=None):
        if self._dict is not None:
            return self._dict.get(key, default)
        return default

    def set_collection(self, value):
        if value is None:
            self._dict = None
        else:
            self._dict = parse_parameters_collection(value, self._name, self)

    def get_collection(self):
        return self._dict

    # noinspection PyUnusedLocal
    def to_node(self, doc, ns_key=None, parent=None, check_validity=False, strict=False):
        if self._dict is None:
            return None  # nothing to be done
        for name in self._dict:
            value = self._dict[name]
            if not isinstance(value, str):
                value = str(value)
            if ns_key is None:
                node = create_text_node(doc, self._child_tag, value, parent=parent)
            else:
                node = create_text_node(doc, '{}:{}'.format(ns_key, self._child_tag), value, parent=parent)
            node.attrib['name'] = name

    # noinspection PyUnusedLocal
    def to_dict(self, check_validity=False, strict=False):
        return copy.deepcopy(self._dict)
