"""
This module contains the base objects for use in the SICD elements, and the base serializable functionality.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from sarpy.io.xml.base import SerializableArray, create_new_node
from sarpy.io.xml.descriptors import BasicDescriptor


logger = logging.getLogger(__name__)
DEFAULT_STRICT = False
FLOAT_FORMAT = '0.17G'


class SerializableCPArrayDescriptor(BasicDescriptor):
    """A descriptor for properties of a list or array of specified extension of Serializable"""
    minimum_length = 4
    maximum_length = 4

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT, docstring=None):
        self.child_type = child_type
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        if not self.array:
            raise ValueError(
                'Attribute {} is populated in the `_collection_tags` dictionary without `array`=True. '
                'This is inconsistent with using SerializableCPArrayDescriptor.'.format(name))

        self.child_tag = tags['child_tag']
        self._typ_string = 'numpy.ndarray[{}]:'.format(str(child_type).strip().split('.')[-1][:-2])

        super(SerializableCPArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(SerializableCPArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, SerializableCPArray):
            self.data[instance] = value
        else:
            xml_ns = getattr(instance, '_xml_ns', None)
            # noinspection PyProtectedMember
            if hasattr(instance, '_child_xml_ns_key') and self.name in instance._child_xml_ns_key:
                # noinspection PyProtectedMember
                xml_ns_key = instance._child_xml_ns_key[self.name]
            else:
                xml_ns_key = getattr(instance, '_xml_ns_key', None)

            the_inst = self.data.get(instance, None)
            if the_inst is None:
                self.data[instance] = SerializableCPArray(
                    coords=value, name=self.name, child_tag=self.child_tag,
                    child_type=self.child_type, _xml_ns=xml_ns, _xml_ns_key=xml_ns_key)
            else:
                the_inst.set_array(value)


class SerializableCPArray(SerializableArray):
    __slots__ = (
        '_child_tag', '_child_type', '_array', '_name', '_minimum_length',
        '_maximum_length', '_index_as_string', '_xml_ns', '_xml_ns_key')

    def __init__(self, coords=None, name=None, child_tag=None, child_type=None, _xml_ns=None, _xml_ns_key=None):
        if hasattr(child_type, '_CORNER_VALUES'):
            self._index_as_string = True
        else:
            self._index_as_string = False
        super(SerializableCPArray, self).__init__(
            coords=coords, name=name, child_tag=child_tag, child_type=child_type,
            _xml_ns=_xml_ns, _xml_ns_key=_xml_ns_key)
        self._minimum_length = 4
        self._maximum_length = 4

    @property
    def FRFC(self):
        if self._array is None:
            return None
        return self._array[0].get_array()

    @property
    def FRLC(self):
        if self._array is None:
            return None
        return self._array[1].get_array()

    @property
    def LRLC(self):
        if self._array is None:
            return None
        return self._array[2].get_array()

    @property
    def LRFC(self):
        if self._array is None:
            return None
        return self._array[3].get_array()

    def _check_indices(self):
        if not self._index_as_string:
            self._array[0].index = 1
            self._array[1].index = 2
            self._array[2].index = 3
            self._array[3].index = 4
        else:
            self._array[0].index = '1:FRFC'
            self._array[1].index = '2:FRLC'
            self._array[2].index = '3:LRLC'
            self._array[3].index = '4:LRFC'

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT):
        if self.size == 0:
            return None  # nothing to be done
        if ns_key is None:
            anode = create_new_node(doc, tag, parent=parent)
        else:
            anode = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)
        for i, entry in enumerate(self._array):
            entry.to_node(doc, self._child_tag, ns_key=ns_key, parent=anode,
                          check_validity=check_validity, strict=strict)
        return anode
