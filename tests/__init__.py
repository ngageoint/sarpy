import sys
import logging
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements.base import Serializable

logging.basicConfig(level=logging.INFO)


def generic_construction_test(instance, the_type, the_dict, tag='The_Type', print_xml=False):
    if not issubclass(the_type, Serializable):
        raise TypeError('Class {} must be a subclass of Serializable'.format(the_type))
    the_item = the_type.from_dict(the_dict)

    with instance.subTest(msg='Comparing json deserialization with original'):
        new_dict = the_item.to_dict()
        instance.assertEqual(the_dict, new_dict)

    with instance.subTest(msg='Test xml serialization issues'):
        # let's serialize to xml
        xml = the_item.to_xml_string(tag=tag)
        if print_xml:
            print(xml)
        # let's deserialize from xml
        node = ElementTree.fromstring(xml)
        item2 = the_type.from_node(node)
        instance.assertEqual(the_item.to_dict(), item2.to_dict())

    with instance.subTest(msg='Test validity'):
        instance.assertTrue(the_item.is_valid())
    return the_item
