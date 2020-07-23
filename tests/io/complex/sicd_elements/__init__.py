
from xml.etree import ElementTree
import json

from sarpy.io.complex.sicd_elements.base import Serializable
from sarpy.io.general.utils import parse_xml_from_string

import sys
if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest


def generic_construction_test(instance, the_type, the_dict, tag='The_Type', print_xml=False, print_json=False):
    if not issubclass(the_type, Serializable):
        raise TypeError('Class {} must be a subclass of Serializable'.format(the_type))
    the_item = the_type.from_dict(the_dict)

    with instance.subTest(msg='Comparing json deserialization with original'):
        new_dict = the_item.to_dict()
        if print_json:
            print(json.dumps(the_dict, indent=1))
            print(json.dumps(new_dict, indent=1))
        instance.assertEqual(the_dict, new_dict)

    with instance.subTest(msg='Test xml serialization issues'):
        # let's serialize to xml
        xml = the_item.to_xml_string(tag=tag)
        if print_xml:
            print(xml)
        # let's deserialize from xml
        node, xml_ns = parse_xml_from_string(xml)
        item2 = the_type.from_node(node, xml_ns)
        instance.assertEqual(the_item.to_dict(), item2.to_dict())

    with instance.subTest(msg='Test validity'):
        instance.assertTrue(the_item.is_valid())
    return the_item
