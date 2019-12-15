import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest


def generic_construction_test(instance, the_type, the_dict):
    the_item = the_type.from_dict(the_dict)

    with instance.subTest(msg='Comparing json deserialization with original'):
        new_dict = the_item.to_dict()
        instance.assertEqual(the_dict, new_dict)

    with instance.subTest(msg='Test xml serialization issues'):
        # let's serialize to xml
        etree = ElementTree.ElementTree()
        xml = ElementTree.tostring(the_item.to_node(etree, 'The_Type')).decode('utf-8')
        # let's deserialize from xml
        node = ElementTree.fromstring(xml)
        item2 = the_type.from_node(node)
        instance.assertEqual(the_item.to_dict(), item2.to_dict())

    with instance.subTest(msg='Test validity'):
        instance.assertTrue(the_item.is_valid())
    return the_item
