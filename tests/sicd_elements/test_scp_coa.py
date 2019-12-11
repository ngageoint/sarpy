import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import SCPCOA


scp_coa_dict = {
    'SCPTime': 10,
    'ARPPos': {'X': 0, 'Y': 0, 'Z': 10},
    'ARPVel': {'X': 1, 'Y': 1, 'Z': 1},
    'ARPAcc': {'X': 0, 'Y': 0, 'Z': 0},
    'SideOfTrack': 'L',
    'SlantRange': 10,
    'GroundRange': 10,
    'DopplerConeAng': 10,
    'GrazeAng': 45,
    'IncidenceAng': 30,
    'TwistAng': 30,
    'SlopeAng': 30,
    'AzimAng': 45,
    'LayoverAng': 10,
}


class TestSCPCOA(unittest.TestCase):
    def test_construction(self):
        the_type = SCPCOA.SCPCOAType
        the_dict = scp_coa_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())
