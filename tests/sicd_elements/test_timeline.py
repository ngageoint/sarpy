from .. import generic_construction_test, unittest

from sarpy.sicd_elements import Timeline


ipp_set_dict = {
    'TStart': 1,
    'TEnd': 10,
    'IPPStart': 1,
    'IPPEnd': 3,
    'IPPPoly': {'Coefs': [1, 2]},
    'index': 0,
}
timeline_dict = {
    'CollectStart': '2019-12-11T13:58:27.000000',
    'CollectDuration': 54,
    'IPP': [ipp_set_dict, ],
}


class TestIPPSet(unittest.TestCase):
    def test_construction(self):
        the_item = generic_construction_test(self, Timeline.IPPSetType, ipp_set_dict)


class TestTimeline(unittest.TestCase):
    def test_construction(self):
        the_item = generic_construction_test(self, Timeline.TimelineType, timeline_dict)
