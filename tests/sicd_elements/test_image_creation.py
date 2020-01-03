from .. import generic_construction_test, unittest

from sarpy.sicd_elements import ImageCreation


image_creation_dict = {
    'Application': 'SARPY 0.1',
    'DateTime': '2019-12-10T18:52:17.000000Z',  # to match the serialized precision
    'Site': 'Some Place',
    'Profile': 'No Idea'
}


class TestImageCreation(unittest.TestCase):
    def test_construction(self):
        the_type = ImageCreation.ImageCreationType
        the_dict = image_creation_dict
        item1 = generic_construction_test(self, the_type, the_dict)
