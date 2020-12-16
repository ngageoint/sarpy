from sarpy.utils.review_class import check_classification

from tests import unittest


class TestClassString(unittest.TestCase):
    def test_class_str(self):
        results_dict = check_classification('sarpy')
        key = '__NO_CLASSIFICATION__'
        if key in results_dict:
           raise ValueError(
               'The following modules have no classification string defined {}'.format(results_dict[key]))
