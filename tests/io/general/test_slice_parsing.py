__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"

import pytest, os, re
from unittest import TestCase

from sarpy.io.general.slice_parsing import validate_slice_int, verify_slice, \
    verify_subscript, get_slice_result_size, get_subscript_result_size


class Test_validate_slice_int(TestCase):
    def setUp(self):
        self.the_int = 5
        self.bound   = 20
        self.include = True

    def testNoParamsFail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "validate_slice_int() missing 2 " + \
                                            "required positional arguments: " + \
                                                "'the_int' and 'bound'")):
            validate_slice_int()

    def testBoundAsZeroFail(self):
        with self.assertRaisesRegex(TypeError, 'bound must be a positive integer.'):
            validate_slice_int(self.the_int, 0, self.include)

    def testBoundAsFloatFail(self):
        with self.assertRaisesRegex(TypeError, 'bound must be a positive integer.'):
            validate_slice_int(self.the_int, 11.5, self.include)

    def testTheIntOutOfBoundFail(self):
        with self.assertRaisesRegex(ValueError, 'Slice argument 5 does not fit with bound 2'):
            validate_slice_int(self.the_int, 2, self.include)
    
    def testTheIntLessThanZeroSuccess(self):
        self.assertEqual(validate_slice_int(-2, 4, self.include), 2)

    def testValidTheIntSuccess(self):
        self.assertEqual(validate_slice_int(self.the_int, self.bound, self.include), self.the_int)

class Test_verify_slice(TestCase):
    def setUp(self):
        self.item = [1, 60, 3]
        self.max_element = 111

    def testNoParamsFail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "verify_slice() missing 2 required " + \
                                            "positional arguments: " + \
                                                "'item' and 'max_element'")):
            verify_slice()

    def testMaxElementFloatFail(self):
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "slice verification requires a " + \
                                            "positive integer limit")):
            verify_slice(None, 3.5)

    def testMaxElementLessThan1Fail(self):
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "slice verification requires a " + \
                                            "positive integer limit")):
            verify_slice(None, 0)
    
    def testItemNonePass(self):
        self.assertEqual(verify_slice(None, self.max_element), \
                         slice(0, self.max_element, 1))
        
    def testItemPositiveIntPass(self):
        self.assertEqual(verify_slice(3, self.max_element), \
                         slice(3, 4, 1))
        
    def testItemNegativeIntPass(self):
        self.assertEqual(verify_slice(-3, self.max_element), \
                         slice(108, 109, 1))
        
    def testItemOutOfBoundsPositiveFail(self):
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "Got out of bounds argument (888) in " + \
                                            "slice limited by `111`")):
            verify_slice(888, self.max_element)

    def testItemOutOfBoundsNegativeFail(self):
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "Got out of bounds argument (-888) in " + \
                                            "slice limited by `111`")):
            verify_slice(-888, self.max_element)

    def testSlicePass(self):
        verify_slice(self.item, self.max_element)

    def testSliceItemOutOfBoundsPositiveFail(self):
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "Got out of bounds argument (6) in " + \
                                            "slice limited by `3`")):
            verify_slice([6, 3, 1], 3)

    def testSliceItemOutOfBoundsNegativeFail(self):
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "Got out of bounds argument (-6) in " + \
                                            "slice limited by `3`")):
            verify_slice([-6, 3, 1], 3)