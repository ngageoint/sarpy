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

    def testBadBoundsFail(self):
        with self.assertRaisesRegex(ValueError, 
                                    re.escape("Slice argument 2 does not fit " + \
                                              "with bound 1")):
            self.assertEqual(validate_slice_int(2, 1, False), 2)

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

class Test_verify_subscript(TestCase):
    def setUp(self):
        self.item = [1, 60, 3]
        self.max_element = 111

    def testNoParamsFail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "verify_subscript() missing 2 required " + \
                                            "positional arguments: 'subscript' " + \
                                                "and 'corresponding_shape'")):
            verify_subscript()

    def testSubscriptNoneSuccess(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        return_value   = verify_subscript(None, self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptElipsisSuccess(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        return_value   = verify_subscript(..., self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptIntSuccess(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        return_value   = verify_subscript(0, self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptSliceSuccess(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        return_value   = verify_subscript(slice(0,1), self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptSequenceListSuccess(self):
        expected_value = (slice(0, 1, 1), slice(1, 2, 1), slice(0, 3, 1))
        return_value   = verify_subscript([0,1], self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptSequenceListWithTwoElipsisFail(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        with self.assertRaisesRegex(KeyError, 
                                    re.escape(
                                        "slice definition cannot contain more " + \
                                             "than one ellipsis")):
             return_value   = verify_subscript([0,..., ...], self.item)
        
    def testSubscriptSequenceListWithOneElipsisSubscriptTooBigFail(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "More subscript entries (4) than shape " + \
                                             "dimensions (3)")):
             return_value   = verify_subscript([0,..., 1, 5], self.item)

    def testSubscriptSequenceListWithLastElipsisSuccess(self):
        expected_value = (slice(0, 1, 1), slice(1, 2, 1), slice(0, 3, 1))
        return_value   = verify_subscript([0,1,...], self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptSequenceListWithFirstElipsisSuccess(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(1, 2, 1))
        return_value   = verify_subscript([...,1], self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptSequenceListWithMiddleElipsisSuccess(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(2, 3, 1))
        return_value   = verify_subscript([0,...,2], self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptSequenceListNoElipsisSubscriptTooBigFail(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        with self.assertRaisesRegex(ValueError, 
                                    re.escape(
                                        "More subscript entries (4) than shape " + \
                                             "dimensions (3)")):
             return_value   = verify_subscript([0,4, 1, 5], self.item)

    def testSubscriptSequenceListNoElipsisSuccess(self):
        expected_value = (slice(0, 1, 1), slice(1, 2, 1), slice(0, 3, 1))
        return_value   = verify_subscript([0,1], self.item)
        self.assertEqual(expected_value, return_value)

    def testSubscriptFloatFail(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        with self.assertRaisesRegex(ValueError, 
                                    re.escape("Got unhandled subscript 4.5")):
             return_value   = verify_subscript(4.5, self.item)

    def testSubscriptStringFail(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        with self.assertRaisesRegex(TypeError, 
                                    re.escape("'<=' not supported between " + \
                                              "instances of 'int' and 'str'")):
             return_value   = verify_subscript("bob", self.item)

    def testSubscriptRangeFail(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        with self.assertRaisesRegex(ValueError, 
                                    re.escape("More subscript entries (5) than " + \
                                              "shape dimensions (3).")):
             return_value   = verify_subscript(range(5), self.item)

    def testSubscriptSequenceRangeSuccess(self):
        expected_value = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        return_value   = verify_subscript(range(1), self.item)
        self.assertEqual(expected_value, return_value)
        
class Test_get_slice_result_size(TestCase):
    def setUp(self):
        self.item = [1, 60, 3]
        self.max_element = 111

    def testNoParamsFail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "get_slice_result_size() missing 1 " + \
                                            "required positional argument: " + \
                                                "'slice_in'")):
            get_slice_result_size()

    def testFullSliceSuccess(self):
        return_value   = get_slice_result_size(slice(0,5,1))
        self.assertEqual(5, return_value)

    def testSliceNegativeStepNoneStopSuccess(self):
        return_value   = get_slice_result_size(slice(0,None,-1))
        self.assertEqual(1, return_value)

    def testSliceNegativeStepSuccess(self):
        return_value   = get_slice_result_size(slice(0,5,-1))
        self.assertEqual(-5, return_value)

class Test_get_subscript_result_size(TestCase):
    def setUp(self):
        self.item = [1, 60, 3]
        self.max_element = 111

    def testNoParamsFail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "get_subscript_result_size() missing " + \
                                            "2 required positional arguments: " + \
                                                "'subscript' and " + \
                                                    "'corresponding_shape'")):
            get_subscript_result_size()

    def testSubscriptNoneSuccess(self):
        expected_subscript = (slice(0, 1, 1), slice(0, 60, 1), slice(0, 3, 1))
        expected_shape = (1, 60, 3)
        return_subscript, return_shape   = get_subscript_result_size(None, self.item)
        self.assertEqual(expected_subscript, return_subscript)
        self.assertEqual(expected_shape, return_shape)
