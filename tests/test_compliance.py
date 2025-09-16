__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"

import pytest, os
from unittest import TestCase

from sarpy.compliance import SarpyError, bytes_to_string

# Test the SarpyError class.
# The SarpyError class is a pass through.
# The only way to test it is to try an operation that fails and check if it is 
#   raised.
def test_SarpyError() :
    file_name = "bad/filename"
    try:
        os.path.exists(file_name)
        assert False
    except:
        pytest.raises(SarpyError)


class Test_bytes_to_string(TestCase):
    def setUp(self):
        self.text_string = "Hello, world!"
        self.byte_data   = self.text_string.encode('utf-8')

    def testStringInputSuccess(self):
        self.assertEqual(self.text_string, bytes_to_string(self.text_string))

    def testBadInputFail(self):
        with self.assertRaisesRegex(TypeError, 'Input is required to be bytes. Got type*'):
            bytes_to_string(11)

    def testByteInputSuccess(self):
        self.assertEqual(self.text_string, bytes_to_string(self.byte_data))