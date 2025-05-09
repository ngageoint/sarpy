import unittest

from sarpy.io.complex.converter import open_complex
from sarpy.io.complex.sicd import SICDReader

try:
    import smart_open
except (ImportError, SyntaxError):
    smart_open = None

file_object = None
if smart_open is not None:
    try:
        file_object = smart_open.open(
            'https://six-library.s3.amazonaws.com/sicd_example_RMA_RGZERO_RE32F_IM32F_cropped_multiple_image_segments.nitf',
            mode='rb',  # must be opened in binary mode
            buffering=4*1024*1024)  # it has been observed that setting a manual buffer size may help
    except Exception:
        pass


class TestRemoteSICD(unittest.TestCase):
    @unittest.skipIf(file_object is None, 'No remote file reader defined')
    def test_remote_reader(self):
        reader = None
        with self.subTest(msg='establish remote sicd reader'):
            reader = open_complex(file_object)
            file_object.close()
            self.assertTrue(reader is not None, msg='Returned None, so opening failed.')

        if reader is None:
            return  # remaining tests make no sense

        with self.subTest(msg='Reader type should be SICD reader'):
            self.assertTrue(
                isinstance(reader, SICDReader),
                msg='Returned reader should be SICDReader')

        if not isinstance(reader, SICDReader):
            return  # remaining tests might be misleading

        with self.subTest(
                msg='Fetch data_sizes and sicds'):
            data_sizes = reader.get_data_size_as_tuple()
            # noinspection PyUnresolvedReferences
            sicds = reader.get_sicds_as_tuple()

        for i, (data_size, sicd) in enumerate(zip(data_sizes, sicds)):
            with self.subTest(
                    msg='Verify image size for sicd index {} in reader'.format(i)):
                self.assertEqual(data_size[0], sicd.ImageData.NumRows, msg='data_size[0] and NumRows do not agree')
                self.assertEqual(data_size[1], sicd.ImageData.NumCols, msg='data_size[1] and NumCols do not agree')

            with self.subTest(msg='Basic fetch test for sicd index {} in reader'.format(i)):
                self.assertEqual(reader[:2, :2, i].shape, (2, 2), msg='upper left fetch')
                self.assertEqual(reader[-2:, :2, i].shape, (2, 2), msg='lower left fetch')
                self.assertEqual(reader[-2:, -2:, i].shape, (2, 2), msg='lower right fetch')
                self.assertEqual(reader[:2, -2:, i].shape, (2, 2), msg='upper right fetch')

            with self.subTest(
                    msg='Verify fetching complete row(s) have correct size for sicd index {}'.format(i)):
                test_data = reader[:, :2, i]
                self.assertEqual(test_data.shape, (data_size[0], 2), msg='Complete row fetch size mismatch')

            with self.subTest(
                    msg='Verify fetching complete columns(s) have correct size for sicd index {}'.format(i)):
                test_data = reader[:2, :, i]
                self.assertEqual(test_data.shape, (2, data_size[1]), msg='Complete row fetch size mismatch')
        reader.close()
