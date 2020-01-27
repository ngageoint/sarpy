import os
import time

from sarpy.io.nitf_headers import NITFDetails

test_root = 'C:/Users/jr80407/Desktop/sarpy_testing/sicd'

# TODO: refactor this into unit tests

# test_file = os.path.join(test_root, 'sicd_example_RMA_RGZERO_RE32F_IM32F_cropped_multiple_image_segments_v1.2.nitf')
test_file = os.path.join(test_root, 'sicd_example_RMA_RGZERO_RE16I_IM16I.nitf')

start = time.time()
details = NITFDetails(test_file)
# how long does it take?
print('unpacked nitf details in {}'.format(time.time() - start))

# how does it look?
# print(details._nitf_header)

# is the output as long as it should be?
header_string = details._nitf_header.to_string()
equality = (len(header_string) == details._nitf_header.HL)
if not equality:
    print('len(produced header) = {}, nitf_header.HL = {}'.format(len(header_string), details._nitf_header.HL))
assert equality

# is the output what it should be?
with open(test_file, 'rb') as fi:
    file_header = fi.read(details._nitf_header.HL)

equality = (file_header == header_string)
if not equality:
    chunk_size = 80
    start_chunk = 0
    while start_chunk < len(header_string):
        end_chunk = min(start_chunk+chunk_size, len(header_string))
        print('real[{}:{}] = {}'.format(
            start_chunk, end_chunk, file_header[start_chunk:end_chunk]))
        print('prod[{}:{}] = {}'.format(
            start_chunk, end_chunk, header_string[start_chunk:end_chunk]))
        start_chunk = end_chunk

assert equality
