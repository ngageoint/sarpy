import mmap
import os
import pathlib

# From https://nsgreg.nga.mil/doc/view?i=5516
nitf_with_tres = '07APR2005_Hyperion_331406N0442000E_SWIR172_1p2B_L1R-BIP.ntf'
cetags = [b'MATESA', b'BANDSB']

for cetag in cetags:
    with open(nitf_with_tres, 'r+b') as nitffile, mmap.mmap(nitffile.fileno(), 0) as mm:
        mm.seek(mm.find(cetag, 0) + len(cetag), os.SEEK_SET)
        cel = mm.read(5)
        data = mm.read(int(cel))
    output_file = 'example_{}_tre.bin'.format(cetag.decode('ascii').lower())
    (pathlib.Path(__file__).parent / output_file).write_bytes(cetag + cel + data)

