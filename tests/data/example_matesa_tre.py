import mmap
import os
import pathlib

# From https://nsgreg.nga.mil/doc/view?i=5516
nitf_with_matesa = '07APR2005_Hyperion_331406N0442000E_SWIR172_1p2B_L1R-BIP/07APR2005_Hyperion_331406N0442000E_SWIR172_1p2B_L1R-BIP.ntf'
cetag = b'MATESA'

with open(nitf_with_matesa, 'r+b') as nitffile, mmap.mmap(nitffile.fileno(), 0) as mm:
    mm.seek(mm.find(cetag, 0) + len(cetag), os.SEEK_SET)
    cel = mm.read(5)
    data = mm.read(int(cel))

(pathlib.Path(__file__).parent / 'example_matesa_tre.bin').write_bytes(cetag + cel + data)
