"""
Just a quick demo to show the basics of how to use SarPy
"""

import matplotlib.pyplot as plt
import sarpy.io.complex as cf
import sarpy.visualization.remap as remap
from sarpy import sarpy_support_dir
import os

# Open file
fname = os.path.expanduser(os.path.join('~/sarpy_data/nitf', 'sicd_example_1_PFA_RE32F_IM32F_HH.nitf'))
ro = cf.open(fname)

# Access SICD metadata (even if file read is not in SICD format)
print(ro.sicdmeta)  # Displays metadata from file in SICD format in human-readable form
ro.sicdmeta  # Displays XML representation of SICD metadata
print(ro.sicdmeta.CollectionInfo.CollectorName)  # Notation for extracting fields from metadata

# Read complex pixel data from file
# cdata = reader_object.read_chip[...] # Reads all complex data from file
# Read every 10th pixel:
cdata = ro.read_chip[::8, ::8]
plt.figure()
plt.imshow(remap.density(cdata), cmap='gray')  # Display subsampled image
plt.show()

# Reads every other row and column from the first thousand rows and columns:
cdata = ro.read_chip[:1000:2, :1000:2]
plt.figure()
plt.imshow(remap.density(cdata), cmap='gray')  # Display subsampled image
plt.show()

# Reads every row and column from the first thousand rows and columns:
cdata = ro.read_chip[0:1000, 0:1000]
plt.figure()
plt.imshow(remap.density(cdata), cmap='gray')  # Display subsampled image
plt.show()

# Convert a complex dataset (in any format handled by SarPy) to SICD
cf.convert(fname, 'C:/Temp/new_sicd.nitf')
# Convert a complex data to SIO
cf.convert(fname, 'C:/Temp/new_sio.sio', output_format='sio')
