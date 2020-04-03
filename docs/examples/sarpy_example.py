"""
Just a quick demo to show the basics of how to use SarPy
"""

import matplotlib.pyplot as plt
import sarpy.io.complex as cf
import sarpy.visualization.remap as remap

# Open file
fname = 'C:/Temp/my_sicd.nitf'
ro = cf.open(fname)

# Access SICD metadata (even if file read is not in SICD format)
print(ro.sicd_meta)  # Displays metadata from file in SICD format in human-readable form
print(ro.sicd_meta.CollectionInfo.CollectorName)  # extracting fields from metadata

###############
# Read complex pixel data from file
# cdata = ro[:]  # Reads all complex data from file

# Read every 10th pixel:
cdata = ro[::10, ::10]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
axs[0].imshow(remap.density(cdata), cmap='gray')  # Display subsampled image

# Reads every other row and column from the first thousand rows and columns:
cdata = ro[:1000:2, :1000:2]
axs[1].imshow(remap.density(cdata), cmap='gray')  # Display subsampled image

# Reads every row and column from the first thousand rows and columns:
cdata = ro[0:1000, 0:1000]
axs[2].imshow(remap.density(cdata), cmap='gray')  # Display subsampled image
plt.show()

##############
# Convert a complex dataset (in any format handled by SarPy) to SICD
cf.convert(fname, 'C:/Temp/new_sicd.nitf')
# Convert a complex data to SIO
output_directory = 'C:/Temp'
cf.convert(fname, output_directory, output_format='sio')
