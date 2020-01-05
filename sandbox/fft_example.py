import matplotlib.pyplot as plt
import sarpy.io.complex as cf
import sarpy.visualization.remap as remap
import os
from scipy.fftpack import fft2, ifft2, fftshift
import numpy as np

output_dir = os.path.expanduser('~/sarpy_data/output')

# Open file
fname = os.path.expanduser(os.path.join('~/sarpy_data/nitf', 'sicd_example_1_PFA_RE32F_IM32F_HH.nitf'))
ro = cf.open(fname)

print("compute the fft to display in range / polar azimuth")
cdata = ro.read_chip()

inverseFlag = False
if ro.sicdmeta.Grid.Col.Sgn > 0 and ro.sicdmeta.Grid.Row.Sgn > 0:
    # use fft2 to go from image to spatial freq
    ft_cdata = fft2(cdata)
else:
    # flip using ifft2
    ft_cdata = ifft2(cdata)
    inverseFlag = True

ft_cdata = fftshift(ft_cdata)

print("display fft'd data")
plt.figure()
plt.imshow(remap.density(ft_cdata), cmap='gray')
plt.show()

print("clip fft data and display reconstruction")
# TODO replace with padded windowing function and multiply
filtered_cdata = np.zeros(ft_cdata.shape, ft_cdata.dtype)
filtered_cdata[1500:2000, 3000:4000] = ft_cdata[1500:2000, 3000:4000]
filtered_cdata = fftshift(filtered_cdata)

if inverseFlag:
    cdata_clip = fft2(filtered_cdata)
else:
    cdata_clip = ifft2(filtered_cdata)

plt.figure()
plt.imshow(remap.density(cdata_clip), cmap='gray')
plt.show()

print("show original for comparison")
plt.figure()
plt.imshow(remap.density(cdata), cmap='gray')
plt.show()

print("finished sarpy example")