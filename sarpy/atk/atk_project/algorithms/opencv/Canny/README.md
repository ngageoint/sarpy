# Canny
This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts.

Finds edges in an image using the Canny algorithm with custom image gradient.
Version: 4.0.0
License: BSD
Homepage: [https://docs.opencv.org/4.0.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de]

## Parameters:
Name|Description|Required
---|---|:---:
image_array|Numpy Image Array|Yes
threshold1|First threshold for the hysteresis procedure.|Yes
threshold2|second threshold for the hysteresis procedure.|Yes
apertureSize|aperture size for the Sobel operator.|
L2gradient|a flag, indicating whether a more accurate L2 norm =(dI/dx)2+(dI/dy)2‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾√ should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).|

## Outputs:
Name|Description
---|---
image_array|Numpy Image Array
