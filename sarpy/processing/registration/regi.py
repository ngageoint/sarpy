"""
Basic image registration, generally best suited most suited for coherent image
collection. This is mostly based on an approach developed at Sandia which is generally referred to
by the name "regi".
"""

__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

# TODO: rename this to something more appropriate, and fix the doc string

# TODO: notes
#  - preprocess - image scaled by root mean square, then square root taken (why square root?)
#       * I think that this is not really necessary
#  - multi-stage control point estimate - goes coarse grid to progressively finer (stopping criterion?)
#       * the matlab forces you to choose the number of divisions, keeps the box size
#         and grid spacing the same, and steps down the decimation by a factor of 2
#  - "single stage control point estimate" (cpssestimate.m)
#       * does correlation over box of given size, over search area of given size
#       * at each point of grid, we determine the minimum difference of patch (after scaling and all that),
#         by doing a correlation. I am going to change that to maximizing the inner product
#         (this eliminates the need for obtuse preprocessing), and could be coherent or incoherent
#       * this determines grid of points and offsets at each grid point
#   - matlab appears to use box size of 25 x 25, grid spacing of 15 x 15, search area of 15 x 15,
#     and decimation that steps down by a factor of 2.
#   - the end product is really this final collection of grid points and offsets at each grid point
#     which is then used to formulate a warp transform.
#   - I will worry about the warp transform later, and just focus on point information for now
#   - Note that there is NO ROTATION here

