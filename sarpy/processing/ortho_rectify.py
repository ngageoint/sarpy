"""
SICD ortho-rectification methodology.

Examples
--------
An basic example.

.. code-block:: python

    import os
    from matplotlib import pyplot
    from sarpy.io.complex.converter import open_complex
    from sarpy.processing.ortho_rectify import NearestNeighborMethod

    reader = open_complex('<file name>')
    orth_method = NearestNeighborMethod(reader, index=0, proj_helper=None,
        complex_valued=False, apply_radiometric=None, subtract_radiometric_noise=False)

    # Perform ortho-rectification of the entire image
    # This will take a long time and be very RAM intensive, unless the image is small
    ortho_bounds = orth_method.get_full_ortho_bounds()
    ortho_data = orth_method.get_orthorectified_for_ortho_bounds(ortho_bounds)

    # or, perform ortho-rectification on a given rectangular region in pixel space
    pixel_bounds = [100, 200, 200, 300]  # [first_row, last_row, first_column, last_column]
    ortho_data = orth_method.get_orthorectified_for_pixel_bounds(pixel_bounds)

    # view the data using matplotlib
    fig, axs = pyplot.subplots(nrows=1, ncols=1)
    h1 = axs.imshow(ortho_data, cmap='inferno', aspect='equal')
    fig.colorbar(h1, ax=axs)
    pyplot.show()

A viable example performing orthorectification on the entire image and then saving
to a JPEG.

.. code-block:: python

    import os
    from sarpy.io.complex.converter import open_complex
    from sarpy.processing.ortho_rectify import NearestNeighborMethod, OrthorectificationIterator, FullResolutionFetcher
    import PIL.Image  # note that this is not a required sarpy dependency

    # again, open a sicd type file
    reader = open_complex('<file name>')
    # construct an ortho-rectification helper
    orth_method = NearestNeighborMethod(reader, index=0, proj_helper=None,
        complex_valued=False, apply_radiometric=None, subtract_radiometric_noise=False)

    # now, construct an orthorectification iterator - for block processing of orthorectification
    calculator = FullResolutionFetcher(
        ortho_helper.reader, index=ortho_helper.index, block_size=10)
    ortho_iterator = OrthorectificationIterator(ortho_helper, calculator=calculator, bounds=bounds)

    # now, iterate and populate - the whole orthorectified image result workspace
    # will be kept in RAM, but the processing is iterative and much faster.
    image_data = numpy.zeros(ortho_iterator.ortho_data_size, dtype='uint8')
    for data, start_indices in ortho_iterator:
        image_data[
        start_indices[0]:start_indices[0]+data.shape[0],
        start_indices[1]:start_indices[1]+data.shape[1]] = data
    # convert image array to PIL image.
    img = PIL.Image.fromarray(image_data)
    # save the file
    img.save('<image file name.jpeg>')
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
from typing import Union, Tuple, List, Any

import numpy
from scipy.interpolate import RectBivariateSpline

from sarpy.io.complex.converter import open_complex
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.general.slice_parsing import validate_slice_int, validate_slice
from sarpy.io.complex.utils import get_fetch_block_size, extract_blocks

from sarpy.io.complex.sicd_elements.blocks import Poly2DType
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic, wgs_84_norm
from sarpy.geometry.geometry_elements import GeometryObject

from sarpy.visualization.remap import RemapFunction


logger = logging.getLogger(__name__)

##################
# module variables and helper methods
_PIXEL_METHODOLOGY = ('MAX', 'MIN', 'MEAN', 'GEOM_MEAN')


def _linear_fill(pixel_array, fill_interval=1):
    """
    This is to final in linear features in pixel space at the given interval.

    Parameters
    ----------
    pixel_array : numpy.ndarray
    fill_interval : int|float

    Returns
    -------
    numpy.ndarray
    """

    if not isinstance(pixel_array, numpy.ndarray):
        raise TypeError('pixel_array must be a numpy array. Got type {}'.format(type(pixel_array)))
    if pixel_array.ndim < 2:
        # nothing to be done
        return pixel_array

    if pixel_array.ndim > 2:
        raise ValueError('pixel_array must be no more than two-dimensional. Got shape {}'.format(pixel_array.shape))
    if pixel_array.shape[1] != 2:
        raise ValueError(
            'pixel_array is two dimensional, and the final dimension must have length 2. '
            'Got shape {}'.format(pixel_array.shape))
    if pixel_array.shape[0] < 2:
        # nothing to be done
        return pixel_array

    def make_segment(start_point, end_point):
        segment_length = numpy.linalg.norm(end_point - start_point)
        segment_count = max(2, int(numpy.ceil(segment_length/float(fill_interval) + 1)))
        segment = numpy.zeros((segment_count, 2), dtype=numpy.float64)
        # NB: it's ok if start == end in linspace
        segment[:, 0] = numpy.linspace(start_point[0], end_point[0], segment_count)
        segment[:, 1] = numpy.linspace(start_point[1], end_point[1], segment_count)
        return segment

    segments = []
    for i in range(pixel_array.shape[0]-1):
        start_segment = pixel_array[i, :]
        end_segment = pixel_array[i+1, :]
        segments.append(make_segment(start_segment, end_segment))
    return numpy.vstack(segments)


#################
# The projection methodology

class ProjectionHelper(object):
    """
    Abstract helper class which defines the projection interface for
    ortho-rectification usage for a sicd type object.
    """

    __slots__ = ('_sicd', '_row_spacing', '_col_spacing', '_default_pixel_method')

    def __init__(self, sicd, row_spacing=None, col_spacing=None, default_pixel_method='GEOM_MEAN'):
        r"""

        Parameters
        ----------
        sicd : SICDType
            The sicd object
        row_spacing : None|float
            The row pixel spacing. If not provided, this will default according
            to `default_pixel_method`.
        col_spacing : None|float
            The row pixel spacing. If not provided, this will default according
            to `default_pixel_method`.
        default_pixel_method : str
            Must be one of ('MAX', 'MIN', 'MEAN', 'GEOM_MEAN'). This determines
            the default behavior for row_spacing/col_spacing. The default value for
            row/column spacing will be the implied function applied to the range
            and azimuth ground resolution. Note that geometric mean is defined as
            :math:`\sqrt(x*x + y*y)`
        """

        self._row_spacing = None
        self._col_spacing = None
        default_pixel_method = default_pixel_method.upper()
        if default_pixel_method not in _PIXEL_METHODOLOGY:
            raise ValueError(
                'default_pixel_method got invalid value {}. Must be one '
                'of {}'.format(default_pixel_method, _PIXEL_METHODOLOGY))
        self._default_pixel_method = default_pixel_method

        if not isinstance(sicd, SICDType):
            raise TypeError('sicd must be a SICDType instance. Got type {}'.format(type(sicd)))
        if not sicd.can_project_coordinates():
            raise ValueError('Ortho-rectification requires the SICD ability to project coordinates.')
        sicd.define_coa_projection(overide=False)
        self._sicd = sicd
        self.row_spacing = row_spacing
        self.col_spacing = col_spacing

    @property
    def sicd(self):
        """
        SICDType: The sicd structure.
        """

        return self._sicd

    @property
    def row_spacing(self):
        """
        float: The row pixel spacing
        """

        return self._row_spacing

    @row_spacing.setter
    def row_spacing(self, value):
        """
        Set the row pixel spacing value. Setting to None will result in a
        default value derived from the SICD structure being used.

        Parameters
        ----------
        value : None|float

        Returns
        -------
        None
        """

        if value is None:
            if self.sicd.RadarCollection.Area is None:
                self._row_spacing = self._get_sicd_ground_pixel()
            else:
                self._row_spacing = self.sicd.RadarCollection.Area.Plane.XDir.LineSpacing
        else:
            value = float(value)
            if value <= 0:
                raise ValueError('row pixel spacing must be positive.')
            self._row_spacing = float(value)

    @property
    def col_spacing(self):
        """
        float: The column pixel spacing
        """

        return self._col_spacing

    @col_spacing.setter
    def col_spacing(self, value):
        """
        Set the col pixel spacing value. Setting to NOne will result in a
        default value derived from the SICD structure being used.

        Parameters
        ----------
        value : None|float

        Returns
        -------
        None
        """

        if value is None:
            if self.sicd.RadarCollection.Area is None:
                self._col_spacing = self._get_sicd_ground_pixel()
            else:
                self._col_spacing = self.sicd.RadarCollection.Area.Plane.YDir.SampleSpacing
        else:
            value = float(value)
            if value <= 0:
                raise ValueError('column pixel spacing must be positive.')
            self._col_spacing = float(value)

    def _get_sicd_ground_pixel(self):
        """
        Gets the SICD ground pixel size.

        Returns
        -------
        float
        """

        ground_row_ss, ground_col_ss = self.sicd.get_ground_resolution()
        if self._default_pixel_method == 'MIN':
            return min(ground_row_ss, ground_col_ss)
        elif self._default_pixel_method == 'MAX':
            return max(ground_row_ss, ground_col_ss)
        elif self._default_pixel_method == 'MEAN':
            return 0.5*(ground_row_ss + ground_col_ss)
        elif self._default_pixel_method == 'GEOM_MEAN':
            return float(numpy.sqrt(ground_row_ss*ground_row_ss + ground_col_ss*ground_col_ss))
        else:
            raise ValueError('Got unhandled default_pixel_method {}'.format(self._default_pixel_method))

    @staticmethod
    def _reshape(array, final_dimension):
        """
        Reshape the input so that the output is two-dimensional with final
        dimension given by `final_dimension`.

        Parameters
        ----------
        array : numpy.ndarray|list|tuple
        final_dimension : int

        Returns
        -------
        (numpy.ndarray, tuple)
            The reshaped data array and original shape.
        """

        if not isinstance(array, numpy.ndarray):
            array = numpy.array(array, dtype=numpy.float64)
        if array.ndim < 1 or array.shape[-1] != final_dimension:
            raise ValueError(
                'ortho_coords must be at least one dimensional with final dimension '
                'of size {}.'.format(final_dimension))

        o_shape = array.shape

        if array.ndim != 2:
            array = numpy.reshape(array, (-1, final_dimension))
        return array, o_shape

    def ecf_to_ortho(self, coords):
        """
        Gets the `(ortho_row, ortho_column)` coordinates in the ortho-rectified
        system for the provided physical coordinates in ECF `(X, Y, Z)` coordinates.

        Parameters
        ----------
        coords : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def ecf_to_pixel(self, coords):
        """
        Gets the `(pixel_row, pixel_column)` coordinates for the provided physical
        coordinates in ECF `(X, Y, Z)` coordinates.

        Parameters
        ----------
        coords : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def ll_to_ortho(self, ll_coords):
        """
        Gets the `(ortho_row, ortho_column)` coordinates in the ortho-rectified
        system for the provided physical coordinates in `(Lat, Lon)` coordinates.

        Note that there is inherent ambiguity when handling the missing elevation,
        and the effect is likely methodology dependent.

        Parameters
        ----------
        ll_coords : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def llh_to_ortho(self, llh_coords):
        """
        Gets the `(ortho_row, ortho_column)` coordinates in the ortho-rectified
        system for the providednphysical coordinates in `(Lat, Lon, HAE)`
        coordinates.

        Parameters
        ----------
        llh_coords : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def pixel_to_ortho(self, pixel_coords):
        """
        Gets the ortho-rectified indices for the point(s) in pixel coordinates.

        Parameters
        ----------
        pixel_coords : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def pixel_to_ecf(self, pixel_coords):
        """
        Gets the ECF coordinates for the point(s) in pixel coordinates.

        Parameters
        ----------
        pixel_coords : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def ortho_to_ecf(self, ortho_coords):
        """
        Get the ecf coordinates for the point(s) in ortho-rectified coordinates.

        Parameters
        ----------
        ortho_coords : numpy.ndarray
            Point(s) in the ortho-recitified coordinate system, of the form
            `(ortho_row, ortho_column)`.

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def ortho_to_llh(self, ortho_coords):
        """
        Get the lat/lon/hae coordinates for the point(s) in ortho-rectified coordinates.

        Parameters
        ----------
        ortho_coords : numpy.ndarray
            Point(s) in the ortho-recitified coordinate system, of the form
            `(ortho_row, ortho_column)`.

        Returns
        -------
        numpy.ndarray
        """

        ecf = self.ortho_to_ecf(ortho_coords)
        return ecf_to_geodetic(ecf)

    def ortho_to_pixel(self, ortho_coords):
        """
        Get the pixel indices for the point(s) in ortho-rectified coordinates.

        Parameters
        ----------
        ortho_coords : numpy.ndarray
            Point(s) in the ortho-recitified coordinate system, of the form
            `(ortho_row, ortho_column)`.

        Returns
        -------
        numpy.ndarray
            The array of indices, of the same shape as `new_coords`, which indicate
            `(row, column)` pixel (fractional) indices.
        """

        raise NotImplementedError

    def get_pixel_array_bounds(self, coords):
        """
        Extract integer bounds of the input array, expected to have final dimension
        of size 2.

        Parameters
        ----------
        coords : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Of the form `(min_row, max_row, min_column, max_column)`.
        """

        coords, o_shape = self._reshape(coords, 2)
        return numpy.array(
            (numpy.ceil(numpy.nanmin(coords[:, 0], axis=0)),
             numpy.floor(numpy.nanmax(coords[:, 0], axis=0)),
             numpy.ceil(numpy.nanmin(coords[:, 1], axis=0)),
             numpy.floor(numpy.nanmax(coords[:, 1], axis=0))), dtype=numpy.int64)


class PGProjection(ProjectionHelper):
    """
    Class which helps perform the Planar Grid (i.e. Ground Plane) ortho-rectification
    for a sicd-type object. **In this implementation, we have that the reference point
    will have ortho-rectification coordinates (0, 0).** All ortho-rectification coordinate
    interpretation should be relative to the fact.
    """

    __slots__ = (
        '_reference_point', '_reference_pixels', '_row_vector', '_col_vector', '_normal_vector', '_reference_hae')

    def __init__(self, sicd, reference_point=None, reference_pixels=None, normal_vector=None, row_vector=None,
                 col_vector=None, row_spacing=None, col_spacing=None,
                 default_pixel_method='GEOM_MEAN'):
        r"""

        Parameters
        ----------
        sicd : SICDType
            The sicd object
        reference_point : None|numpy.ndarray
            The reference point (origin) of the planar grid. If None, a default
            derived from the SICD will be used.
        reference_pixels : None|numpy.ndarray
            The projected pixel
        normal_vector : None|numpy.ndarray
            The unit normal vector of the plane.
        row_vector : None|numpy.ndarray
            The vector defining increasing column direction. If None, a default
            derived from the SICD will be used.
        col_vector : None|numpy.ndarray
            The vector defining increasing column direction. If None, a default
            derived from the SICD will be used.
        row_spacing : None|float
            The row pixel spacing.
        col_spacing : None|float
            The column pixel spacing.
        default_pixel_method : str
            Must be one of ('MAX', 'MIN', 'MEAN', 'GEOM_MEAN'). This determines
            the default behavior for row_spacing/col_spacing. The default value for
            row/column spacing will be the implied function applied to the range
            and azimuth ground resolution. Note that geometric mean is defined as
            :math:`\sqrt(x*x + y*y)`
        """

        self._reference_point = None
        self._reference_hae = None
        self._reference_pixels = None
        self._normal_vector = None
        self._row_vector = None
        self._col_vector = None
        super(PGProjection, self).__init__(
            sicd, row_spacing=row_spacing, col_spacing=col_spacing, default_pixel_method=default_pixel_method)
        self.set_reference_point(reference_point=reference_point)
        self.set_reference_pixels(reference_pixels=reference_pixels)
        self.set_plane_frame(
            normal_vector=normal_vector, row_vector=row_vector, col_vector=col_vector)

    @property
    def reference_point(self):
        # type: () -> numpy.ndarray
        """
        numpy.ndarray: The grid reference point.
        """

        return self._reference_point

    @property
    def reference_pixels(self):
        # type: () -> numpy.ndarray
        """
        numpy.ndarray: The ortho-rectified pixel coordinates of the grid reference point.
        """

        return self._reference_pixels

    @property
    def normal_vector(self):
        # type: () -> numpy.ndarray
        """
        numpy.ndarray: The normal vector.
        """

        return self._normal_vector

    def set_reference_point(self, reference_point=None):
        """
        Sets the reference point, which must be provided in ECF coordinates.

        Parameters
        ----------
        reference_point : None|numpy.ndarray
            The reference point (origin) of the planar grid. If None, then the
            `sicd.GeoData.SCP.ECF` will be used.

        Returns
        -------
        None
        """

        if reference_point is None:
            if self.sicd.RadarCollection.Area is None:
                reference_point = self.sicd.GeoData.SCP.ECF.get_array()
            else:
                reference_point = self.sicd.RadarCollection.Area.Plane.RefPt.ECF.get_array()

        if not (isinstance(reference_point, numpy.ndarray) and reference_point.ndim == 1
                and reference_point.size == 3):
            raise ValueError('reference_point must be a vector of size 3.')
        self._reference_point = reference_point
        # set the reference hae
        ref_llh = ecf_to_geodetic(reference_point)
        self._reference_hae = ref_llh[2]

    def set_reference_pixels(self, reference_pixels=None):
        """
        Sets the reference point, which must be provided in ECF coordinates.

        Parameters
        ----------
        reference_pixels : None|numpy.ndarray
            The ortho-rectified pixel coordinates for the reference point (origin) of the planar grid.
            If None, then the (0, 0) will be used.

        Returns
        -------
        None
        """

        if reference_pixels is None:
            if self.sicd.RadarCollection.Area is not None:
                reference_pixels = numpy.array([
                    self.sicd.RadarCollection.Area.Plane.RefPt.Line,
                    self.sicd.RadarCollection.Area.Plane.RefPt.Sample],
                    dtype='float64')
            else:
                reference_pixels = numpy.zeros((2, ), dtype='float64')

        if not (isinstance(reference_pixels, numpy.ndarray) and reference_pixels.ndim == 1
                and reference_pixels.size == 2):
            raise ValueError('reference_pixels must be a vector of size 2.')
        self._reference_pixels = reference_pixels

    @property
    def row_vector(self):
        """
        numpy.ndarray: The grid increasing row direction (ECF) unit vector.
        """

        return self._row_vector

    @property
    def col_vector(self):
        """
        numpy.ndarray: The grid increasing column direction (ECF) unit vector.
        """

        return self._col_vector

    @property
    def reference_hae(self):
        """
        float: The height above the ellipsoid of the reference point.
        """

        return self._reference_hae

    def set_plane_frame(self, normal_vector=None, row_vector=None, col_vector=None):
        """
        Set the plane unit normal, and the row and column vectors, in ECF coordinates.
        Note that the perpendicular component of col_vector with respect to the
        row_vector will be used.

        If `normal_vector`, `row_vector`, and `col_vector` are all `None`, then
        the normal to the Earth tangent plane at the reference point is used for
        `normal_vector`. The `row_vector` will be defined as the perpendicular
        component of `sicd.Grid.Row.UVectECF` to `normal_vector`. The `colummn_vector`
        will be defined as the component of `sicd.Grid.Col.UVectECF` perpendicular
        to both `normal_vector` and `row_vector`.

        If only `normal_vector` is supplied, then the `row_vector` and `column_vector`
        will be defined similarly as the perpendicular components of
        `sicd.Grid.Row.UVectECF` and `sicd.Grid.Col.UVectECF`.

        Otherwise, all vectors supplied will be normalized, but are required to be
        mutually perpendicular. If only two vectors are supplied, then the third
        will be determined.

        Parameters
        ----------
        normal_vector : None|numpy.ndarray
            The vector defining the outward unit normal in ECF coordinates.
        row_vector : None|numpy.ndarray
            The vector defining increasing column direction.
        col_vector : None|numpy.ndarray
            The vector defining increasing column direction.

        Returns
        -------
        None
        """

        def normalize(vec, name, perp=None):
            if not isinstance(vec, numpy.ndarray):
                vec = numpy.array(vec, dtype=numpy.float64)
            if not (isinstance(vec, numpy.ndarray) and vec.ndim == 1 and vec.size == 3):
                raise ValueError('{} vector must be a numpy.ndarray of dimension 1 and size 3.'.format(name))
            vec = numpy.copy(vec)
            if perp is None:
                pass
            elif isinstance(perp, numpy.ndarray):
                vec = vec - perp*(perp.dot(vec))
            else:
                for entry in perp:
                    vec = vec - entry*(entry.dot(vec))

            norm = numpy.linalg.norm(vec)
            if norm == 0:
                raise ValueError('{} vector cannot be the zero vector.'.format(name))
            elif norm != 1:
                vec = vec/norm  # avoid modifying row_vector def exterior to this class
            return vec

        def check_perp(vec1, vec2, name1, name2, tolerance=1e-6):
            if abs(vec1.dot(vec2)) > tolerance:
                raise ValueError('{} vector and {} vector are not perpendicular'.format(name1, name2))

        if self._reference_point is None:
            raise ValueError('This requires that reference point is previously set.')

        if normal_vector is None and row_vector is None and col_vector is None:
            if self.sicd.RadarCollection.Area is None:
                self._normal_vector = wgs_84_norm(self.reference_point)
                self._row_vector = normalize(
                    self.sicd.Grid.Row.UVectECF.get_array(), 'row', perp=self.normal_vector)
                self._col_vector = normalize(
                    self.sicd.Grid.Col.UVectECF.get_array(), 'column', perp=(self.normal_vector, self.row_vector))
            else:
                self._row_vector = self.sicd.RadarCollection.Area.Plane.XDir.UVectECF.get_array()
                self._col_vector = normalize(
                    self.sicd.RadarCollection.Area.Plane.YDir.UVectECF.get_array(), 'col', perp=self._row_vector)
                self._normal_vector = numpy.cross(self._row_vector, self._col_vector)
        elif normal_vector is not None and row_vector is None and col_vector is None:
            self._normal_vector = normalize(normal_vector, 'normal')
            self._row_vector = normalize(
                self.sicd.Grid.Row.UVectECF.get_array(), 'row', perp=self.normal_vector)
            self._col_vector = normalize(
                self.sicd.Grid.Col.UVectECF.get_array(), 'column', perp=(self.normal_vector, self.row_vector))
        elif normal_vector is None:
            if row_vector is None or col_vector is None:
                raise ValueError('normal_vector is not defined, so both row_vector and col_vector must be.')
            row_vector = normalize(row_vector, 'row')
            col_vector = normalize(col_vector, 'col')
            check_perp(row_vector, col_vector, 'row', 'col')
            self._row_vector = row_vector
            self._col_vector = col_vector
            self._normal_vector = numpy.cross(row_vector, col_vector)
        elif col_vector is None:
            if row_vector is None:
                raise ValueError('col_vector is not defined, so both normal_vector and row_vector must be.')
            normal_vector = normalize(normal_vector, 'normal')
            row_vector =  normalize(row_vector, 'row')
            check_perp(normal_vector, row_vector, 'normal', 'row')
            self._normal_vector = normal_vector
            self._row_vector = row_vector
            self._col_vector = numpy.cross(self.normal_vector, self.row_vector)
        elif row_vector is None:
            normal_vector = normalize(normal_vector, 'normal')
            col_vector =  normalize(col_vector, 'col')
            check_perp(normal_vector, col_vector, 'normal', 'col')
            self._normal_vector = normal_vector
            self._col_vector = col_vector
            self._row_vector = numpy.cross(self.col_vector, self.normal_vector)
        else:
            normal_vector = normalize(normal_vector, 'normal')
            row_vector = normalize(row_vector, 'row')
            col_vector =  normalize(col_vector, 'col')
            check_perp(normal_vector, row_vector, 'normal', 'row')
            check_perp(normal_vector, col_vector, 'normal', 'col')
            check_perp(row_vector, col_vector, 'row', 'col')
            self._normal_vector = normal_vector
            self._row_vector = row_vector
            self._col_vector = col_vector
        # check for outward unit norm
        if numpy.dot(self.normal_vector, self.reference_point) < 0:
            logger.warning(
                'The normal vector appears to be outward pointing, so reversing.')
            self._normal_vector *= -1

    def ecf_to_ortho(self, coords):
        coords, o_shape = self._reshape(coords, 3)
        diff = coords - self.reference_point
        if len(o_shape) == 1:
            out = numpy.zeros((2, ), dtype=numpy.float64)
            out[0] = self._reference_pixels[0] + numpy.sum(diff*self.row_vector)/self.row_spacing
            out[1] = self._reference_pixels[1] + numpy.sum(diff*self.col_vector)/self.col_spacing
        else:
            out = numpy.zeros((coords.shape[0], 2), dtype=numpy.float64)
            out[:, 0] = self._reference_pixels[0] + numpy.sum(diff*self.row_vector, axis=1)/self.row_spacing
            out[:, 1] = self._reference_pixels[1] + numpy.sum(diff*self.col_vector, axis=1)/self.col_spacing
            out = numpy.reshape(out, o_shape[:-1] + (2, ))
        return out

    def ecf_to_pixel(self, coords):
        pixel, _, _ = self.sicd.project_ground_to_image(coords)
        return pixel

    def ll_to_ortho(self, ll_coords):
        """
        Gets the `(ortho_row, ortho_column)` coordinates in the ortho-rectified
        system for the provided physical coordinates in `(Lat, Lon)` coordinates.
        In this case, the missing altitude will be set to `reference_hae`, which
        is imperfect.

        Parameters
        ----------
        ll_coords : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        ll_coords, o_shape = self._reshape(ll_coords, 2)
        llh_temp = numpy.zeros((ll_coords.shape[0], 3), dtype=numpy.float64)
        llh_temp[:, :2] = ll_coords
        llh_temp[:, 2] = self.reference_hae
        llh_temp = numpy.reshape(llh_temp, o_shape[:-1]+ (3, ))
        return self.llh_to_ortho(llh_temp)

    def llh_to_ortho(self, llh_coords):
        llh_coords, o_shape = self._reshape(llh_coords, 3)
        ground = geodetic_to_ecf(llh_coords)
        return self.ecf_to_ortho(numpy.reshape(ground, o_shape))

    def ortho_to_ecf(self, ortho_coords):
        ortho_coords, o_shape = self._reshape(ortho_coords, 2)
        xs = (ortho_coords[:, 0] - self._reference_pixels[0])*self.row_spacing
        ys = (ortho_coords[:, 1] - self._reference_pixels[1])*self.col_spacing
        if xs.ndim == 0:
            coords = self.reference_point + xs*self.row_vector + ys*self.col_vector
        else:
            coords = self.reference_point + numpy.outer(xs, self.row_vector) + \
                     numpy.outer(ys, self.col_vector)
        return numpy.reshape(coords, o_shape[:-1] + (3, ))

    def ortho_to_pixel(self, ortho_coords):
        ortho_coords, o_shape = self._reshape(ortho_coords, 2)
        pixel, _, _ = self.sicd.project_ground_to_image(self.ortho_to_ecf(ortho_coords))
        return numpy.reshape(pixel, o_shape)

    def pixel_to_ortho(self, pixel_coords):
        return self.ecf_to_ortho(self.pixel_to_ecf(pixel_coords))

    def pixel_to_ecf(self, pixel_coords):
        return self.sicd.project_image_to_ground(
            pixel_coords, projection_type='PLANE',
            gref=self.reference_point, ugpn=self.normal_vector)


################
# The orthorectification methodology

class OrthorectificationHelper(object):
    """
    Abstract helper class which defines ortho-rectification process for a sicd-type
    reader object.
    """

    __slots__ = (
        '_reader', '_index', '_sicd', '_proj_helper', '_out_dtype', '_complex_valued',
        '_pad_value', '_apply_radiometric', '_subtract_radiometric_noise',
        '_rad_poly', '_noise_poly', '_default_physical_bounds')

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False,
                 pad_value=None, apply_radiometric=None, subtract_radiometric_noise=False):
        """

        Parameters
        ----------
        reader : SICDTypeReader
        index : int
        proj_helper : None|ProjectionHelper
            If `None`, this will default to `PGProjection(<sicd>)`, where `<sicd>`
            will be the sicd from `reader` at `index`. Otherwise, it is the user's
            responsibility to ensure that `reader`, `index` and `proj_helper` are
            in sync.
        complex_valued : bool
            Do we want complex values returned? If `False`, the magnitude values
            will be used.
        pad_value : None|Any
            Value to use for any out-of-range pixels. Defaults to `0` if not provided.
        apply_radiometric : None|str
            If provided, must be one of `['RCS', 'Sigma0', 'Gamma0', 'Beta0']`
            (not case-sensitive). This will apply the given radiometric scale factor
            to calculated pixel power, with noise subtracted if `subtract_radiometric_noise = True`.
            **Only valid if `complex_valued=False`**.
        subtract_radiometric_noise : bool
            This indicates whether the radiometric noise should be subtracted from
            the pixel amplitude. **Only valid if `complex_valued=False`**.
        """

        self._index = None
        self._sicd = None
        self._proj_helper = None
        self._apply_radiometric = None
        self._subtract_radiometric_noise = None
        self._rad_poly = None  # type: [None, Poly2DType]
        self._noise_poly = None  # type: [None, Poly2DType]
        self._default_physical_bounds = None

        self._pad_value = pad_value
        self._complex_valued = complex_valued
        if self._complex_valued:
            self._out_dtype = numpy.dtype('complex64')
        else:
            self._out_dtype = numpy.dtype('float32')
        if not isinstance(reader, SICDTypeReader):
            raise TypeError('Got unexpected type {} for reader'.format(type(reader)))
        self._reader = reader
        self.apply_radiometric = apply_radiometric
        self.subtract_radiometric_noise = subtract_radiometric_noise
        self.set_index_and_proj_helper(index, proj_helper=proj_helper)

    @property
    def reader(self):
        # type: () -> SICDTypeReader
        """
        SICDTypeReader: The reader instance.
        """

        return self._reader

    @property
    def index(self):
        # type: () -> int
        """
        int: The index for the desired sicd element.
        """
        return self._index

    @property
    def sicd(self):
        # type: () -> SICDType
        """
        SICDType: The sicd structure.
        """

        return self._sicd

    @property
    def proj_helper(self):
        # type: () -> ProjectionHelper
        """
        ProjectionHelper: The projection helper instance.
        """

        return self._proj_helper

    @property
    def out_dtype(self):
        # type: () -> numpy.dtype
        """
        numpy.dtype: The output data type.
        """

        return self._out_dtype

    @property
    def pad_value(self):
        """
        The value to use for any portions of the array which extend beyond the range
        of where the reader has data.
        """

        return self._pad_value

    @pad_value.setter
    def pad_value(self, value):
        self._pad_value = value

    def set_index_and_proj_helper(self, index, proj_helper=None):
        """
        Sets the index and proj_helper objects.

        Parameters
        ----------
        index : int
        proj_helper : ProjectionHelper

        Returns
        -------
        None
        """

        self._index = index
        self._sicd = self.reader.get_sicds_as_tuple()[index]
        self._is_radiometric_valid()
        self._is_radiometric_noise_valid()

        default_ortho_bounds = None
        if proj_helper is None:
            proj_helper = PGProjection(self.sicd)
            if self.sicd.RadarCollection is not None and self.sicd.RadarCollection.Area is not None \
                    and self.sicd.RadarCollection.Area.Plane is not None:
                plane = self.sicd.RadarCollection.Area.Plane
                default_ortho_bounds = numpy.array([
                    plane.XDir.FirstLine, plane.XDir.FirstLine+plane.XDir.NumLines,
                    plane.YDir.FirstSample, plane.YDir.FirstSample+plane.YDir.NumSamples], dtype=numpy.uint32)

        if not isinstance(proj_helper, ProjectionHelper):
            raise TypeError('Got unexpected type {} for proj_helper'.format(proj_helper))
        self._proj_helper = proj_helper
        if default_ortho_bounds is not None:
            _, ortho_rectangle = self.bounds_to_rectangle(default_ortho_bounds)
            self._default_physical_bounds = self.proj_helper.ortho_to_ecf(ortho_rectangle)

    @property
    def apply_radiometric(self):
        # type: () -> Union[None, str]
        """
        None|str: This indicates which, if any, of the radiometric scale factors
        to apply in the result. If not `None`, this must be one of ('RCS', 'SIGMA0', 'GAMMA0', 'BETA0').

        Setting to a value other than `None` will result in an error if 1.) `complex_valued` is `True`, or
        2.) the appropriate corresponding element `sicd.Radiometric.RCSSFPoly`,
        `sicd.Radiometric.SigmaZeroSFPoly`, `sicd.Radiometric.GammaZeroSFPoly`, or
        `sicd.Radiometric.BetaZeroSFPoly` is not populated with a valid polynomial.
        """

        return self._apply_radiometric

    @apply_radiometric.setter
    def apply_radiometric(self, value):
        if value is None:
            self._apply_radiometric = None
        elif isinstance(value, str):
            val = value.upper()
            allowed = ('RCS', 'SIGMA0', 'GAMMA0', 'BETA0')
            if val not in allowed:
                raise ValueError('Require that value is one of {}, got {}'.format(allowed, val))
            self._apply_radiometric = val
            self._is_radiometric_valid()
        else:
            raise TypeError('Got unexpected type {} for apply_radiometric'.format(type(value)))

    @property
    def subtract_radiometric_noise(self):
        """
        bool: This indicates whether the radiometric noise should be subtracted from
        the pixel amplitude. If `apply_radiometric` is not `None`, then this subtraction
        will happen applying the corresponding scaling.

        Setting this to `True` will **result in an error** unless the given sicd structure has
        `sicd.Radiometric.NoiseLevel.NoisePoly` populated with a viable polynomial and
        `sicd.Radiometric.NoiseLevel.NoiseLevelType == 'ABSOLUTE'`.
        """

        return self._subtract_radiometric_noise

    @subtract_radiometric_noise.setter
    def subtract_radiometric_noise(self, value):
        if value:
            self._subtract_radiometric_noise = True
        else:
            self._subtract_radiometric_noise = False
        self._is_radiometric_noise_valid()

    def _is_radiometric_valid(self):
        """
        Checks whether the apply radiometric settings are valid.

        Returns
        -------
        None
        """

        if self.apply_radiometric is None:
            self._rad_poly = None
            return  # nothing to be done
        if self._complex_valued:
            raise ValueError('apply_radiometric is not None, which requires real valued output.')
        if self.sicd is None:
            return  # nothing to be done, no sicd set (yet)

        if self.sicd.Radiometric is None:
            raise ValueError(
                'apply_radiometric is {}, but sicd.Radiometric is unpopulated.'.format(self.apply_radiometric))

        if self.apply_radiometric == 'RCS':
            if self.sicd.Radiometric.RCSSFPoly is None:
                raise ValueError('apply_radiometric is "RCS", but the sicd.Radiometric.RCSSFPoly is not populated.')
            else:
                self._rad_poly = self.sicd.Radiometric.RCSSFPoly
        elif self.apply_radiometric == 'SIGMA0':
            if self.sicd.Radiometric.SigmaZeroSFPoly is None:
                raise ValueError('apply_radiometric is "SIGMA0", but the sicd.Radiometric.SigmaZeroSFPoly is not populated.')
            else:
                self._rad_poly = self.sicd.Radiometric.SigmaZeroSFPoly
        elif self.apply_radiometric == 'GAMMA0':
            if self.sicd.Radiometric.GammaZeroSFPoly is None:
                raise ValueError('apply_radiometric is "GAMMA0", but the sicd.Radiometric.GammaZeroSFPoly is not populated.')
            else:
                self._rad_poly = self.sicd.Radiometric.GammaZeroSFPoly
        elif self.apply_radiometric == 'BETA0':
            if self.sicd.Radiometric.BetaZeroSFPoly is None:
                raise ValueError('apply_radiometric is "BETA0", but the sicd.Radiometric.BetaZeroSFPoly is not populated.')
            else:
                self._rad_poly = self.sicd.Radiometric.BetaZeroSFPoly
        else:
            raise ValueError('Got unhandled value {} for apply_radiometric'.format(self.apply_radiometric))

    def _is_radiometric_noise_valid(self):
        """
        Checks whether the subtract_radiometric_noise setting is valid.

        Returns
        -------
        None
        """

        if not self.subtract_radiometric_noise:
            self._noise_poly = None
            return  # nothing to be done
        if self._complex_valued:
            raise ValueError('subtract_radiometric_noise is True, which requires real valued output.')
        if self.sicd is None:
            return  # nothing to be done, no sicd set (yet)

        # set the noise polynomial value
        if self.sicd.Radiometric is None:
            raise ValueError('subtract_radiometric_noise is True, but sicd.Radiometric is unpopulated.')

        if self.sicd.Radiometric.NoiseLevel is None:
            raise ValueError(
                'subtract_radiometric_noise is set to True, but sicd.Radiometric.NoiseLevel is not populated.')
        if self.sicd.Radiometric.NoiseLevel.NoisePoly is None:
            raise ValueError(
                'subtract_radiometric_noise is set to True, but sicd.Radiometric.NoiseLevel.NoisePoly is not populated.')
        if self.sicd.Radiometric.NoiseLevel.NoiseLevelType == 'RELATIVE':
            raise ValueError(
                'subtract_radiometric_noise is set to True, but sicd.Radiometric.NoiseLevel.NoiseLevelType is "RELATIVE"')
        self._noise_poly = self.sicd.Radiometric.NoiseLevel.NoisePoly

    def get_full_ortho_bounds(self):
        """
        Gets the bounds for the ortho-rectified coordinates for the full sicd image.

        Returns
        -------
        numpy.ndarray
            Of the form `[min row, max row, min column, max column]`.
        """

        if self._default_physical_bounds is not None:
            ortho_rectangle = self.proj_helper.ecf_to_ortho(self._default_physical_bounds)
            return self.proj_helper.get_pixel_array_bounds(ortho_rectangle)

        full_coords = self.sicd.ImageData.get_full_vertex_data()
        full_line = _linear_fill(full_coords, fill_interval=1)
        return self.get_orthorectification_bounds_from_pixel_object(full_line)

    def get_valid_ortho_bounds(self):
        """
        Gets the bounds for the ortho-rectified coordinates for the valid portion
        of the sicd image. This is the outer bounds of the valid portion, so may contain
        some portion which is not itself valid.

        If sicd.ImageData.ValidData is not defined, then the full image bounds will
        be returned.

        Returns
        -------
        numpy.ndarray
            Of the form `[min row, max row, min column, max column]`.
        """

        if self._default_physical_bounds is not None:
            ortho_rectangle = self.proj_helper.ecf_to_ortho(self._default_physical_bounds)
            return self.proj_helper.get_pixel_array_bounds(ortho_rectangle)

        valid_coords = self.sicd.ImageData.get_valid_vertex_data()
        if valid_coords is None:
            valid_coords = self.sicd.ImageData.get_full_vertex_data()
        valid_line = _linear_fill(valid_coords, fill_interval=1)
        return self.get_orthorectification_bounds_from_pixel_object(valid_line)

    def get_orthorectification_bounds_from_pixel_object(self, coordinates):
        """
        Determine the ortho-rectified (coordinate-system aligned) rectangular bounding
        region which contains the provided coordinates in pixel space.

        Parameters
        ----------
        coordinates : GeometryObject|numpy.ndarray|list|tuple
            The coordinate system of the input will be assumed to be pixel space.

        Returns
        -------
        numpy.ndarray
            Of the form `(row_min, row_max, col_min, col_max)`.
        """

        if isinstance(coordinates, GeometryObject):
            pixel_bounds = coordinates.get_bbox()
            siz = int(len(pixel_bounds)/2)
            coordinates = numpy.array(
                [[pixel_bounds[0], pixel_bounds[1]],
                 [pixel_bounds[siz], pixel_bounds[1]],
                 [pixel_bounds[siz], pixel_bounds[siz]],
                 [pixel_bounds[0], pixel_bounds[siz]]], dtype=numpy.float64)
        filled_coordinates = _linear_fill(coordinates, fill_interval=1)
        ortho = self.proj_helper.pixel_to_ortho(filled_coordinates)
        return self.proj_helper.get_pixel_array_bounds(ortho)

    def get_orthorectification_bounds_from_latlon_object(self, coordinates):
        """
        Determine the ortho-rectified (coordinate-system aligned) rectangular bounding
        region which contains the provided coordinates in lat/lon space.

        Parameters
        ----------
        coordinates : GeometryObject|numpy.ndarray|list|tuple
            The coordinate system of the input will be assumed to be lat/lon space.
            **Note** a GeometryObject is expected to follow lon/lat ordering paradigm,
            by convention.

        Returns
        -------
        numpy.ndarray
            Of the form `(row_min, row_max, col_min, col_max)`.
        """

        if isinstance(coordinates, GeometryObject):
            # Note we assume a geometry object is using lon/lat ordering of coordinates.
            bounds = coordinates.get_bbox()
            if len(bounds) == 4:
                coordinates = numpy.array(
                    [[bounds[1], bounds[0]],
                     [bounds[1], bounds[2]],
                     [bounds[3], bounds[2]],
                     [bounds[3], bounds[0]]], dtype=numpy.float64)
            elif len(bounds) >= 6:
                siz = int(len(bounds)/2)
                coordinates = numpy.array(
                    [[bounds[1], bounds[0], bounds[3]],
                     [bounds[1], bounds[siz], bounds[3]],
                     [bounds[3], bounds[2], bounds[3]],
                     [bounds[3], bounds[0], bounds[3]]], dtype=numpy.float64)
            else:
                raise ValueError(
                    'It is expected that the geometry object "coordinates" uses two '
                    'or three dimensional coordinates. Got {} for a bounding box.'.format(bounds))
        if not isinstance(coordinates, numpy.ndarray):
            coordinates = numpy.array(coordinates, dtype=numpy.float64)
        if coordinates.shape[-1] == 2:
            ortho = self.proj_helper.ll_to_ortho(coordinates)
        elif coordinates.shape[-1] == 3:
            ortho = self.proj_helper.llh_to_ortho(coordinates)
        else:
            raise ValueError('Got unexpected shape for coordinates {}'.format(coordinates.shape))
        return self.proj_helper.get_pixel_array_bounds(ortho)

    @staticmethod
    def validate_bounds(bounds):
        """
        Validate a pixel type bounds array.

        Parameters
        ----------
        bounds : numpy.ndarray|list|tuple

        Returns
        -------
        numpy.ndarray
        """

        if not isinstance(bounds, numpy.ndarray):
            bounds = numpy.asarray(bounds)

        if bounds.ndim != 1 or bounds.size != 4:
            raise ValueError(
                'bounds is required to be one-dimensional and size 4. '
                'Got input shape {}'.format(bounds.shape))
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError(
                'bounds is required to be of the form (min row, max row, min col, max col), '
                'got {}'.format(bounds))

        if issubclass(bounds.dtype.type, numpy.integer):
            return bounds
        else:
            out = numpy.zeros((4,), dtype=numpy.int32)
            out[0:3:2] = (numpy.floor(bounds[0]), numpy.floor(bounds[2]))
            out[1:4:2] = (numpy.ceil(bounds[1]), numpy.ceil(bounds[3]))
            return out

    @staticmethod
    def _get_ortho_mesh(ortho_bounds):
        """
        Fetch a the grid of rows/columns coordinates for the desired rectangle.

        Parameters
        ----------
        ortho_bounds : numpy.ndarray
            Of the form `(min row, max row, min col, max col)`.

        Returns
        -------
        numpy.ndarray
        """

        ortho_shape = (int(ortho_bounds[1]-ortho_bounds[0]), int(ortho_bounds[3]-ortho_bounds[2]), 2)
        ortho_mesh = numpy.zeros(ortho_shape, dtype=numpy.int32)
        ortho_mesh[:, :, 1], ortho_mesh[:, :, 0] = numpy.meshgrid(numpy.arange(ortho_bounds[2], ortho_bounds[3]),
                                                                  numpy.arange(ortho_bounds[0], ortho_bounds[1]))
        return ortho_mesh

    @staticmethod
    def _get_mask(pixel_rows, pixel_cols, row_array, col_array):
        """
        Construct the valid mask. This is a helper function, and no error checking
        will be performed for any issues.

        Parameters
        ----------
        pixel_rows : numpy.ndarray
            The array of pixel rows. Must be the same shape as `pixel_cols`.
        pixel_cols : numpy.ndarray
            The array of pixel columns. Must be the same shape as `pixel_rows'.
        row_array : numpy.ndarray
            The rows array used for bounds, must be one dimensional and monotonic.
        col_array : numpy.ndarray
            The columns array used for bounds, must be one dimensional and monotonic.

        Returns
        -------
        numpy.ndarray
        """

        mask = (numpy.isfinite(pixel_rows) &
                numpy.isfinite(pixel_cols) &
                (pixel_rows >= row_array[0]) & (pixel_rows < row_array[-1]) &
                (pixel_cols >= col_array[0]) & (pixel_cols < col_array[-1]))
        return mask

    def bounds_to_rectangle(self, bounds):
        """
        From a bounds style array, construct the four corner coordinate array.
        This follows the SICD convention of going CLOCKWISE around the corners.

        Parameters
        ----------
        bounds : numpy.ndarray|list|tuple
            Of the form `(row min, row max, col min, col max)`.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The (integer valued) bounds and rectangular coordinates.
        """

        bounds = self.validate_bounds(bounds)
        coords = numpy.zeros((4, 2), dtype=numpy.int32)
        coords[0, :] = (bounds[0], bounds[2])  # row min, col min
        coords[1, :] = (bounds[0], bounds[3])  # row min, col max
        coords[2, :] = (bounds[1], bounds[3])  # row max, col max
        coords[3, :] = (bounds[1], bounds[2])  # row max, col min
        return bounds, coords

    def extract_pixel_bounds(self, bounds):
        """
        Validate the bounds array of orthorectified pixel coordinates, and determine
        the required bounds in reader pixel coordinates. If the

        Parameters
        ----------
        bounds : numpy.ndarray|list|tuple

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The integer valued orthorectified and reader pixel coordinate bounds.
        """

        bounds, coords = self.bounds_to_rectangle(bounds)
        filled_coords = _linear_fill(coords, fill_interval=1)
        pixel_coords = self.proj_helper.ortho_to_pixel(filled_coords)
        pixel_bounds = self.proj_helper.get_pixel_array_bounds(pixel_coords)
        return bounds, self.validate_bounds(pixel_bounds)

    def _initialize_workspace(self, ortho_bounds, final_dimension=0):
        """
        Initialize the orthorectification array workspace.

        Parameters
        ----------
        ortho_bounds : numpy.ndarray
            Of the form `(min row, max row, min col, max col)`.
        final_dimension : int
            The size of the third dimension. If `0`, then it will be omitted.

        Returns
        -------
        numpy.ndarray
        """

        if final_dimension > 0:
            out_shape = (
                int(ortho_bounds[1] - ortho_bounds[0]),
                int(ortho_bounds[3] - ortho_bounds[2]),
                int(final_dimension))
        else:
            out_shape = (
                int(ortho_bounds[1]-ortho_bounds[0]),
                int(ortho_bounds[3]-ortho_bounds[2]))
        return numpy.zeros(out_shape, dtype=self.out_dtype) if self._pad_value is None else \
            numpy.full(out_shape, self._pad_value, dtype=self.out_dtype)

    def get_real_pixel_bounds(self, pixel_bounds):
        """
        Fetch the real pixel limit from the nominal pixel limits - this just
        factors in the image reader extent.

        Parameters
        ----------
        pixel_bounds : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        if pixel_bounds[0] > pixel_bounds[1] or pixel_bounds[2] > pixel_bounds[3]:
            raise ValueError('Got unexpected and invalid pixel_bounds array {}'.format(pixel_bounds))

        pixel_limits = self.reader.get_data_size_as_tuple()[self.index]
        if (pixel_bounds[0] >= pixel_limits[0]) or (pixel_bounds[1] < 0) or \
                (pixel_bounds[2] >= pixel_limits[1]) or (pixel_bounds[3] < 0):
            # this entirely misses the whole region
            return numpy.array([0, 0, 0, 0], dtype=numpy.int32)

        real_pix_bounds = numpy.array([
            max(0, pixel_bounds[0]), min(pixel_limits[0], pixel_bounds[1]),
            max(0, pixel_bounds[2]), min(pixel_limits[1], pixel_bounds[3])], dtype=numpy.int32)
        return real_pix_bounds

    def _apply_radiometric_params(self, pixel_rows, pixel_cols, value_array):
        """
        Apply the radiometric parameters to the solution array.

        Parameters
        ----------
        pixel_rows : numpy.ndarray
        pixel_cols : numpy.ndarray
        value_array : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        if self._rad_poly is None and self._noise_poly is None:
            # nothing to be done.
            return value_array

        if pixel_rows.shape == value_array.shape and pixel_cols.shape == value_array.shape:
            rows_meters = (pixel_rows - self.sicd.ImageData.SCPPixel.Row)*self.sicd.Grid.Row.SS
            cols_meters = (pixel_cols - self.sicd.ImageData.SCPPixel.Col)*self.sicd.Grid.Col.SS
        elif value_array.ndim == 2 and \
            (pixel_rows.ndim == 1 and pixel_rows.size == value_array.shape[0]) and \
                (pixel_cols.ndim == 1 and pixel_cols.size == value_array.shape[1]):
            cols_meters, rows_meters = numpy.meshgrid(
                (pixel_cols - self.sicd.ImageData.SCPPixel.Col)*self.sicd.Grid.Col.SS,
                (pixel_rows - self.sicd.ImageData.SCPPixel.Row) * self.sicd.Grid.Row.SS)
        else:
            raise ValueError(
                'Either pixel_rows, pixel_cols, and value_array must all have the same shape, '
                'or pixel_rows/pixel_cols are one dimensional and '
                'value_array.shape = (pixel_rows.size, pixel_cols.size). Got shapes {}, {}, and '
                '{}'.format(pixel_rows.shape, pixel_cols.shape, value_array.shape))

        # calculate pixel power, with noise subtracted if necessary
        if self._noise_poly is not None:
            noise = numpy.exp(10 * self._noise_poly(rows_meters, cols_meters))  # convert from db to power
            pixel_power = value_array*value_array - noise
            del noise
        else:
            pixel_power = value_array*value_array

        if self._rad_poly is None:
            return numpy.sqrt(pixel_power)
        else:
            return pixel_power*self._rad_poly(rows_meters, cols_meters)

    def _validate_row_col_values(self, row_array, col_array, value_array, value_is_flat=False):
        """
        Helper method for validating the types and shapes.

        Parameters
        ----------
        row_array : numpy.ndarray
            The rows of the pixel array. Must be one-dimensional, monotonically
            increasing, and have `row_array.size = value_array.shape[0]`.
        col_array : numpy.ndarray
            The columns of the pixel array. Must be one-dimensional, monotonically
            increasing, and have and have `col_array.size = value_array.shape[1]`.
        value_array : numpy.ndarray
            The values array, whihc must be two or three dimensional. If this has
            complex dtype and `complex_valued=False`, then the :func:`numpy.abs`
            will be applied.
        value_is_flat : bool
            If `True`, then `value_array` must be exactly two-dimensional.

        Returns
        -------
        numpy.ndarray
        """

        # verify numpy arrays
        if not isinstance(value_array, numpy.ndarray):
            raise TypeError('value_array must be numpy.ndarray, got type {}'.format(type(value_array)))
        if not isinstance(row_array, numpy.ndarray):
            raise TypeError('row_array must be numpy.ndarray, got type {}'.format(type(row_array)))
        if not isinstance(col_array, numpy.ndarray):
            raise TypeError('col_array must be numpy.ndarray, got type {}'.format(type(col_array)))

        # verify array shapes make sense
        if value_array.ndim not in (2, 3):
            raise ValueError('value_array must be two or three dimensional')
        if row_array.ndim != 1 or row_array.size != value_array.shape[0]:
            raise ValueError(
                'We must have row_array is one dimensional and row_array.size = value.array.shape[0]. '
                'Got row_array.shape = {}, and value_array = {}'.format(row_array.shape, value_array.shape))
        if col_array.ndim != 1 or col_array.size != value_array.shape[1]:
            raise ValueError(
                'We must have col_array is one dimensional and col_array.size = value.array.shape[1]. '
                'Got col_array.shape = {}, and value_array = {}'.format(col_array.shape, value_array.shape))
        if value_is_flat and len(value_array.shape) != 2:
            raise ValueError('value_array must be two-dimensional. Got shape {}'.format(value_array.shape))

        # verify row_array/col_array are monotonically increasing
        if numpy.any(numpy.diff(row_array.astype('float64')) <= 0):
            raise ValueError('row_array must be monotonically increasing.')
        if numpy.any(numpy.diff(col_array.astype('float64')) <= 0):
            raise ValueError('col_array must be monotonically increasing.')

        # address the dtype of value_array
        if (not self._complex_valued) and numpy.iscomplexobj(value_array):
            return numpy.abs(value_array)

        return value_array

    def get_orthorectified_from_array(self, ortho_bounds, row_array, col_array, value_array):
        """
        Construct the orthorecitified array covering the orthorectified region given by
        `ortho_bounds` based on the `values_array`, which spans the pixel region defined by
        `row_array` and `col_array`.

        This is mainly a helper method, and should only be called directly for specific and
        directed reasons.

        Parameters
        ----------
        ortho_bounds : numpy.ndarray
            Determines the orthorectified bounds region, of the form
            `(min row, max row, min column, max column)`.
        row_array : numpy.ndarray
            The rows of the pixel array. Must be one-dimensional, monotonically increasing,
            and have `row_array.size = value_array.shape[0]`.
        col_array
            The columns of the pixel array. Must be one-dimensional, monotonically increasing,
            and have `col_array.size = value_array.shape[1]`.
        value_array
            The values array. If this has complex dtype and `complex_valued=False`, then
            the :func:`numpy.abs` will be applied.

        Returns
        -------
        numpy.ndarray
        """

        # validate our inputs
        value_array = self._validate_row_col_values(row_array, col_array, value_array, value_is_flat=False)
        if value_array.size == 0:
            if value_array.ndim == 3:
                return self._initialize_workspace(ortho_bounds, final_dimension=value_array.shape[2])
            else:
                return self._initialize_workspace(ortho_bounds)
        if value_array.ndim == 2:
            return self._get_orthrectified_from_array_flat(ortho_bounds, row_array, col_array, value_array)
        else:  # it must be three dimensional, as checked by _validate_row_col_values()
            ortho_array = self._initialize_workspace(ortho_bounds, final_dimension=value_array.shape[2])
            for i in range(value_array.shape[2]):
                ortho_array[:, :, i] = self._get_orthrectified_from_array_flat(
                    ortho_bounds, row_array, col_array, value_array[:, :, i])
            return ortho_array

    def get_orthorectified_for_ortho_bounds(self, bounds):
        """
        Determine the array corresponding to the array of bounds given in
        ortho-rectified pixel coordinates.

        Parameters
        ----------
        bounds : numpy.ndarray|list|tuple
            Of the form `(row_min, row_max, col_min, col_max)`. Note that non-integer
            values will be expanded outwards (floor of minimum and ceil at maximum).
            Following Python convention, this will be inclusive at the minimum and
            exclusive at the maximum.

        Returns
        -------
        numpy.ndarray
        """

        ortho_bounds, nominal_pixel_bounds = self.extract_pixel_bounds(bounds)
        # extract the values - ensure that things are within proper image bounds
        pixel_bounds = self.get_real_pixel_bounds(nominal_pixel_bounds)
        pixel_array = self.reader[
            pixel_bounds[0]:pixel_bounds[1], pixel_bounds[2]:pixel_bounds[3], self.index]
        row_arr = numpy.arange(pixel_bounds[0], pixel_bounds[1])
        col_arr = numpy.arange(pixel_bounds[2], pixel_bounds[3])
        return self.get_orthorectified_from_array(ortho_bounds, row_arr, col_arr, pixel_array)

    def get_orthorectified_for_pixel_bounds(self, pixel_bounds):
        """
        Determine the array corresponding to the given array bounds given in reader
        pixel coordinates.

        Parameters
        ----------
        pixel_bounds : numpy.ndarray|list|tuple
            Of the form `(row_min, row_max, col_min, col_max)`.

        Returns
        -------
        numpy.ndarray
        """

        pixel_bounds, pixel_rect = self.bounds_to_rectangle(pixel_bounds)
        return self.get_orthorectified_for_pixel_object(pixel_rect)

    def get_orthorectified_for_pixel_object(self, coordinates):
        """
        Determine the ortho-rectified rectangular array values, which will bound
        the given object - with coordinates expressed in pixel space.

        Parameters
        ----------
        coordinates : GeometryObject|numpy.ndarray|list|tuple
            The coordinate system of the input will be assumed to be pixel space.

        Returns
        -------
        numpy.ndarray
        """

        bounds = self.get_orthorectification_bounds_from_pixel_object(coordinates)
        return self.get_orthorectified_for_ortho_bounds(bounds)

    def get_orthorectified_for_latlon_object(self, ll_coordinates):
        """
        Determine the ortho-rectified rectangular array values, which will bound
        the given object - with coordinates expressed in lat/lon space.

        Parameters
        ----------
        ll_coordinates : GeometryObject|numpy.ndarray|list|tuple
            The coordinate system of the input will be assumed to be pixel space.
            **Note** a GeometryObject is expected to follow lon/lat ordering paradigm,
            by convention.

        Returns
        -------
        numpy.ndarray
        """

        bounds = self.get_orthorectification_bounds_from_latlon_object(ll_coordinates)
        return self.get_orthorectified_for_ortho_bounds(bounds)

    def _setup_flat_workspace(self, ortho_bounds, row_array, col_array, value_array):
        """
        Helper method for setting up the flat workspace.

        Parameters
        ----------
        ortho_bounds : numpy.ndarray
            Determines the orthorectified bounds region, of the form
            `(min row, max row, min column, max column)`.
        row_array : numpy.ndarray
            The rows of the pixel array. Must be one-dimensional, monotonically
            increasing, and have `row_array.size = value_array.shape[0]`.
        col_array : numpy.ndarray
            The columns of the pixel array. Must be one-dimensional, monotonically
            increasing, and have `col_array.size = value_array.shape[1]`.
        value_array : numpy.ndarray
            The values array. If this has complex dtype and `complex_valued=False`,
            then the :func:`numpy.abs` will be applied.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """

        value_array = self._validate_row_col_values(row_array, col_array, value_array, value_is_flat=True)
        # set up the results workspace
        ortho_array = self._initialize_workspace(ortho_bounds)
        # determine the pixel coordinates for the ortho coordinates meshgrid
        ortho_mesh = self._get_ortho_mesh(ortho_bounds)
        # determine the nearest neighbor pixel coordinates
        pixel_mesh = self.proj_helper.ortho_to_pixel(ortho_mesh)
        pixel_rows = pixel_mesh[:, :, 0]
        pixel_cols = pixel_mesh[:, :, 1]
        return value_array, pixel_rows, pixel_cols, ortho_array

    def _get_orthrectified_from_array_flat(self, ortho_bounds, row_array, col_array, value_array):
        """
        Construct the orthorecitified array covering the orthorectified region given by
        `ortho_bounds` based on the `values_array`, which spans the pixel region defined by
        `row_array` and `col_array`.

        Parameters
        ----------
        ortho_bounds : numpy.ndarray
            Determines the orthorectified bounds region, of the form
            `(min row, max row, min column, max column)`.
        row_array : numpy.ndarray
            The rows of the pixel array. Must be one-dimensional, monotonically
            increasing, and have `row_array.size = value_array.shape[0]`.
        col_array
            The columns of the pixel array. Must be one-dimensional, monotonically
            increasing, and have `col_array.size = value_array.shape[1]`.
        value_array
            The values array. If this has complex dtype and `complex_valued=False`,
            then the :func:`numpy.abs` will be applied.

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError


class NearestNeighborMethod(OrthorectificationHelper):
    """
    Nearest neighbor ortho-rectification method.

    .. warning::
        Modification of the proj_helper parameters when the default full image
        bounds have been defained (i.e. sicd.RadarCollection.Area is defined) may
        result in unintended results.
    """

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False,
                 pad_value=None, apply_radiometric=None, subtract_radiometric_noise=False):
        """

        Parameters
        ----------
        reader : SICDTypeReader
        index : int
        proj_helper : None|ProjectionHelper
            If `None`, this will default to `PGProjection(<sicd>)`, where `<sicd>`
            will be the sicd from `reader` at `index`. Otherwise, it is the user's
            responsibility to ensure that `reader`, `index` and `proj_helper` are
            in sync.
        complex_valued : bool
            Do we want complex values returned? If `False`, the magnitude values
            will be used.
        pad_value : None|Any
            Value to use for any out-of-range pixels. Defaults to `0` if not provided.
        apply_radiometric : None|str
            **Only valid if `complex_valued=False`**. If provided, must be one of
            `['RCS', 'Sigma0', 'Gamma0', 'Beta0']` (not case-sensitive). This will
            apply the given radiometric scale factor to the array values.
        subtract_radiometric_noise : bool
            **Only has any effect if `apply_radiometric` is provided.** This indicates that
            the radiometric noise should be subtracted prior to applying the given
            radiometric scale factor.
        """

        super(NearestNeighborMethod, self).__init__(
            reader, index=index, proj_helper=proj_helper, complex_valued=complex_valued,
            pad_value=pad_value, apply_radiometric=apply_radiometric,
            subtract_radiometric_noise=subtract_radiometric_noise)

    def _get_orthrectified_from_array_flat(self, ortho_bounds, row_array, col_array, value_array):
        # setup the result workspace
        value_array, pixel_rows, pixel_cols, ortho_array = self._setup_flat_workspace(
            ortho_bounds, row_array, col_array, value_array)
        # potentially apply the radiometric parameters to the value array
        value_array = self._apply_radiometric_params(row_array, col_array, value_array)
        if value_array.size > 0:
            # determine the in bounds points
            mask = self._get_mask(pixel_rows, pixel_cols, row_array, col_array)
            # determine the nearest neighbors for our row/column indices
            row_inds = numpy.digitize(pixel_rows[mask], row_array)
            col_inds = numpy.digitize(pixel_cols[mask], col_array)
            ortho_array[mask] = value_array[row_inds, col_inds]
        return ortho_array


class BivariateSplineMethod(OrthorectificationHelper):
    """
    Bivariate spline interpolation ortho-rectification method.

    .. warning::
        Modification of the proj_helper parameters when the default full image
        bounds have been defained (i.e. sicd.RadarCollection.Area is defined) may
        result in unintended results.
    """

    __slots__ = ('_row_order', '_col_order')

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False,
                 pad_value=None, apply_radiometric=None, subtract_radiometric_noise=False,
                 row_order=1, col_order=1):
        """

        Parameters
        ----------
        reader : SICDTypeReader
        index : int
        proj_helper : None|ProjectionHelper
            If `None`, this will default to `PGProjection(<sicd>)`, where `<sicd>`
            will be the sicd from `reader` at `index`. Otherwise, it is the user's
            responsibility to ensure that `reader`, `index` and `proj_helper` are
            in sync.
        complex_valued : bool
            Do we want complex values returned? If `False`, the magnitude values
            will be used.
        pad_value : None|Any
            Value to use for any out-of-range pixels. Defaults to `0` if not provided.
        apply_radiometric : None|str
            **Only valid if `complex_valued=False`**. If provided, must be one of
            `['RCS', 'Sigma0', 'Gamma0', 'Beta0']` (not case-sensitive). This will
            apply the given radiometric scale factor to the array values.
        subtract_radiometric_noise : bool
            **Only has any effect if `apply_radiometric` is provided.** This indicates that
            the radiometric noise should be subtracted prior to applying the given
            radiometric scale factor.
        row_order : int
            The row degree for the spline.
        col_order : int
            The column degree for the spline.
        """

        self._row_order = None
        self._col_order = None
        if complex_valued:
            raise ValueError('BivariateSpline only supports real valued results for now.')
        super(BivariateSplineMethod, self).__init__(
            reader, index=index, proj_helper=proj_helper, complex_valued=complex_valued,
            pad_value=pad_value, apply_radiometric=apply_radiometric,
            subtract_radiometric_noise=subtract_radiometric_noise)
        self.row_order = row_order
        self.col_order = col_order

    @property
    def row_order(self):
        """
        int : The spline order for the x/row coordinate, where `1 <= row_order <= 5`.
        """
        return self._row_order

    @row_order.setter
    def row_order(self, value):
        value = int(value)
        if not (1 <= value <= 5):
            raise ValueError('row_order must take value between 1 and 5.')
        self._row_order = value

    @property
    def col_order(self):
        """
        int : The spline order for the y/col coordinate, where `1 <= col_order <= 5`.
        """

        return self._col_order

    @col_order.setter
    def col_order(self, value):
        value = int(value)
        if not (1 <= value <= 5):
            raise ValueError('col_order must take value between 1 and 5.')
        self._col_order = value

    def _get_orthrectified_from_array_flat(self, ortho_bounds, row_array, col_array, value_array):
        # setup the result workspace
        value_array, pixel_rows, pixel_cols, ortho_array = self._setup_flat_workspace(
            ortho_bounds, row_array, col_array, value_array)
        value_array = self._apply_radiometric_params(row_array, col_array, value_array)

        if value_array.size > 0:
            # set up our spline
            sp = RectBivariateSpline(row_array, col_array, value_array, kx=self.row_order, ky=self.col_order, s=0)
            # determine the in bounds points
            mask = self._get_mask(pixel_rows, pixel_cols, row_array, col_array)
            result = sp.ev(pixel_rows[mask], pixel_cols[mask])
            # potentially apply the radiometric parameters
            ortho_array[mask] = result
        return ortho_array


#################
# Ortho-rectification generator/iterator

class FullResolutionFetcher(object):
    """
    This is a base class for provided a simple API for processing schemes where
    full resolution is required along the processing dimension, so sub-sampling
    along the processing dimension does not decrease the amount of data which
    must be fetched.
    """

    __slots__ = (
        '_reader', '_index', '_sicd', '_dimension', '_data_size', '_block_size')

    def __init__(self, reader, dimension=0, index=0, block_size=10):
        """

        Parameters
        ----------
        reader : str|SICDTypeReader
            Input file path or reader object, which must be of sicd type.
        dimension : int
            The dimension over which to split the sub-aperture.
        index : int
            The sicd index to use.
        block_size : None|int|float
            The approximate processing block size to fetch, given in MB. The
            minimum value for use here will be 0.25. `None` represents processing
            as a single block.
        """

        self._index = None # set explicitly
        self._sicd = None  # set with index setter
        self._dimension = None # set explicitly
        self._data_size = None  # set with index setter
        self._block_size = None # set explicitly

        # validate the reader
        if isinstance(reader, str):
            reader = open_complex(reader)
        if not isinstance(reader, SICDTypeReader):
            raise TypeError('reader is required to be a path name for a sicd-type image, '
                            'or an instance of a reader object.')
        self._reader = reader
        # set the other properties
        self.dimension = dimension
        self.index = index
        self.block_size = block_size

    @property
    def reader(self):
        # type: () -> SICDTypeReader
        """
        SICDTypeReader: The reader instance.
        """

        return self._reader

    @property
    def dimension(self):
        # type: () -> int
        """
        int: The dimension along which to perform the color subaperture split.
        """

        return self._dimension

    @dimension.setter
    def dimension(self, value):
        value = int(value)
        if value not in [0, 1]:
            raise ValueError('dimension must be 0 or 1, got {}'.format(value))
        self._dimension = value

    @property
    def data_size(self):
        # type: () -> Tuple[int, int]
        """
        Tuple[int, int]: The data size for the reader at the given index.
        """

        return self._data_size

    @property
    def index(self):
        # type: () -> int
        """
        int: The index of the reader.
        """

        return self._index

    @index.setter
    def index(self, value):
        self._set_index(value)

    def _set_index(self, value):
        value = int(value)
        if value < 0:
            raise ValueError('The index must be a non-negative integer, got {}'.format(value))

        sicds = self.reader.get_sicds_as_tuple()
        if value >= len(sicds):
            raise ValueError('The index must be less than the sicd count.')
        self._index = value
        self._sicd = sicds[value]
        self._data_size = self.reader.get_data_size_as_tuple()[value]

    @property
    def block_size(self):
        # type: () -> float
        """
        None|float: The approximate processing block size in MB, where `None`
        represents processing in a single block.
        """

        return self._block_size

    @block_size.setter
    def block_size(self, value):
        if value is None:
            self._block_size = None
        else:
            value = float(value)
            if value < 0.25:
                value = 0.25
            self._block_size = value

    @property
    def block_size_in_bytes(self):
        # type: () -> Union[None, int]
        """
        None|int: The approximate processing block size in bytes.
        """

        return None if self._block_size is None else int(self._block_size*(2**20))

    @property
    def sicd(self):
        # type: () -> SICDType
        """
        SICDType: The sicd structure.
        """

        return self._sicd

    def _parse_slicing(self, item):
        # type: (Union[None, int, slice, tuple]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Any]

        def parse(entry, dimension):
            bound = self.data_size[dimension]
            if entry is None:
                return 0, bound, 1
            elif isinstance(entry, int):
                entry = validate_slice_int(entry, bound)
                return entry, entry+1, 1
            elif isinstance(entry, slice):
                entry = validate_slice(entry, bound)
                return entry.start, entry.stop, entry.step
            else:
                raise TypeError('No support for slicing using type {}'.format(type(entry)))

        # this input is assumed to come from slice parsing
        if isinstance(item, tuple):
            if len(item) > 3:
                raise ValueError(
                    'Received slice argument {}. We cannot slice '
                    'on more than two dimensions.'.format(item))
            elif len(item) == 3:
                return parse(item[0], 0), parse(item[1], 1), item[2]
            elif len(item) == 2:
                return parse(item[0], 0), parse(item[1], 1), None
            elif len(item) == 1:
                return parse(item[0], 0), parse(None, 1), None
            else:
                return parse(None, 0), parse(None, 1), None
        elif isinstance(item, slice):
            return parse(item, 0), parse(None, 1), None
        elif isinstance(item, int):
            return parse(item, 0), parse(None, 1), None
        else:
            raise TypeError('Slicing using type {} is unsupported'.format(type(item)))

    def get_fetch_block_size(self, start_element, stop_element):
        """
        Gets the fetch block size for the given full resolution section.
        This assumes that the fetched data will be 8 bytes per pixel, in
        accordance with single band complex64 data.

        Parameters
        ----------
        start_element : int
        stop_element : int

        Returns
        -------
        int
        """

        return get_fetch_block_size(start_element, stop_element, self.block_size_in_bytes, bands=1)

    @staticmethod
    def extract_blocks(the_range, index_block_size):
        # type: (Tuple[int, int, int], Union[None, int, float]) -> (List[Tuple[int, int, int]], List[Tuple[int, int]])
        """
        Convert the single range definition into a series of range definitions in
        keeping with fetching of the appropriate block sizes.

        Parameters
        ----------
        the_range : Tuple[int, int, int]
            The input (off processing axis) range.
        index_block_size : None|int|float
            The size of blocks (number of indices).

        Returns
        -------
        List[Tuple[int, int, int]], List[Tuple[int, int]]
            The sequence of range definitions `(start index, stop index, step)`
            relative to the overall image, and the sequence of start/stop indices
            for positioning of the given range relative to the original range.

        """

        return extract_blocks(the_range, index_block_size)

    def _full_row_resolution(self, row_range, col_range):
        # type: (Tuple[int, int, int], Tuple[int, int, int]) -> numpy.ndarray
        """
        Perform the full row resolution data, with any appropriate calculations.

        Parameters
        ----------
        row_range : Tuple[int, int, int]
        col_range : Tuple[int, int, int]

        Returns
        -------
        numpy.ndarray
        """

        # fetch the data and perform the csi calculation
        if row_range[2] not in [1, -1]:
            raise ValueError('The step for row_range must be +/- 1, for full row resolution data.')
        if row_range[1] == -1:
            data = self.reader[
                   row_range[0]::row_range[2],
                   col_range[0]:col_range[1]:col_range[2],
                   self.index]
        else:
            data = self.reader[
                   row_range[0]:row_range[1]:row_range[2],
                   col_range[0]:col_range[1]:col_range[2],
                   self.index]

        if data.ndim < 2:
            data = numpy.reshape(data, (-1, 1))
        # handle nonsense data with zeros
        data[~numpy.isfinite(data)] = 0
        return data

    def _full_column_resolution(self, row_range, col_range):
        # type: (Tuple[int, int, int], Tuple[int, int, int]) -> numpy.ndarray
        """
        Perform the full column resolution data, with any appropriate calculations.

        Parameters
        ----------
        row_range : Tuple[int, int, int]
        col_range : Tuple[int, int, int]

        Returns
        -------
        numpy.ndarray
        """

        # fetch the data and perform the csi calculation
        if col_range[2] not in [1, -1]:
            raise ValueError('The step for col_range must be +/- 1, for full col resolution data.')
        if col_range[1] == -1:
            data = self.reader[
                   row_range[0]:row_range[1]:row_range[2],
                   col_range[0]::col_range[2],
                   self.index]
        else:
            data = self.reader[
                   row_range[0]:row_range[1]:row_range[2],
                   col_range[0]:col_range[1]:col_range[2],
                   self.index]

        if data.ndim < 2:
            data = numpy.reshape(data, (1, -1))
        # handle nonsense data with zeros
        data[~numpy.isfinite(data)] = 0
        return data

    def _prepare_output(self, row_range, col_range):
        """
        Prepare the output workspace for :func:`__getitem__`.

        Parameters
        ----------
        row_range
        col_range

        Returns
        -------
        numpy.ndarray
        """

        row_count = int((row_range[1] - row_range[0]) / float(row_range[2]))
        col_count = int((col_range[1] - col_range[0]) / float(col_range[2]))
        out_size = (row_count, col_count)
        return numpy.zeros(out_size, dtype=numpy.complex64)

    def __getitem__(self, item):
        """
        Fetches the processed data based on the input slice.

        Parameters
        ----------
        item

        Returns
        -------
        numpy.ndarray
        """

        # parse the slicing to ensure consistent structure
        row_range, col_range, _ = self._parse_slicing(item)
        return self.reader[
               row_range[0]:row_range[1]:row_range[2],
               col_range[0]:col_range[1]:col_range[2],
               self.index]


class OrthorectificationIterator(object):
    """
    This provides a generator for an Orthorectification process on a given
    reader/index/(pixel) bounds.
    """

    __slots__ = (
        '_calculator', '_ortho_helper', '_pixel_bounds', '_ortho_bounds',
        '_this_index', '_iteration_blocks', '_remap_function')

    def __init__(
            self, ortho_helper, calculator=None, bounds=None,
            remap_function=None, recalc_remap_globals=False):
        """

        Parameters
        ----------
        ortho_helper : OrthorectificationHelper
            The ortho-rectification helper.
        calculator : None|FullResolutionFetcher
            The FullResolutionFetcher instance. If not provided, then this will
            default to a base FullResolutionFetcher instance - which is only
            useful for a basic detected image.
        bounds : None|numpy.ndarray|list|tuple
            The pixel bounds of the form `(min row, max row, min col, max col)`.
            This will default to the full image.
        remap_function : None|RemapFunction
            The remap function to apply, if desired.
        recalc_remap_globals : bool
            Only applies if a remap function is provided, should we recalculate
            any required global parameters? This will automatically happen if
            they are not already set.
        """

        self._this_index = None
        self._iteration_blocks = None
        self._remap_function = None

        # validate ortho_helper
        if not isinstance(ortho_helper, OrthorectificationHelper):
            raise TypeError(
                'ortho_helper must be an instance of OrthorectificationHelper, got '
                'type {}'.format(type(ortho_helper)))
        self._ortho_helper = ortho_helper

        # validate calculator
        if calculator is None:
            calculator = FullResolutionFetcher(ortho_helper.reader, index=ortho_helper.index, dimension=0)
        if not isinstance(calculator, FullResolutionFetcher):
            raise TypeError(
                'calculator must be an instance of FullResolutionFetcher, got '
                'type {}'.format(type(calculator)))
        self._calculator = calculator

        if os.path.abspath(ortho_helper.reader.file_name) != \
                os.path.abspath(calculator.reader.file_name):
            raise ValueError(
                'ortho_helper has reader for file {}, while calculator has reader '
                'for file {}'.format(ortho_helper.reader.file_name, calculator.reader.file_name))
        if ortho_helper.index != calculator.index:
            raise ValueError(
                'ortho_helper is using index {}, while calculator is using '
                'index {}'.format(ortho_helper.index, calculator.index))

        # validate the bounds
        if bounds is not None:
            pixel_bounds, pixel_rectangle = ortho_helper.bounds_to_rectangle(bounds)
            # get the corresponding ortho bounds
            ortho_bounds = ortho_helper.get_orthorectification_bounds_from_pixel_object(pixel_rectangle)
        else:
            ortho_bounds = ortho_helper.get_full_ortho_bounds()
            ortho_bounds, nominal_pixel_bounds = ortho_helper.extract_pixel_bounds(ortho_bounds)
            # extract the values - ensure that things are within proper image bounds
            pixel_bounds = ortho_helper.get_real_pixel_bounds(nominal_pixel_bounds)

        # validate remap function
        if remap_function is None or isinstance(remap_function, RemapFunction):
            self._remap_function = remap_function
        else:
            raise TypeError(
                'remap_function is expected to be an instance of RemapFunction, '
                'got type `{}`'.format(type(remap_function)))

        self._pixel_bounds = pixel_bounds
        self._ortho_bounds = ortho_bounds
        self._prepare_state(recalc_remap_globals=recalc_remap_globals)

    @property
    def ortho_helper(self):
        # type: () -> OrthorectificationHelper
        """
        OrthorectificationHelper: The ortho-rectification helper.
        """

        return self._ortho_helper

    @property
    def calculator(self):
        # type: () -> FullResolutionFetcher
        """
        FullResolutionFetcher : The calculator instance.
        """

        return self._calculator

    @property
    def sicd(self):
        """
        SICDType: The sicd structure.
        """

        return self.calculator.sicd

    @property
    def pixel_bounds(self):
        """
        numpy.ndarray : Of the form `(row min, row max, col min, col max)`.
        """

        return self._pixel_bounds

    @property
    def ortho_bounds(self):
        """
        numpy.ndarray : Of the form `(row min, row max, col min, col max)`. Note that
        these are "unnormalized" orthorectified pixel coordinates.
        """

        return self._ortho_bounds

    @property
    def ortho_data_size(self):
        """
        Tuple[int, int] : The size of the overall ortho-rectified output.
        """

        return (
            int(self.ortho_bounds[1] - self.ortho_bounds[0]),
            int(self.ortho_bounds[3] - self.ortho_bounds[2]))

    @property
    def remap_function(self):
        # type: () -> Union[None, RemapFunction]
        """
        None|RemapFunction: The remap function to be applied.
        """

        return self._remap_function

    def get_ecf_image_corners(self):
        """
        The corner points of the overall ortho-rectified output in ECF
        coordinates. The ordering of these points follows the SICD convention.

        Returns
        -------
        numpy.ndarray
        """

        if self.ortho_bounds is None:
            return None
        _, ortho_pixel_corners = self._ortho_helper.bounds_to_rectangle(self.ortho_bounds)
        return self._ortho_helper.proj_helper.ortho_to_ecf(ortho_pixel_corners)

    def get_llh_image_corners(self):
        """
        The corner points of the overall ortho-rectified output in Lat/Lon/HAE
        coordinates. The ordering of these points follows the SICD convention.

        Returns
        -------
        numpy.ndarray
        """

        ecf_corners = self.get_ecf_image_corners()
        if ecf_corners is None:
            return None
        else:
            return ecf_to_geodetic(ecf_corners)

    def _prepare_state(self, recalc_remap_globals=False):
        """
        Prepare the iteration state.

        Returns
        -------
        None
        """

        if self.calculator.dimension == 0:
            column_block_size = self.calculator.get_fetch_block_size(self.ortho_bounds[0], self.ortho_bounds[1])
            self._iteration_blocks, _ = self.calculator.extract_blocks(
                (self.ortho_bounds[2], self.ortho_bounds[3], 1), column_block_size)
        else:
            row_block_size = self.calculator.get_fetch_block_size(self.ortho_bounds[2], self.ortho_bounds[3])
            self._iteration_blocks, _ = self.calculator.extract_blocks(
                (self.ortho_bounds[0], self.ortho_bounds[1], 1), row_block_size)

        if self.remap_function is not None and \
                (recalc_remap_globals or not self.remap_function.are_global_parameters_set):
            self.remap_function.calculate_global_parameters_from_reader(
                self.ortho_helper.reader, index=self.ortho_helper.index, pixel_bounds=self.pixel_bounds)

    @staticmethod
    def _get_ortho_helper(pixel_bounds, this_data):
        """
        Get helper data for ortho-rectification.

        Parameters
        ----------
        pixel_bounds
        this_data

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
        """

        rows_temp = pixel_bounds[1] - pixel_bounds[0]
        if this_data.shape[0] == rows_temp:
            row_array = numpy.arange(pixel_bounds[0], pixel_bounds[1])
        elif this_data.shape[0] == (rows_temp - 1):
            row_array = numpy.arange(pixel_bounds[0], pixel_bounds[1] - 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(this_data.shape, rows_temp))

        cols_temp = pixel_bounds[3] - pixel_bounds[2]
        if this_data.shape[1] == cols_temp:
            col_array = numpy.arange(pixel_bounds[2], pixel_bounds[3])
        elif this_data.shape[1] == (cols_temp - 1):
            col_array = numpy.arange(pixel_bounds[2], pixel_bounds[3] - 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(this_data.shape, cols_temp))
        return row_array, col_array

    def _get_orthorectified_version(self, this_ortho_bounds, pixel_bounds, this_data):
        """
        Get the orthorectified version from the raw values and pixel information.

        Parameters
        ----------
        this_ortho_bounds
        pixel_bounds
        this_data

        Returns
        -------
        numpy.ndarray
        """

        row_array, col_array = self._get_ortho_helper(pixel_bounds, this_data)
        ortho_data = self._ortho_helper.get_orthorectified_from_array(this_ortho_bounds, row_array, col_array, this_data)
        if self.remap_function is None:
            return ortho_data
        else:
            return self.remap_function(ortho_data)

    def _get_state_parameters(self):
        """
        Gets the pixel information associated with the current state.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
        """

        if self._calculator.dimension == 0:
            this_column_range = self._iteration_blocks[self._this_index]
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = self._ortho_helper.extract_pixel_bounds(
                (self.ortho_bounds[0], self.ortho_bounds[1], this_column_range[0], this_column_range[1]))
        else:
            this_row_range = self._iteration_blocks[self._this_index]
            this_ortho_bounds, this_pixel_bounds = self._ortho_helper.extract_pixel_bounds(
                (this_row_range[0], this_row_range[1], self.ortho_bounds[2], self.ortho_bounds[3]))
        return this_ortho_bounds, this_pixel_bounds

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next iteration of orthorectified data.

        Returns
        -------
        (numpy.ndarray, Tuple[int, int])
            The data and the (normalized) indices (start_row, start_col) for this section of data, relative
            to overall output shape.
        """

        # NB: this is the Python 3 pattern for iteration
        if self._this_index is None:
            self._this_index = 0
        else:
            self._this_index += 1
        # at this point, _this_index indicates which entry to return
        if self._this_index >= len(self._iteration_blocks):
            self._this_index = None  # reset the iteration scheme
            raise StopIteration()

        this_ortho_bounds, this_pixel_bounds = self._get_state_parameters()
        # accommodate for real pixel limits
        this_pixel_bounds = self._ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
        # extract the csi data and ortho-rectify
        logger.info(
            'Fetching orthorectified coordinate block ({}:{}, {}:{}) of ({}, {})'.format(
                this_ortho_bounds[0] - self.ortho_bounds[0], this_ortho_bounds[1] - self.ortho_bounds[0],
                this_ortho_bounds[2] - self.ortho_bounds[2], this_ortho_bounds[3] - self.ortho_bounds[2],
                self.ortho_bounds[1] - self.ortho_bounds[0], self.ortho_bounds[3] - self.ortho_bounds[2]))
        ortho_data = self._get_orthorectified_version(
            this_ortho_bounds, this_pixel_bounds,
            self._calculator[this_pixel_bounds[0]:this_pixel_bounds[1], this_pixel_bounds[2]:this_pixel_bounds[3]])
        # determine the relative image size
        start_indices = (this_ortho_bounds[0] - self.ortho_bounds[0],
                         this_ortho_bounds[2] - self.ortho_bounds[2])
        return ortho_data, start_indices

    def next(self):
        """
        Get the next iteration of ortho-rectified data.

        Returns
        -------
        numpy.ndarray, Tuple[int, int]
            The data and the (normalized) indices (start_row, start_col) for this section of data, relative
            to overall output shape.
        """

        # NB: this is the Python 2 pattern for iteration
        return self.__next__()
