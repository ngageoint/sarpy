# -*- coding: utf-8 -*-
"""
SICD ortho-rectification methodology.
"""

import numpy
from scipy.interpolate import RectBivariateSpline

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.base import BaseReader
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic
from sarpy.geometry.geometry_elements import GeometryObject

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

#################
# The projection methodology

class ProjectionHelper(object):
    """
    Abstract helper class which defines the projection interface for
    ortho-rectification usage for a sicd type object.
    """

    __slots__ = ('_sicd', '_row_spacing', '_col_spacing')

    def __init__(self, sicd, row_spacing=None, col_spacing=None):
        """

        Parameters
        ----------
        sicd : SICDType
            The sicd object
        row_spacing : None|float
            The row pixel spacing.
        col_spacing : None|float
            The column pixel spacing.
        """

        self._row_spacing = None
        self._col_spacing = None
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
        Set the row pixel spacing value. Will default to sicd.Grid.Row.SS.

        Parameters
        ----------
        value : None|float

        Returns
        -------
        None
        """

        if value is None:
            self._row_spacing = self.sicd.Grid.Row.SS
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
        Set the col pixel spacing value. Will default to sicd.Grid.Col.SS.

        Parameters
        ----------
        value : None|float

        Returns
        -------
        None
        """

        if value is None:
            self._col_spacing = self.sicd.Grid.Col.SS
        else:
            value = float(value)
            if value <= 0:
                raise ValueError('column pixel spacing must be positive.')
            self._col_spacing = float(value)

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

        if array.ndim != final_dimension:
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
        offsets
            numpy.ndarray
        """

        raise NotImplementedError

    def ll_to_ortho(self, ll_coords):
        """
        Gets the `(ortho_row, ortho_column)` coordinates in the ortho-rectified
        system for the provided physical coordinates in `(Lat, Lon)` coordinates.

        Note that the handling of ambiguity when handling the missing elevation
        is likely methodology dependent.

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
        system for the providednphysical coordinates in `(Lat, Lon)` or `(Lat, Lon, HAE)`
        coordinates.

        Note that the handling of ambiguity when provided coordinates are in `(Lat, Lon)`
        form is likely methodology dependent.

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
        Extract bounds of the input array, expected to have final dimension
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
            (numpy.min(coords[:, 0], axis=0),
             numpy.max(coords[:, 0], axis=0),
             numpy.min(coords[:, 1], axis=0),
             numpy.max(coords[:, 1], axis=0)), dtype=numpy.float64)


class PGProjection(ProjectionHelper):
    """
    Class which helps perform the Planar Grid (i.e. Ground Plane) ortho-rectification
    for a sicd-type object. **In this implementation, we have that the reference point
    will have ortho-rectification coordinates (0, 0).** All ortho-rectification coordinate
    interpretation should be relative to the fact.
    """

    __slots__ = (
        '_reference_point', '_row_vector', '_col_vector')

    def __init__(self, sicd, reference_point=None, row_vector=None, col_vector=None,
                 row_spacing=None, col_spacing=None):
        """

        Parameters
        ----------
        sicd : SICDType
            The sicd object
        reference_point : None|numpy.ndarray
            The reference point (origin) of the planar grid. If None, the sicd.GeoData.SCP
            will be used.
        row_vector : None|numpy.ndarray
            The vector defining increasing column direction. If None, then the
            sicd.Grid.Row.UVectECF will be used.
        col_vector : None|numpy.ndarray
            The vector defining increasing column direction. It is required
            that `row_vector` and `col_vector` are orthogonal. If None, then the
            perpendicular component of sicd.Grid.Col.UVectECF will be used.
        row_spacing : None|float
            The row pixel spacing.
        col_spacing : None|float
            The column pixel spacing.
        """

        self._reference_point = None
        self._row_vector = None
        self._col_vector = None
        super(PGProjection, self).__init__(sicd, row_spacing=row_spacing, col_spacing=col_spacing)
        self.set_reference_point(reference_point)
        self.set_row_and_col_vector(row_vector, col_vector)

    @property
    def reference_point(self):
        """
        numpy.ndarray: The grid reference point.
        """

        return self._reference_point

    def set_reference_point(self, reference_point):
        """
        Sets the reference point.

        Parameters
        ----------
        reference_point : None|numpy.ndarray
            The reference point (origin) of the planar grid. If None, the sicd.GeoData.SCP
            will be used.

        Returns
        -------
        None
        """

        if reference_point is None:
            reference_point = self.sicd.GeoData.SCP.ECF.get_array()

        if not (isinstance(reference_point, numpy.ndarray) and reference_point.ndim == 1
                and reference_point.size == 3):
            raise ValueError('reference_point must be a vector of size 3.')
        self._reference_point = reference_point

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

    def set_row_and_col_vector(self, row_vector, col_vector):
        """
        Set the row and column vectors, in ECF coordinates. Note that the perpendicular
        component of col_vector with respect to the row_vector will be used.

        Parameters
        ----------
        row_vector : None|numpy.ndarray
            The vector defining increasing column direction. If None, then the
            sicd.Grid.Row.UVectECF will be used.
        col_vector : None|numpy.ndarray
            The vector defining increasing column direction. It is required
            that `row_vector` and `col_vector` are orthogonal. If None, then the
            perpendicular component of sicd.Grid.Col.UVectECF will be used.

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
            if perp is not None:
                vec = vec - perp*(perp.dot(vec))

            norm = numpy.linalg.norm(vec)
            if norm == 0:
                raise ValueError('{} vector cannot be the zero vector.'.format(name))
            elif norm != 1:
                vec = vec/norm  # avoid modifying row_vector def exterior to this class
            return vec

        if row_vector is None:
            row_vector = self.sicd.Grid.Row.UVectECF.get_array()
        if col_vector is None:
            col_vector = self.sicd.Grid.Col.UVectECF.get_array()

        self._row_vector = normalize(row_vector, 'row')
        self._col_vector = normalize(col_vector, 'column', self._row_vector)

    def ecf_to_ortho(self, coords):
        coords, o_shape = self._reshape(coords, 3)

        diff = coords - self.reference_point
        if len(o_shape) == 1:
            out = numpy.zeros((2, ), dtype=numpy.float64)
            out[0] = diff.dot(self.row_vector)/self.row_spacing
            out[1] = diff.dot(self.col_vector)/self.col_spacing
        else:
            out = numpy.zeros((coords.shape[0], 2), dtype=numpy.float64)
            out[:, 0] = diff.dot(self.row_vector)/self.row_spacing
            out[:, 1] = diff.dot(self.col_vector)/self.col_spacing
            out = numpy.reshape(out, o_shape[:-1] + (2, ))
        return out

    def ll_to_ortho(self, ll_coords):
        ll_coords, o_shape = self._reshape(ll_coords, 2)
        llh_temp = numpy.zeros((ll_coords.shape[0], 3), dtype=numpy.float64)
        llh_temp[:, :2] = ll_coords
        llh_temp = numpy.reshape(llh_temp, o_shape[:-1]+ (3, ))
        return self.llh_to_ortho(llh_temp)

    def llh_to_ortho(self, llh_coords):
        llh_coords, o_shape = self._reshape(llh_coords, 3)
        ground = geodetic_to_ecf(llh_coords)
        ground = numpy.reshape(ground, o_shape)
        return self.ecf_to_ortho(ground)

    def ortho_to_ecf(self, ortho_coords):
        ortho_coords, o_shape = self._reshape(ortho_coords, 2)
        xs = ortho_coords[:, 0]*self.row_spacing
        ys = ortho_coords[:, 1]*self.col_spacing
        if xs.ndim == 0:
            coords = self._reference_point + xs*self.row_vector + ys*self.col_vector
        else:
            coords = self._reference_point + xs[:, numpy.newaxis]*self._row_vector + \
                     ys[:, numpy.newaxis]*self._col_vector
        return numpy.reshape(coords, o_shape)

    def ortho_to_pixel(self, ortho_coords):
        ortho_coords, o_shape = self._reshape(ortho_coords, 2)
        xs = ortho_coords[:, 0]*self.row_spacing
        ys = ortho_coords[:, 1]*self.col_spacing
        if xs.ndim == 0:
            coords = self.sicd.project_ground_to_image(
                self._reference_point + xs*self._row_vector + ys*self._col_vector)
        else:
            coords = self.sicd.project_ground_to_image(
                self._reference_point + xs[:, numpy.newaxis]*self._row_vector +
                ys[:, numpy.newaxis]*self._col_vector)
        return numpy.reshape(coords, o_shape)

    def pixel_to_ortho(self, pixel_coords):
        return self.ecf_to_ortho(self.sicd.project_image_to_ground(pixel_coords, projection_type='HAE'))

################
# The orthorectification methodology

class OrthorectificationHelper(object):
    """
    Abstract helper class which defines ortho-rectification process for a sicd-type
    reader object.
    """

    __slots__ = ('_reader', '_index', '_proj_helper', '_out_dtype', '_complex_valued', '_pad_value')

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False, pad_value=None):
        """

        Parameters
        ----------
        reader : BaseReader
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
        """

        self._pad_value = pad_value
        self._out_dtype = numpy.dtype('float32')
        if complex_valued:
            self._out_dtype = numpy.dtype('complex64')
        self._index = None
        self._proj_helper = None
        if not isinstance(reader, BaseReader):
            raise TypeError('Got unexpected type {} for reader'.format(type(reader)))
        if not reader.is_sicd_type:
            raise ValueError('Reader is required to have is_sicd_type property value equals True')
        self._reader = reader
        self.set_index_and_proj_helper(index, proj_helper=proj_helper)

    @property
    def reader(self):
        # type: () -> BaseReader
        """
        BaseReader: The reader instance.
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

        if proj_helper is None:
            proj_helper = PGProjection(self.reader.get_sicds_as_tuple()[index])
        if not isinstance(proj_helper, ProjectionHelper):
            raise TypeError('Got unexpected type {} for proj_helper'.format(proj_helper))
        self._proj_helper = proj_helper

    def get_orthorectification_bounds(self, coordinates):
        """
        Determine the ortho-rectified (coordinate-system aligned) rectangular bounding
        region which contains the provided coordinates.

        .. Note: This assumes that the coordinate transforms are convex transformations,
            which **should** be safe for SAR associated transforms.

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
                 [pixel_bounds[siz], pixel_bounds[siz+1]],
                 [pixel_bounds[0], pixel_bounds[siz+1]]], dtype=numpy.float64)
        ortho = self.proj_helper.pixel_to_ortho(coordinates)
        return self.proj_helper.get_pixel_array_bounds(ortho)

    def _validate_bounds(self, bounds):
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
            raise ValueError('bounds is required to be one-dimensional and size 4.')
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

    def _bounds_to_rectangle(self, bounds):
        """
        From a bounds style array, construct the four corner coordinate array.

        Parameters
        ----------
        bounds : numpy.ndarray|list|tuple

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The (integer valued) bounds and rectangular coordinates.
        """

        bounds = self._validate_bounds(bounds)
        coords = numpy.zeros((4, 2), dtype=numpy.int32)
        coords[0, :] = (bounds[0], bounds[2])
        coords[1, :] = (bounds[1], bounds[2])
        coords[2, :] = (bounds[1], bounds[3])
        coords[3, :] = (bounds[0], bounds[3])
        return bounds, coords

    def _extract_bounds(self, bounds):
        """
        Validate the bounds array of orthorectified pixel coordinates, and determine
        the required bounds in reader pixel coordinates.

        Parameters
        ----------
        bounds : numpy.ndarray|list|tuple

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The integer valued orthorectified and reader pixel coordinate bounds.
        """

        bounds, coords = self._bounds_to_rectangle(bounds)
        pixel_coords = self.proj_helper.ortho_to_pixel(coords)
        pixel_bounds = self.proj_helper.get_pixel_array_bounds(pixel_coords)
        return bounds, self._validate_bounds(pixel_bounds)

    def _initialize_workspace(self, ortho_bounds):
        """
        Initialize the orthorectification array workspace.

        Parameters
        ----------
        ortho_bounds : numpy.ndarray
            Of the form `(min row, max row, min col, max col)`.

        Returns
        -------
        numpy.ndarray
        """

        out_shape = (int(ortho_bounds[1]-ortho_bounds[0]), int(ortho_bounds[3]-ortho_bounds[2]))
        return numpy.zeros(out_shape, dtype=self.out_dtype) if self._pad_value is None else \
            numpy.full(out_shape, self._pad_value, dtype=self.out_dtype)

    def _get_ortho_mesh(self, ortho_bounds):
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
        ortho_mesh[:, :, 1], ortho_mesh[:, :, 0] = numpy.meshgrid(numpy.range(ortho_bounds[2], ortho_bounds[3]),
                                                                  numpy.range(ortho_bounds[0], ortho_bounds[1]))
        return ortho_mesh

    def _get_real_pixel_limits_and_bounds(self, pixel_bounds):
        """
        Fetch the real pixel limit from the nominal pixel limits - this just
        factors in the image reader extent.

        Parameters
        ----------
        pixel_bounds : numpy.ndarray

        Returns
        -------
        (tuple, numpy.ndarray)
        """

        pixel_limits = self.reader.get_data_size_as_tuple()[self.index]
        real_pix_bounds = numpy.array([
            max(0, pixel_bounds[0]), min(pixel_limits[0], pixel_bounds[1]),
            max(0, pixel_bounds[2]), min(pixel_limits[1], pixel_bounds[3])], dtype=numpy.int32)
        return pixel_limits, real_pix_bounds

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

        raise NotImplementedError

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

        pixel_bounds, pixel_rect = self._validate_bounds(pixel_bounds)
        return self.get_orthorectified_for_pixel_object(pixel_rect)

    def get_orthorectified_for_pixel_object(self, coordinates):
        """
        Determine the ortho-rectified rectangular array values, which will bound
        the given object - with coordinates expressed in pixel space.

        .. Note: This assumes that the coordinate transforms are convex transformations,
            which should be safe for basic SAR associated transforms.

        Parameters
        ----------
        coordinates : GeometryObject|numpy.ndarray|list|tuple
            The coordinate system of the input will be assumed to be pixel space.

        Returns
        -------
        numpy.ndarray
        """

        bounds = self.get_orthorectification_bounds(coordinates)
        return self.get_orthorectified_for_ortho_bounds(bounds)


class NearestNeighborMethod(OrthorectificationHelper):
    """
    Nearest neighbor ortho-rectification method.
    """

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False):
        """

        Parameters
        ----------
        reader : BaseReader
        index : int
        proj_helper : None|ProjectionHelper
            If `None`, this will default to `PGProjection(<sicd>)`, where `<sicd>`
            will be the sicd from `reader` at `index`. Otherwise, it is the user's
            responsibility to ensure that `reader`, `index` and `proj_helper` are
            in sync.
        complex_valued : bool
            Do we want complex values returned? If `False`, the magnitude values
            will be used.
        """

        super(NearestNeighborMethod, self).__init__(
            reader, index=index, proj_helper=proj_helper, complex_valued=complex_valued)

    def get_orthorectified_for_ortho_bounds(self, bounds):
        ortho_bounds, nominal_pixel_bounds = self._extract_bounds(bounds)
        ortho_array = self._initialize_workspace(ortho_bounds)
        # extract the values - ensure that things are within proper image bounds
        pixel_limits, pixel_bounds = self._get_real_pixel_limits_and_bounds(nominal_pixel_bounds)
        pixel_array = self.reader[
            pixel_bounds[0]:pixel_bounds[1], pixel_bounds[2]:pixel_bounds[3], self.index]
        if not self._complex_valued:
            pixel_array = numpy.abs(pixel_array)

        # determine the pixel coordinates for the ortho coordinates meshgrid
        ortho_mesh = self._get_ortho_mesh(ortho_bounds)
        # determine the nearest neighbor pixel coordinates
        pixel_mesh = numpy.cast[numpy.int32](numpy.round(self.proj_helper.ortho_to_pixel(ortho_mesh)))
        pixel_rows = pixel_mesh[:, :, 0]
        pixel_cols = pixel_mesh[:, :, 1]
        # determine the in bounds points
        mask = ((pixel_rows >= 0) & (pixel_rows < pixel_limits[0]) &
                (pixel_cols >= 0) & (pixel_cols < pixel_limits[1]))
        ortho_array[mask] = pixel_array[pixel_rows[mask]-pixel_bounds[0], pixel_cols[mask]-pixel_bounds[2]]
        return ortho_array


class BivariateSplineMethod(OrthorectificationHelper):
    """
    Bivariate spline interpolation ortho-rectification method.
    """

    __slots__ = ('_row_order', '_col_order')

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False, row_order=1, col_order=1):
        """

        Parameters
        ----------
        reader : BaseReader
        index : int
        proj_helper : None|ProjectionHelper
            If `None`, this will default to `PGProjection(<sicd>)`, where `<sicd>`
            will be the sicd from `reader` at `index`. Otherwise, it is the user's
            responsibility to ensure that `reader`, `index` and `proj_helper` are
            in sync.
        complex_valued : bool
            Do we want complex values returned? If `False`, the magnitude values
            will be used.
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
            reader, index=index, proj_helper=proj_helper, complex_valued=complex_valued)
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

    def get_orthorectified_for_ortho_bounds(self, bounds):
        ortho_bounds, nominal_pixel_bounds = self._extract_bounds(bounds)
        ortho_array = self._initialize_workspace(ortho_bounds)
        # extract the values - ensure that things are within proper image bounds
        pixel_limits, pixel_bounds = self._get_real_pixel_limits_and_bounds(nominal_pixel_bounds)
        pixel_array = self.reader[
            pixel_bounds[0]:pixel_bounds[1], pixel_bounds[2]:pixel_bounds[3], self.index]
        if not self._complex_valued:
            pixel_array = numpy.abs(pixel_array)

        # setup the spline
        sp = RectBivariateSpline(
            numpy.range(pixel_bounds[0], pixel_bounds[1]),
            numpy.range(pixel_bounds[2], pixel_bounds[3]),
            pixel_array,
            kx=self.row_order, ky=self.col_order, s=0)
        # determine the pixel coordinates for the ortho coordinates meshgrid
        ortho_mesh = self._get_ortho_mesh(ortho_bounds)
        # determine the nearest neighbor pixel coordinates
        pixel_mesh = self.proj_helper.ortho_to_pixel(ortho_mesh)
        pixel_rows = pixel_mesh[:, :, 0]
        pixel_cols = pixel_mesh[:, :, 1]
        # determine the in bounds points
        mask = ((pixel_rows >= 0) & (pixel_rows < pixel_limits[0]) &
                (pixel_cols >= 0) & (pixel_cols < pixel_limits[1]))
        ortho_array[mask] = sp(pixel_rows[mask], pixel_cols[mask])
        return ortho_array
