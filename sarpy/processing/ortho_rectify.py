# -*- coding: utf-8 -*-
"""
SICD ortho-rectification methodology.
"""

import numpy

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.base import BaseReader
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic
from sarpy.geometry.geometry_elements import GeometryObject


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


class OrthorectificationHelper(object):
    """
    Abstract helper class which defines ortho-rectification process for a sicd-type
    reader object.
    """

    __slots__ = ('_reader', '_index', '_proj_helper')

    def __init__(self, reader, index=0, proj_helper=None):
        """

        Parameters
        ----------
        reader : BaseReader
        index : int
        proj_helper : None|ProjectionHelper
            If `None`, this will default to `PGProjection(<sicd>)`, where `<sicd>`
            will be the sicd from `reader` at `index`.
        """

        if not isinstance(reader, BaseReader):
            raise TypeError('Got unexpected type {} for reader'.format(type(reader)))
        if not reader.is_sicd_type:
            raise ValueError('Reader is required to have is_sicd_type property value equals True')
        self._reader = reader
        self._index = index

        if proj_helper is None:
            proj_helper = PGProjection(reader.get_sicds_as_tuple()[index])
        if not isinstance(proj_helper, ProjectionHelper):
            raise TypeError('Got unexpected type {} for proj_helper'.format(proj_helper))
        self._proj_helper = proj_helper

    @property
    def reader(self):
        """
        BaseReader: The reader instance.
        """

        return self._reader

    @property
    def index(self):
        """
        int: The index for the desired sicd element.
        """
        return self._index


    @property
    def proj_helper(self):
        """
        ProjectionHelper: The projection helper instance.
        """

        return self._proj_helper

    def get_orthorectification_bounds(self, object):
        """
        Determine the ortho-rectified (coordinate-system aligned) rectangular bounding
        region which contains the provided object.

        .. Note: This assumes that the coordinate transforms are convex transformations,
            which should be safe for basic SAR associated transforms.

        Parameters
        ----------
        object : GeometryObject|numpy.ndarray|list|tuple
            The coordinate system of the input will be assumed to be pixel space.

        Returns
        -------
        numpy.ndarray
            Of the form `(row_min, row_max, col_min, col_max)`.
        """

        if isinstance(object, GeometryObject):
            pixel_bounds = object.get_bbox()
            siz = int(len(pixel_bounds)/2)
            object = numpy.array(
                [[pixel_bounds[0], pixel_bounds[1]],
                 [pixel_bounds[siz], pixel_bounds[1]],
                 [pixel_bounds[siz], pixel_bounds[siz+1]],
                 [pixel_bounds[0], pixel_bounds[siz+1]]], dtype=numpy.float64)
        ortho = self.proj_helper.pixel_to_ortho(object)
        return self.proj_helper.get_pixel_array_bounds(ortho)
