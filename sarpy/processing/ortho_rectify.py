# -*- coding: utf-8 -*-
"""
Some ortho-rectification methods.
"""

import numpy

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.geometry.geocoords import geodetic_to_ecf


class sicd_pgd_method(object):
    """
    Class which helps perform the Planar Grid (i.e. Ground Plane) orthorectification from a sicd.
    """

    __slots__ = ('_sicd', '_reference_point', '_row_vector', '_col_vector')

    def __init__(self, sicd, reference_point=None, row_vector=None, col_vector=None):
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
        """

        self._row_vector = None
        self._col_vector = None
        self._reference_point = None

        if not isinstance(sicd, SICDType):
            raise TypeError('sicd must be a SICDType instance. Got type {}'.format(type(sicd)))
        if not sicd.can_project_coordinates():
            raise ValueError('Ortho-rectification requires the SICD ability to project coordinates.')
        sicd.define_coa_projection(overide=False)
        self._sicd = sicd

        self.set_reference_point(reference_point)
        self.set_row_and_col_vector(row_vector, col_vector)

    @property
    def sicd(self):
        """
        SICDType: The sicd structure.
        """

        return self._sicd

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

    def get_offset(self, coords):
        """
        Gets the offsets for the given coordinates, assumed to be in ECF coordinates.
        This will return the offsets (row, col) of the coordinate vector(s) from the
        reference_point projected into the plane determined by row_vector and col_vector.

        Parameters
        ----------
        coords : numpy.ndarray|list|tuple

        Returns
        -------
        offsets
            numpy.ndarray
        """

        if not isinstance(coords, numpy.ndarray):
            coords = numpy.array(coords, dtype=numpy.float64)

        if not ((coords.ndim == 1 and coords.shape[0] == 3) or (coords.ndim == 2 and coords[1] == 3)):
            raise ValueError(
                'coords must be of shape (3, ) or (N, 3), and got {}.'.format(coords.shape))

        diff = coords - self.reference_point
        if coords.ndim == 1:
            out = numpy.zeros((2, ), dtype=numpy.float64)
            out[0] = diff.dot(self.row_vector)
            out[1] = diff.dot(self.col_vector)
        else:
            out = numpy.zeros((coords.shape[0], 2), dtype=numpy.float64)
            out[:, 0] = diff.dot(self.row_vector)
            out[:, 1] = diff.dot(self.col_vector)
        return out

    def get_llh_offsets(self, llh_coords):
        """
        Gets the offsets for the given coordinates, assumed to be in Lat/Lon or Lat/Lon/HAE
        coordinates. This will return the offsets (row, col) of the coordinate vector(s) from
        the reference_point projected into the plane determined by row_vector and col_vector.

        Parameters
        ----------
        llh_coords : numpy.ndarray|list|tuple

        Returns
        -------
        offsets
            numpy.ndarray
        """

        if not isinstance(llh_coords, numpy.ndarray):
            llh_coords = numpy.array(llh_coords, dtype=numpy.float64)

        if llh_coords.ndim == 1:
            if llh_coords.shape[0] == 2:
                return self.get_offset(geodetic_to_ecf([llh_coords[0], llh_coords[1], 0]))
            elif llh_coords.shape[0] == 3:
                return self.get_offset(geodetic_to_ecf(llh_coords))
        elif llh_coords.ndim == 2:
            if llh_coords.shape[1] == 2:
                out = numpy.zeros((llh_coords.shape[0], 3), dtype=numpy.float64)
                out[:, :2] = llh_coords
                return self.get_offset(geodetic_to_ecf(out))
            elif llh_coords.shape[1] == 3:
                return self.get_offset(geodetic_to_ecf(llh_coords))
        raise ValueError(
            'llh_coords must be a one or two-dimensional array of the form '
            '[Lat, Lon] or [Lat, Lon, HAE].')

    def get_indices(self, xs, ys):
        r"""
        Get the indices for the point(s) :math:`ref_point + xs\cdot row_vector + ys\cdot col_vector`.
        This relies on :func:`SICDType.ground_to_image`. This requires that `xs` and `ys` are arrays
        of the same shape.

        Parameters
        ----------
        xs : numpy.ndarray
            The x offsets - in meters.
        ys : numpy.ndarray
            The y offsets - in meters.
        Returns
        -------
        numpy.ndarray
            The array of indices, where the shape is `xs.shape + (2, )`, and the final dimension
            indicate (row, column) indices.
        """

        if not isinstance(xs, numpy.ndarray):
            xs = numpy.array(xs, dtype=numpy.float64)
        if not isinstance(ys, numpy.ndarray):
            ys = numpy.array(ys, dtype=numpy.float64)
        if xs.shape != ys.shape:
            raise ValueError('xs and ys must have the same shape.')

        o_shape = xs.shape

        if xs.ndim != 1:
            xs = numpy.ravel(xs)
            ys = numpy.ravel(ys)

        coords = self._sicd.project_ground_to_image(self._reference_point +
                                                    xs[:, numpy.newaxis]*self._row_vector +
                                                    ys[:, numpy.newaxis]*self._col_vector)
        if len(o_shape) == 0:
            return numpy.reshape(coords, (2, ))
        elif len(o_shape) == 1:
            return coords
        else:
            return numpy.reshape(coords, o_shape + (2, ))
