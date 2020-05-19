# -*- coding: utf-8 -*-
"""
Some ortho-rectification methods.
"""

import numpy

from sarpy.io.complex.sicd_elements.SICD import SICDType


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
            The reference point (origin) of the planar grid. If None, the SCP will be used.
        row_vector : None|numpy.ndarray
            The vector defining increasing column direction. If None, then the
            sicd.Grid.Row.UVectECF will be used.
        col_vector : None|numpy.ndarray
            The vector defining increasing column direction. It is required
            that `row_vector` and `col_vector` are orthogonal. If None, then the
            perpendicular component of sicd.Grid.Col.UVectECF will be used.
        """

        if not isinstance(sicd, SICDType):
            raise TypeError('sicd must be a SICDType instance. Got type {}'.format(type(sicd)))
        if not sicd.can_project_coordinates():
            raise ValueError('Orthorectification requires the SICD ability to project coordinates.')
        sicd.define_coa_projection(overide=False)
        self._sicd = sicd

        if reference_point is None:
            reference_point = sicd.GeoData.SCP.ECF.get_array()

        if not (isinstance(reference_point, numpy.ndarray) and reference_point.ndim == 1
                and reference_point.size == 3):
            raise ValueError('reference_point must be a vector of size 3.')
        self._reference_point = reference_point

        if row_vector is None:
            row_vector = sicd.Grid.Row.UVectECF.get_array()

        if not (isinstance(row_vector, numpy.ndarray) and row_vector.ndim == 1 and row_vector.size == 3):
            raise ValueError('row_vector must be a vector of size 3.')
        row_norm = numpy.linalg.norm(row_vector)
        if row_norm == 0:
            raise ValueError('row_vector cannot be the zero vector.')
        row_vector = row_vector/row_norm  # avoid modifying row_vector def exterior to this class

        if col_vector is None:
            col_vector = sicd.Grid.Col.UVectECF.get_array()
            # take perpendicular component to row_vector, since they may not be perpendicular
            #   normalization handled below
            col_vector = col_vector - row_vector*(row_vector.dot(col_vector))

        if not (isinstance(col_vector, numpy.ndarray) and col_vector.ndim == 1 and col_vector.size == 3):
            raise ValueError('col_vector must be a vector of size 3.')
        col_norm = numpy.linalg.norm(col_vector)
        if col_norm == 0:
            raise ValueError('col_vector cannot be the zero vector.')
        col_vector = col_vector/col_norm  # avoid modifying col_vector exterior to this class

        overlap = col_vector.dot(row_vector)
        if overlap != 0.0:
            raise ValueError('row_vector and col_vector must be orthogonal. '
                             'Got dot product {}'.format(overlap))

        self._row_vector = row_vector
        self._col_vector = col_vector

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
