# -*- coding: utf-8 -*-
"""
SICD ortho-rectification methodology.

Example Usage

>>> import os
>>> from matplotlib import pyplot
>>> from sarpy.io.complex.converter import open_complex
>>> from sarpy.processing.ortho_rectify import NearestNeighborMethod
>>> from sarpy.visualization.remap import linear_discretization
>>> from PIL import Image

>>> reader = open_complex(<file name>)
>>> orth_method = NearestNeighborMethod(reader, index=0, proj_helper=None,
>>>     complex_valued=False, apply_radiometric=None, subtract_radiometric_noise=False)

>>> # Perform ortho-rectification of the entire image
>>> # This will take a long time and be very RAM intensive, unless the image is small
>>> ortho_bounds = orth_method.get_full_ortho_bounds()
>>> ortho_data = orth_method.get_orthorectified_for_ortho_bounds(ortho_bounds)

>>> # or, perform ortho-rectification on a given rectangular region in pixel space
>>> pixel_bounds = [100, 200, 200, 300]  # [first_row, last_row, first_column, last_column]
>>> ortho_data = orth_method.get_orthorectified_for_pixel_bounds(pixel_bounds)

>>> # view the data using matplotlib
>>> fig, axs = pyplot.subplots(nrows=1, ncols=1)
>>> h1 = axs.imshow(ortho_data, cmap='inferno', aspect='equal')
>>> fig.colorbar(h1, ax=axs)
>>> pyplot.show()

>>> # write out the data to some image file format
>>> discrete_image = linear_discretization(ortho_data, min_value=None, max_value=None, bit_depth=8)
>>> Image.fromarray(discrete_image).save('filename.jpg')
"""

from typing import Union

import numpy
from scipy.interpolate import RectBivariateSpline

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.general.base import BaseReader
from sarpy.io.general.utils import string_types
from sarpy.io.complex.sicd_elements.blocks import Poly2DType
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic, wgs_84_norm
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
            The row pixel spacing. If not provided, this will default to
            `min(sicd.Grid.Row.SS, sicd.Grid.Col.SS).`
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
            self._row_spacing = min(self.sicd.Grid.Row.SS, self.sicd.Grid.Col.SS)
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
            self._col_spacing = min(self.sicd.Grid.Row.SS, self.sicd.Grid.Col.SS)
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
        '_reference_point', '_row_vector', '_col_vector', '_normal_vector', '_reference_hae')

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
        self._reference_hae = None
        self._row_vector = None
        self._col_vector = None
        self._normal_vector = None
        super(PGProjection, self).__init__(sicd, row_spacing=row_spacing, col_spacing=col_spacing)
        self.set_reference_point(reference_point)
        self.set_row_and_col_vector(row_vector, col_vector)

    @property
    def reference_point(self):
        # type: () -> numpy.ndarray
        """
        numpy.ndarray: The grid reference point.
        """

        return self._reference_point

    @property
    def normal_vector(self):
        # type: () -> numpy.ndarray
        """
        numpy.ndarray: The normal vector.
        """

        return self._normal_vector

    @property
    def reference_hae(self):
        # type: () -> float
        """
        float: The reference point HAE.
        """

        return self._reference_hae

    def set_reference_point(self, reference_point):
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
            reference_point = self.sicd.GeoData.SCP.ECF.get_array()

        if not (isinstance(reference_point, numpy.ndarray) and reference_point.ndim == 1
                and reference_point.size == 3):
            raise ValueError('reference_point must be a vector of size 3.')
        self._reference_point = reference_point
        self._normal_vector = wgs_84_norm(reference_point)
        llh = ecf_to_geodetic(reference_point)
        self._reference_hae = float(llh[2])

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

        if row_vector is None:
            row_vector = self.sicd.Grid.Row.UVectECF.get_array()
        if col_vector is None:
            col_vector = self.sicd.Grid.Col.UVectECF.get_array()

        # make perpendicular to the plane normal vector
        self._row_vector = normalize(row_vector, 'row', perp=self.normal_vector)
        # make perpendicular to the plane normal vector and row_vector
        self._col_vector = normalize(col_vector, 'column', perp=(self.normal_vector, self._row_vector))

    def ecf_to_ortho(self, coords):
        coords, o_shape = self._reshape(coords, 3)
        diff = coords - self.reference_point
        if len(o_shape) == 1:
            out = numpy.zeros((2, ), dtype=numpy.float64)
            out[0] = numpy.sum(diff*self.row_vector)/self.row_spacing
            out[1] = numpy.sum(diff*self.col_vector)/self.col_spacing
        else:
            out = numpy.zeros((coords.shape[0], 2), dtype=numpy.float64)
            out[:, 0] = numpy.sum(diff*self.row_vector, axis=1)/self.row_spacing
            out[:, 1] = numpy.sum(diff*self.col_vector, axis=1)/self.col_spacing
            out = numpy.reshape(out, o_shape[:-1] + (2, ))
        return out

    def ecf_to_pixel(self, coords):
        # ground to image
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
        xs = ortho_coords[:, 0]*self.row_spacing
        ys = ortho_coords[:, 1]*self.col_spacing
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
        return self.sicd.project_image_to_ground(pixel_coords, projection_type='PLANE')


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
        '_rad_poly', '_noise_poly')

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False,
                 pad_value=None, apply_radiometric=None, subtract_radiometric_noise=False):
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

        self._pad_value = pad_value
        self._complex_valued = complex_valued
        if self._complex_valued:
            self._out_dtype = numpy.dtype('complex64')
        else:
            self._out_dtype = numpy.dtype('float32')
        if not isinstance(reader, BaseReader):
            raise TypeError('Got unexpected type {} for reader'.format(type(reader)))
        if not reader.is_sicd_type:
            raise ValueError('Reader is required to have is_sicd_type property value equals True')
        self._reader = reader
        self.apply_radiometric = apply_radiometric
        self.subtract_radiometric_noise = subtract_radiometric_noise
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

        if proj_helper is None:
            proj_helper = PGProjection(self._sicd)
        if not isinstance(proj_helper, ProjectionHelper):
            raise TypeError('Got unexpected type {} for proj_helper'.format(proj_helper))
        self._proj_helper = proj_helper

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
        elif isinstance(value, string_types):
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
        full_coords = self.sicd.ImageData.get_full_vertex_data()
        return self.get_orthorectification_bounds_from_pixel_object(full_coords)

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

        valid_coords = self.sicd.ImageData.get_valid_vertex_data()
        if valid_coords is None:
            valid_coords = self.sicd.ImageData.get_full_vertex_data()
        return self.get_orthorectification_bounds_from_pixel_object(valid_coords)

    def get_orthorectification_bounds_from_pixel_object(self, coordinates):
        """
        Determine the ortho-rectified (coordinate-system aligned) rectangular bounding
        region which contains the provided coordinates in pixel space.

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

    def get_orthorectification_bounds_from_latlon_object(self, coordinates):
        """
        Determine the ortho-rectified (coordinate-system aligned) rectangular bounding
        region which contains the provided coordinates in lat/lon space.

        .. Note: This neglects the non-convexity of lat/lon coordinate space.

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
    def _validate_bounds(bounds):
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

        mask = ((pixel_rows >= row_array[0]) & (pixel_rows < row_array[-1]) &
                (pixel_cols >= col_array[0]) & (pixel_cols < col_array[-1]))
        return mask

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

        ortho_bounds, nominal_pixel_bounds = self._extract_bounds(bounds)
        # extract the values - ensure that things are within proper image bounds
        pixel_limits, pixel_bounds = self._get_real_pixel_limits_and_bounds(nominal_pixel_bounds)
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

        pixel_bounds, pixel_rect = self._bounds_to_rectangle(pixel_bounds)
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

        bounds = self.get_orthorectification_bounds_from_pixel_object(coordinates)
        return self.get_orthorectified_for_ortho_bounds(bounds)

    def get_orthorectified_for_latlon_object(self, ll_coordinates):
        """
        Determine the ortho-rectified rectangular array values, which will bound
        the given object - with coordinates expressed in lat/lon space.

        .. Note: This assumes that the coordinate transforms are convex transformations,
            which should be safe for basic SAR associated transforms.

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
            Determines the orthorectified bounds reguon, of the form
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
            Determines the orthorectified bounds reguon, of the form
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
    """

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False,
                 pad_value=None, apply_radiometric=None, subtract_radiometric_noise=False):
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
    """

    __slots__ = ('_row_order', '_col_order')

    def __init__(self, reader, index=0, proj_helper=None, complex_valued=False,
                 pad_value=None, apply_radiometric=None, subtract_radiometric_noise=False,
                 row_order=1, col_order=1):
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

        # set up our spline
        sp = RectBivariateSpline(row_array, col_array, value_array, kx=self.row_order, ky=self.col_order, s=0)
        # determine the in bounds points
        mask = self._get_mask(pixel_rows, pixel_cols, row_array, col_array)
        result = sp.ev(pixel_rows[mask], pixel_cols[mask])
        # potentially apply the radiometric parameters
        ortho_array[mask] = result
        return ortho_array
