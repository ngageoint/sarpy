"""
Unified methods of projection between sicd pixel coordinates,
some ortho-rectified pixel grid coordinates, and geophysical coordinates
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging

import numpy
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic, wgs_84_norm
from sarpy.processing.rational_polynomial import SarpyRatPolyError, \
    get_rational_poly_2d, get_rational_poly_3d, CombinedRationalPolynomial


logger = logging.getLogger(__name__)

_PIXEL_METHODOLOGY = ('MAX', 'MIN', 'MEAN', 'GEOM_MEAN')


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
    for a sicd-type object using the SICD projection model directly.
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
        ProjectionHelper.__init__(
            self, sicd, row_spacing=row_spacing, col_spacing=col_spacing, default_pixel_method=default_pixel_method)
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
            row_vector = normalize(row_vector, 'row')
            check_perp(normal_vector, row_vector, 'normal', 'row')
            self._normal_vector = normal_vector
            self._row_vector = row_vector
            self._col_vector = numpy.cross(self.normal_vector, self.row_vector)
        elif row_vector is None:
            normal_vector = normalize(normal_vector, 'normal')
            col_vector = normalize(col_vector, 'col')
            check_perp(normal_vector, col_vector, 'normal', 'col')
            self._normal_vector = normal_vector
            self._col_vector = col_vector
            self._row_vector = numpy.cross(self.col_vector, self.normal_vector)
        else:
            normal_vector = normalize(normal_vector, 'normal')
            row_vector = normalize(row_vector, 'row')
            col_vector = normalize(col_vector, 'col')
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

    def plane_ecf_to_ortho(self, coords):
        """
        Converts ECF coordinates **known to be in the ground plane** to ortho grid coordinates.

        Parameters
        ----------
        coords : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

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

    def ecf_to_ortho(self, coords):
        return self.pixel_to_ortho(self.ecf_to_pixel(coords))

    def ecf_to_pixel(self, coords):
        pixel, _, _ = self.sicd.project_ground_to_image(coords, tolerance=1e-6, max_iterations=40)
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
        llh_temp = numpy.reshape(llh_temp, o_shape[:-1] + (3, ))
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
        pixel, _, _ = self.sicd.project_ground_to_image(self.ortho_to_ecf(ortho_coords), tolerance=1e-3, max_iterations=25)
        return numpy.reshape(pixel, o_shape)

    def pixel_to_ortho(self, pixel_coords):
        return self.plane_ecf_to_ortho(self.pixel_to_ecf(pixel_coords))

    def pixel_to_ecf(self, pixel_coords):
        return self.sicd.project_image_to_ground(
            pixel_coords, projection_type='PLANE',
            gref=self.reference_point, ugpn=self.normal_vector)


class PGRatPolyProjection(PGProjection):

    __slots__ = (
        '_reference_point', '_reference_pixels', '_row_vector', '_col_vector',
        '_normal_vector', '_reference_hae',
        '_row_samples', '_col_samples', '_alt_samples', '_alt_span',
        '_ecf_to_pixel_func', '_pixel_to_ortho_func', '_ortho_to_pixel_func')

    def __init__(self, sicd, reference_point=None, reference_pixels=None, normal_vector=None, row_vector=None,
                 col_vector=None, row_spacing=None, col_spacing=None,
                 default_pixel_method='GEOM_MEAN',
                 row_samples=51, col_samples=51, alt_samples=11, alt_span=250):
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
        row_samples : int
            How many row samples to use in fitting
        col_samples : int
            How many column samples to use in fitting
        alt_samples : int
            How many altitude samples to use in fitting
        alt_span : int|float
            Fitting for reference point hae +/- alt_span information.
        """

        self._ecf_to_pixel_func = None
        self._pixel_to_ortho_func = None
        self._ortho_to_pixel_func = None
        self._row_samples = int(row_samples)
        self._col_samples = int(col_samples)
        self._alt_samples = int(alt_samples)
        self._alt_span = float(alt_span)

        PGProjection.__init__(
            self, sicd, reference_point=reference_point, reference_pixels=reference_pixels,
            normal_vector=normal_vector, row_vector=row_vector, col_vector=col_vector,
            row_spacing=row_spacing, col_spacing=col_spacing, default_pixel_method=default_pixel_method)
        self.perform_rational_poly_fitting()

    def _perform_ecf_func_fitting(self):
        num_rows = self.sicd.ImageData.NumRows
        num_cols = self.sicd.ImageData.NumCols

        row_array = numpy.linspace(0, num_rows-1, self._row_samples)
        col_array = numpy.linspace(0, num_cols-1, self._col_samples)
        hae_array = self.reference_hae + numpy.linspace(
            -self._alt_span, self._alt_span, self._alt_samples)

        row_col_grid = numpy.empty((row_array.size, col_array.size, 2), dtype='float64')
        row_col_grid[:, :, 1], row_col_grid[:, :, 0] = numpy.meshgrid(col_array, row_array)

        ECF_data = numpy.empty((row_array.size, col_array.size, hae_array.size, 3))
        for i, hae0 in enumerate(hae_array):
            ECF_data[:, :, i, :] = self.sicd.project_image_to_ground(row_col_grid, projection_type='HAE', hae0=hae0)

        if not numpy.all(numpy.isfinite(ECF_data)):
            raise SarpyRatPolyError(
                'NaN values are encountered when projecting across the image area,\n\t'
                'this SICD is not a good candidate for projection using rational polynomials')
        row_func = get_rational_poly_3d(
            ECF_data[:, :, :, 0].flatten(), ECF_data[:, :, :, 1].flatten(), ECF_data[:, :, :, 2].flatten(),
            numpy.stack([row_col_grid[:, :, 0] for _ in range(self._alt_samples)], axis=2).flatten(), order=3)
        col_func = get_rational_poly_3d(
            ECF_data[:, :, :, 0].flatten(), ECF_data[:, :, :, 1].flatten(), ECF_data[:, :, :, 2].flatten(),
            numpy.stack([row_col_grid[:, :, 1] for _ in range(self._alt_samples)], axis=2).flatten(), order=3)
        self._ecf_to_pixel_func = CombinedRationalPolynomial(row_func, col_func)

    def _perform_pixel_fitting(self):
        num_rows = self.sicd.ImageData.NumRows
        num_cols = self.sicd.ImageData.NumCols
        row_array = numpy.linspace(0, num_rows-1, 2*self._row_samples)
        col_array = numpy.linspace(0, num_cols-1, 2*self._col_samples)
        pixel_data = numpy.empty((row_array.size, col_array.size, 2), dtype='float64')
        pixel_data[:, :, 1], pixel_data[:, :, 0] = numpy.meshgrid(col_array, row_array)

        ortho_data = PGProjection.pixel_to_ortho(self, pixel_data)
        pix_to_orth_row = get_rational_poly_2d(
            pixel_data[:, :, 0].flatten(), pixel_data[:, :, 1].flatten(),
            ortho_data[:, :, 0], order=3)
        pix_to_orth_col = get_rational_poly_2d(
            pixel_data[:, :, 0].flatten(), pixel_data[:, :, 1].flatten(),
            ortho_data[:, :, 1], order=3)
        self._pixel_to_ortho_func = CombinedRationalPolynomial(pix_to_orth_row, pix_to_orth_col)

        orth_to_pix_row = get_rational_poly_2d(
            ortho_data[:, :, 0].flatten(), ortho_data[:, :, 1].flatten(),
            pixel_data[:, :, 0], order=3)
        orth_to_pix_col = get_rational_poly_2d(
            ortho_data[:, :, 0].flatten(), ortho_data[:, :, 1].flatten(),
            pixel_data[:, :, 1], order=3)
        self._ortho_to_pixel_func = CombinedRationalPolynomial(orth_to_pix_row, orth_to_pix_col)

    def perform_rational_poly_fitting(self):
        """
        Defined the rational polynomial functions via fitting.
        """

        self._perform_ecf_func_fitting()
        self._perform_pixel_fitting()

    def ecf_to_ortho(self, coords):
        return self.pixel_to_ortho(self.ecf_to_pixel(coords))

    def ecf_to_pixel(self, coords):
        return self._ecf_to_pixel_func(coords, combine=True)

    def ortho_to_pixel(self, ortho_coords):
        return self._ortho_to_pixel_func(ortho_coords, combine=True)

    def pixel_to_ortho(self, pixel_coords):
        return self._pixel_to_ortho_func(pixel_coords, combine=True)
