"""
Radar Generalized Image Quality Equation (RGIQE) calculation(s) and tools for
application to SICD structures and files.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging

import numpy

from sarpy.io.complex.base import SICDTypeReader, FlatSICDReader
from sarpy.processing.windows import get_hamming_broadening_factor
from sarpy.processing.normalize_sicd import sicd_degrade_reweight, is_uniform_weight
from sarpy.io.complex.converter import open_complex

logger = logging.getLogger(__name__)

RNIIRS_FIT_PARAMETERS = numpy.array([3.7555, .3960], dtype='float64')
"""
The RNIIRS calculation parameters determined by empirical data fit 
"""


#####################
# methods for extracting necessary information from the sicd structure

def _verify_sicd_with_noise(sicd):
    """
    Verify that the sicd is appropriately populated with noise.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    """

    if sicd.Radiometric is None:
        raise ValueError(
            'Radiometric is not populated,\n\t'
            'so no noise estimate can be derived.')
    if sicd.Radiometric.SigmaZeroSFPoly is None:
        raise ValueError(
            'Radiometric.SigmaZeroSFPoly is not populated,\n\t'
            'so no sigma0 noise estimate can be derived.')
    if sicd.Radiometric.NoiseLevel is None:
        raise ValueError(
            'Radiometric.NoiseLevel is not populated,\n\t'
            'so no noise estimate can be derived.')
    if sicd.Radiometric.NoiseLevel.NoiseLevelType != 'ABSOLUTE':
        raise ValueError(
            'Radiometric.NoiseLevel.NoiseLevelType is not `ABSOLUTE``,\n\t'
            'so no noise estimate can be derived.')


def get_sigma0_noise(sicd):
    """
    Calculate the absolute noise estimate, in sigma0 power units.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    float
    """

    _verify_sicd_with_noise(sicd)
    noise = sicd.Radiometric.NoiseLevel.NoisePoly[0, 0]  # this is in db
    noise = numpy.exp(numpy.log(10)*0.1*noise)  # this is absolute

    # convert to SigmaZero value
    noise *= sicd.Radiometric.SigmaZeroSFPoly[0, 0]
    return noise


def get_default_signal_estimate(sicd):
    """
    Gets default signal for use in the RNIIRS calculation. This will be
    1.0 for copolar (or unknown) collections, and 0.25 for cross-pole
    collections.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    float
    """

    if sicd.ImageFormation is None or sicd.ImageFormation.TxRcvPolarizationProc is None:
        return 1.0

    # use 1.0 for co-polar collection and 0.25 from cross-polar collection
    pol = sicd.ImageFormation.TxRcvPolarizationProc
    if pol is None or ':' not in pol:
        return 1.0

    pols = pol.split(':')

    return 1.0 if pols[0] == pols[1] else 0.25


def get_bandwidth_area(sicd):
    """
    Calculate the bandwidth area.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    float
    """

    return abs(
        sicd.Grid.Row.ImpRespBW *
        sicd.Grid.Col.ImpRespBW *
        numpy.cos(numpy.deg2rad(sicd.SCPCOA.SlopeAng)))


#########################
# methods for calculating information density and rniirs

def get_information_density(bandwidth_area, signal, noise):
    """
    Calculate the information density from bandwidth area and signal/noise estimates.

    Parameters
    ----------
    bandwidth_area : float|numpy.ndarray
    signal : float|numpy.ndarray
    noise : float|numpy.ndarray

    Returns
    -------
    float|numpy.ndarray
    """

    return bandwidth_area * numpy.log2(1 + signal/noise)


def get_rniirs(information_density):
    r"""
    Calculate an RNIIRS estimate from the information density or
    Shannon-Hartley channel capacity.

    This mapping has been empirically determined by fitting Shannon-Hartley channel
    capacity to RNIIRS for some sample images.

    The basic model is given by
    :math:`rniirs = a_0 + a_1*\log_2(information\_density)`

    To maintain positivity of the estimated rniirs, this transitions to a linear
    model :math:`rniirs = slope*information\_density` with slope given by
    :math:`slope = a_1/(iim\_transition*\log(2))` below the transition point at
    :math:`transition = \exp(1 - \log(2)*a_0/a_1)`.

    Parameters
    ----------
    information_density : float|numpy.ndarray

    Returns
    -------
    float|numpy.ndarray
    """

    a = RNIIRS_FIT_PARAMETERS
    iim_transition = numpy.exp(1 - numpy.log(2) * a[0] / a[1])
    slope = a[1] / (iim_transition * numpy.log(2))

    if not isinstance(information_density, numpy.ndarray):
        information_density = numpy.array(information_density, dtype='float64')
    orig_ndim = information_density.ndim
    if orig_ndim == 0:
        information_density = numpy.reshape(information_density, (1, ))

    out = numpy.empty(information_density.shape, dtype='float64')
    mask = (information_density > iim_transition)
    mask_other = ~mask
    if numpy.any(mask):
        out[mask] = a[0] + a[1]*numpy.log2(information_density[mask])
    if numpy.any(mask_other):
        out[mask_other] = slope*information_density[mask_other]

    if orig_ndim == 0:
        return float(out[0])
    return out


def get_information_density_for_rniirs(rniirs):
    """
    The inverse of :func:`get_rniirs`, this determines the information density
    which yields the given RNIIRS.

    *New in version 1.2.35.*

    Parameters
    ----------
    rniirs : float

    Returns
    -------
    float
    """

    a = RNIIRS_FIT_PARAMETERS
    iim_transition = numpy.exp(1 - numpy.log(2) * a[0] / a[1])
    slope = a[1] / (iim_transition * numpy.log(2))
    rniirs_transition = slope*iim_transition

    if not isinstance(rniirs, numpy.ndarray):
        rniirs = numpy.array(rniirs, dtype='float64')
    orig_ndim = rniirs.ndim
    if orig_ndim == 0:
        rniirs = numpy.reshape(rniirs, (1, ))

    out = numpy.empty(rniirs.shape, dtype='float64')
    mask = (rniirs > rniirs_transition)
    mask_other = ~mask

    if numpy.any(mask):
        out[mask] = numpy.exp2((rniirs[mask] - a[0])/a[1])
    if numpy.any(mask_other):
        out[mask_other] = rniirs[mask_other]/slope

    if orig_ndim == 0:
        return float(out[0])
    return out


def snr_to_rniirs(bandwidth_area, signal, noise):
    """
    Calculate the information_density and RNIIRS estimate from bandwidth area and
    signal/noise estimates.

    It is assumed that geometric effects for signal and noise have been accounted for
    (i.e. use SigmaZeroSFPoly), and signal and noise have each been averaged to a
    single pixel value.

    This mapping has been empirically determined by fitting Shannon-Hartley channel
    capacity to RNIIRS for some sample images.

    Parameters
    ----------
    bandwidth_area : float
    signal : float
    noise : float

    Returns
    -------
    information_density : float
    rniirs : float
    """

    information_density = get_information_density(bandwidth_area, signal, noise)
    rniirs = get_rniirs(information_density)
    return information_density, rniirs


def rgiqe(sicd):
    """
    Calculate the information_density and (default) estimated RNIIRS for the
    given sicd.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    information_density : float
    rniirs : float
    """

    bandwidth_area = get_bandwidth_area(sicd)
    signal = get_default_signal_estimate(sicd)
    noise = get_sigma0_noise(sicd)
    return snr_to_rniirs(bandwidth_area, signal, noise)


def populate_rniirs_for_sicd(sicd, signal=None, noise=None, override=False):
    """
    This populates the value(s) for RNIIRS and information density in the SICD
    structure, according to the RGIQE. **This modifies the sicd structure in place.**

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    signal : None|float
        The signal value, in sigma zero.
    noise : None|float
        The noise equivalent sigma zero value.
    override : bool
        Override the value, if present.
    """

    if sicd.CollectionInfo is None:
        logger.error(
            'CollectionInfo must not be None.\n\t'
            'Nothing to be done for calculating RNIIRS.')
        return

    if sicd.CollectionInfo.Parameters is not None and \
            sicd.CollectionInfo.Parameters.get('PREDICTED_RNIIRS', None) is not None:
        if override:
            logger.info('PREDICTED_RNIIRS already populated, and this value will be overridden.')
        else:
            logger.info('PREDICTED_RNIIRS already populated. Nothing to be done.')
            return

    if noise is None:
        try:
            noise = get_sigma0_noise(sicd)
        except Exception as e:
            logger.error(
                'Encountered an error estimating noise for RNIIRS.\n\t{}'.format(e))
            return

    if signal is None:
        signal = get_default_signal_estimate(sicd)

    try:
        bw_area = get_bandwidth_area(sicd)
    except Exception as e:
        logger.error(
            'Encountered an error estimating bandwidth area for RNIIRS\n\t{}'.format(e))
        return

    inf_density, rniirs = snr_to_rniirs(bw_area, signal, noise)
    logger.info(
        'Calculated INFORMATION_DENSITY = {0:0.5G},\n\t'
        'PREDICTED_RNIIRS = {1:0.5G}'.format(inf_density, rniirs))
    if sicd.CollectionInfo.Parameters is None:
        sicd.CollectionInfo.Parameters = {}  # initialize
    sicd.CollectionInfo.Parameters['INFORMATION_DENSITY'] = '{0:0.2G}'.format(inf_density)
    sicd.CollectionInfo.Parameters['PREDICTED_RNIIRS'] = '{0:0.1f}'.format(rniirs)


def get_bandwidth_noise_distribution(sicd, alpha, desired_information_density=None, desired_rniirs=None):
    r"""
    This function determines SICD degradation parameters (nominally symmetric in
    row/column subaperture degradation) to achieve the desired information density/rniirs.

    There is natural one parameter distribution of reducing bandwidth and adding noise
    to achieve the desired RNIIRS/information density, based on :math:`\alpha \in [0, 1]`.

    The nominal relation is

    .. math::

        desired\_information\_density = bandwidth\_area*(bw\_mult(\alpha))^2)\cdot\log_2\left(1 + signal/(noise*noise\_mult(\alpha))\right).

    For :math:`\alpha=0`, we add no additional noise (:math:`bw\_mult(0) = bw\_min,noise\_mult(0)=1`)
    and use purely sub-aperture degradation, achieved at

    .. math::

        desired\_information\_density = bandwidth\_area*(bw\_min)^2)\cdot\log_2(1 + snr).

    On the other end, at :math:`\alpha=1`, we have :math:`bw\_mult(1)=1, noise\_mult(1)=noise\_mult\_max`
    and derive the noise multiplier which fulfills the required information density.
    For intermediate :math:`0 < \alpha < 1`, we find

    .. math::

        bw_mult(\alpha) = bw\_min\cdot (1-\alpha)

    Then, we find :math:`noise\_multiplier(\alpha)` which fulfills the required
    information density.

    .. note::

        Choosing subaperture windows is fundamentally a discrete operation, and
        this carries over to the realities of choosing bandwidth multipliers. See
        :func:`get_bidirectional_bandwidth_multiplier_possibilities` for all
        feasible values for bandwidth multipliers and associated noise adjustment
        details.

    *Refined in version 1.2.35.*

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    alpha : float|numpy.ndarray
    desired_information_density : None|float
    desired_rniirs : None|float

    Returns
    -------
    bandwidth_multiplier : (float, float)|numpy.ndarray
        The `(row, column)` bandwidth multiplier including the discrete nature
        of this aperture, so the two may not be precisely equal.
    noise_multiplier : float|numpy.ndarray
        The noise multiplier, indicating how much noise to add before the
        subaperture processing.
    """

    # validate the desired information density/rniirs
    if (desired_information_density is None and desired_rniirs is None) or \
            (desired_information_density is not None and desired_rniirs is not None):
        raise ValueError('Exactly one of desired_information_density and desired_rniirs must be provided')

    if not isinstance(alpha, numpy.ndarray):
        alpha = numpy.array(alpha, dtype='float64')
    orig_ndim = alpha.ndim
    if orig_ndim == 0:
        alpha = numpy.reshape(alpha, (1, ))

    if not numpy.all((alpha >= 0) & (alpha <= 1)):
        raise ValueError('values for alpha must be in the interval [0, 1]')

    # get the current information density
    bandwidth_area = get_bandwidth_area(sicd)
    signal = get_default_signal_estimate(sicd)  # NB: this is just 1 or 0.25, no scaling issues
    current_nesz = get_sigma0_noise(sicd)
    snr = signal/current_nesz
    current_inf_density = get_information_density(bandwidth_area, signal, current_nesz)

    if desired_information_density is not None:
        desired_information_density = float(desired_information_density)
    elif desired_rniirs is not None:
        desired_information_density = get_information_density_for_rniirs(float(desired_rniirs))

    if desired_information_density > current_inf_density:
        raise ValueError(
            'The desired information density is {},\n\t'
            'but the current deweighted information density is {}'.format(
                desired_information_density, current_inf_density))

    aperture_size, bw_multiplier = get_bidirectional_bandwidth_multiplier_possibilities(sicd)

    # construct the whole list of bandwidth areas and resulting noises after
    # subaperture degrading and deweighting
    bw_areas = bandwidth_area*numpy.multiply.reduce(bw_multiplier, 1)
    inf_densities = get_information_density(bw_areas, signal, current_nesz)

    if desired_information_density < inf_densities[-1]:
        raise ValueError(
            'The desired information density is {},\n\t'
            'but the minimum possible with pure subaperture degradation is {}'.format(
                desired_information_density, inf_densities[-1]))

    best_index = numpy.argmin((desired_information_density - inf_densities)**2)

    indices = numpy.cast['int32'](best_index - alpha*best_index)
    indices = numpy.clip(indices, 0, best_index)

    this_bw_areas = bw_areas[indices]

    # NB: inf_dens = bw_area*log2(1 + snr/mult))
    #   snr/mult = 2^(inf_dens/bw_area) - 1
    #   mult = snr/(2^(inf_dens/bw_area) - 1))

    required_noise_multiplier = snr/(numpy.exp2(desired_information_density/this_bw_areas) - 1)
    required_noise_multiplier[required_noise_multiplier < 1] = 1

    bw_mult_out = numpy.empty(required_noise_multiplier.shape + (2, ), dtype='float64')
    bw_mult_out[:, 0] = bw_multiplier[indices, 0]
    bw_mult_out[:, 1] = bw_multiplier[indices, 1]

    if orig_ndim == 0:
        return (float(bw_mult_out[0, 0]), float(bw_mult_out[0, 1])), float(required_noise_multiplier[0])
    return bw_mult_out, required_noise_multiplier


#########################
# helpers for quality degradation function

def _get_uniform_weight_dicts(sicd):
    """
    Gets the dictionaries denoting uniform weighting.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    row_weighting : None|dict
    column_weighting : None|dict
    """

    row_weighting = None if is_uniform_weight(sicd, 0) else \
        {'WindowName': 'UNIFORM', 'WgtFunct': numpy.ones((32,), dtype='float64')}
    column_weighting = None if is_uniform_weight(sicd, 1) else \
        {'WindowName': 'UNIFORM', 'WgtFunct': numpy.ones((32,), dtype='float64')}
    return row_weighting, column_weighting


def _validate_reader(reader, index):
    """
    Validate the method input:

    Parameters
    ----------
    reader : SICDTypeReader
    index : int
        The reader index.

    Returns
    -------
    (SICDTypeReader, int)
    """


    if isinstance(reader, str):
        reader = open_complex(reader)

    if not isinstance(reader, SICDTypeReader):
        raise TypeError('reader input must be a path to a complex file, or a sicd type reader instance')
    index = int(index)
    if not (0 <= index < reader.image_count):
        raise ValueError('index must be between 0 and {}, got {}'.format(reader.image_count, index))
    return reader, index


def _map_desired_resolution_to_aperture(
        current_imp_resp_bw, sample_size, direction, direction_size,
        desired_resolution=None, desired_bandwidth=None, broadening_factor=None):
    """
    Determine the appropriate symmetric subaperture range to achieve the desired
    bandwidth or resolution, assuming the given broadening factor.

    Parameters
    ----------
    current_imp_resp_bw : float
    sample_size : float
    direction : str
    direction_size : int
        The size of the array along the given direction.
    desired_resolution : None|float
        The desired ImpRespWid (Row, Col) tuple, which will be mapped to ImpRespBW
        assuming uniform weighting. Exactly one of `desired_resolution` and
        `desired_bandwidth` must be provided.
    desired_bandwidth : None|float
        The desired ImpRespBW. Exactly one of `desired_resolution`
        and `desired_bandwidth` must be provided.
    broadening_factor : None|float
        The only applies if `desired_resolution` is provided. If not provided,
        then UNIFORM weighting will be assumed.

    Returns
    -------
    (None|tuple, bw_factor)
    """

    if desired_resolution is None and desired_bandwidth is None:
        raise ValueError('One of desire_resolution or desired_bandwidth must be supplied.')

    if desired_resolution is not None:
        if broadening_factor is None:
            broadening_factor = get_hamming_broadening_factor(1.0)
        else:
            broadening_factor = float(broadening_factor)
        use_resolution = float(desired_resolution)

        use_bandwidth = broadening_factor/use_resolution
    else:
        use_bandwidth = float(desired_bandwidth)

    if use_bandwidth > current_imp_resp_bw:
        if desired_resolution is not None:
            raise ValueError(
                'After mapping from Desired {} ImpRespWid considering uniform weighting,\n\t'
                'the equivalent desired ImpRespBW is {},\n\t'
                'but the current ImpRespBW is {}'.format(direction, use_bandwidth, current_imp_resp_bw))
        else:
            raise ValueError(
                'Desired {} ImpRespBW is given as {},\n\t'
                'but the current ImpRespBW is {}'.format(direction, use_bandwidth, current_imp_resp_bw))
    elif use_bandwidth == current_imp_resp_bw:
        return None, 1.0
    else:
        oversample = max(1., 1./(sample_size*use_bandwidth))
        ap_size = round(direction_size/oversample)
        start_ind = int(numpy.floor(0.5*(direction_size - ap_size)))
        return (start_ind, start_ind+ap_size), use_bandwidth/current_imp_resp_bw


def _map_bandwidth_parameters(sicd, desired_resolution=None, desired_bandwidth=None):
    """
    Helper function to map desired resolution or bandwidth to the suitable (centered)
    aperture.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    desired_resolution : None|tuple[float, float]
    desired_bandwidth : None|tuple[float, float]

    Returns
    -------
    row_aperture : tuple[int, int]
    row_bw_factor : float
    column_aperture : tuple[float, float]
    column_bw_factor : float
    """

    if desired_resolution is not None:
        # get the broadening factor for uniform weighting
        broadening_factor = get_hamming_broadening_factor(1.0)
        row_aperture, row_bw_factor = _map_desired_resolution_to_aperture(
            sicd.Grid.Row.ImpRespBW, sicd.Grid.Row.SS, 'Row', sicd.ImageData.NumRows,
            desired_resolution=desired_resolution[0], broadening_factor=broadening_factor)
        column_aperture, column_bw_factor = _map_desired_resolution_to_aperture(
            sicd.Grid.Col.ImpRespBW, sicd.Grid.Col.SS, 'Col', sicd.ImageData.NumCols,
            desired_resolution=desired_resolution[1], broadening_factor=broadening_factor)
    elif desired_bandwidth is not None:
        row_aperture, row_bw_factor = _map_desired_resolution_to_aperture(
            sicd.Grid.Row.ImpRespBW, sicd.Grid.Row.SS, 'Row', sicd.ImageData.NumRows,
            desired_bandwidth=desired_bandwidth[0])
        column_aperture, column_bw_factor = _map_desired_resolution_to_aperture(
            sicd.Grid.Col.ImpRespBW, sicd.Grid.Col.SS, 'Col', sicd.ImageData.NumCols,
            desired_bandwidth=desired_bandwidth[1])
    else:
        row_aperture, row_bw_factor = None, 1
        column_aperture, column_bw_factor = None, 1

    return row_aperture, row_bw_factor, column_aperture, column_bw_factor


def get_dimension_bandwidth_multiplier_possibilities(sicd, dimension):
    """
    Gets the bandwidth possibilities for all centered subapertures along the given
    dimension.

    *Introduced in 1.2.35*

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    dimension : int
        One of `{0, 1}`.

    Returns
    -------
    aperture_size : numpy.ndarray
        Of shape `(N, )`
    bandwidth_multiplier : numpy.ndarray
        Of shape `(N, )`
    """

    if dimension == 0:
        ap_size = round(sicd.ImageData.NumRows / sicd.Grid.Row.get_oversample_rate())
    else:
        ap_size = round(sicd.ImageData.NumCols / sicd.Grid.Col.get_oversample_rate())

    aperture_size = numpy.arange(ap_size, 0, -1, dtype='int32')
    bandwidth_multiplier = aperture_size/float(ap_size)

    return aperture_size, bandwidth_multiplier


def get_bidirectional_bandwidth_multiplier_possibilities(sicd):
    """
    Gets the bandwidth possibilities for all centered subapertures shrinking
    along both dimensions symmetrically.

    *Introduced in 1.2.35*

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    aperture_size : numpy.ndarray
        An array of shape `(N, 2)` for row/column separately.
    bandwidth_multiplier : numpy.ndarray
        An array of shape `(N, 2)` for row/column separately.
    """

    row_aperture_size, row_bw_multiplier = get_dimension_bandwidth_multiplier_possibilities(sicd, 0)
    col_aperture_size, col_bw_multiplier = get_dimension_bandwidth_multiplier_possibilities(sicd, 1)

    the_size = max(row_aperture_size.size, col_aperture_size.size)
    aperture_size = numpy.empty((the_size, 2), dtype='int32')
    bandwidth_multiplier = numpy.empty((the_size, 2), dtype='float64')

    row_indexing = numpy.cast['int32'](
        numpy.ceil(float(row_aperture_size.size - 1)*numpy.arange(the_size)/float(the_size - 1)))
    col_indexing = numpy.cast['int32'](
        numpy.ceil(float(col_aperture_size.size - 1)*numpy.arange(the_size)/float(the_size - 1)))

    aperture_size[:, 0] = row_aperture_size[row_indexing]
    aperture_size[:, 1] = col_aperture_size[col_indexing]

    bandwidth_multiplier[:, 0] = row_bw_multiplier[row_indexing]
    bandwidth_multiplier[:, 1] = col_bw_multiplier[col_indexing]

    return aperture_size, bandwidth_multiplier


#########################
# SICD quality degradation functions

def quality_degrade(
        reader, index=0, output_file=None, desired_resolution=None, desired_bandwidth=None,
        desired_nesz=None, **kwargs):
    r"""
    Create a degraded quality SICD based on the desired resolution (impulse response width)
    or bandwidth (impulse response bandwidth), and the desired Noise Equivalent
    Sigma Zero value. The produced SICD will have **uniform weighting**.

    No more than one of `desired_resolution` and `desired_bandwidth` can be provided.
    If None of `desired_resolution`, `desired_bandwidth`, or `desired_nesz` are provided,
    then the SICD will be re-weighted with uniform weighting - even this will
    change the noise and RNIIRS values slightly.

    .. warning::

        Unless `desired_nesz=None`, this will fail for a SICD which is not fully
        Radiometrically calibrated with `'ABSOLUTE'` noise type.


    Parameters
    ----------
    reader : str|BaseReader
    index : int
        The reader index to be used.
    output_file : None|str
        If `None`, an in-memory SICD reader instance will be returned. Otherwise,
        this is the path for the produced output SICD file.
    desired_resolution : None|tuple
        The desired ImpRespWid (Row, Col) tuple, which will be mapped to ImpRespBW
        assuming uniform weighting. You cannot provide both `desired_resolution`
        and `desired_bandwidth`.
    desired_bandwidth : None|tuple
        The desired ImpRespBW (Row, Col) tuple. You cannot provide both
        `desired_resolution` and `desired_bandwidth`.
    desired_nesz : None|float
        The desired Noise Equivalent Sigma Zero value in power units, this is after
        modifications which change the noise due to sub-aperture degradation and/or
        de-weighting.
    kwargs
        Keyword arguments passed through to :func:`sarpy.processing.normalize_sicd.sicd_degrade_reweight`

    Returns
    -------
    None|FlatSICDReader
        No return if `output_file` is provided, otherwise the returns the in-memory
        reader object.
    """

    reader, index = _validate_reader(reader, index)
    if desired_resolution is not None and desired_bandwidth is not None:
        raise ValueError('Both desired_resolution and desired_bandwidth cannot be supplied.')

    sicd = reader.get_sicds_as_tuple()[index]

    if desired_nesz is None:
        add_noise = None
    else:
        current_nesz = get_sigma0_noise(sicd)
        add_noise_factor = (desired_nesz - current_nesz)/current_nesz
        if abs(add_noise_factor) < 1e-5:
            add_noise = None
        elif add_noise_factor < 0:
            raise ValueError(
                'The current nesz value is {},\n\t'
                'the desired nesz value of {} cannot be achieved.'.format(current_nesz, desired_nesz))
        else:
            add_noise = numpy.exp(numpy.log(10)*0.1*sicd.Radiometric.NoiseLevel.NoisePoly[0, 0])*add_noise_factor

    row_aperture, row_bw_factor, column_aperture, column_bw_factor = _map_bandwidth_parameters(
        sicd, desired_resolution=desired_resolution, desired_bandwidth=desired_bandwidth)

    row_weighting, column_weighting = _get_uniform_weight_dicts(sicd)
    return sicd_degrade_reweight(
        reader, output_file=output_file, index=index,
        row_aperture=row_aperture, row_weighting=row_weighting,
        column_aperture=column_aperture, column_weighting=column_weighting,
        add_noise=add_noise, **kwargs)


def quality_degrade_resolution(
        reader, index=0, output_file=None,
        desired_resolution=None, desired_bandwidth=None,
        **kwargs):
    """
    Create a degraded quality SICD based on INCREASING the impulse response width
    to the desired resolution or DECREASING the impulse response bandwidth to the
    desired bandwidth.

    The produced SICD will have uniform weighting.

    Parameters
    ----------
    reader : str|SICDTypeReader
    index : int
        The reader index to be used.
    output_file : None|str
        If `None`, an in-memory SICD reader instance will be returned. Otherwise,
        this is the path for the produced output SICD file.
    desired_resolution : None|tuple
        The desired ImpRespWid (Row, Col) tuple, which will be mapped to ImpRespBW
        assuming uniform weighting. Exactly one of `desired_resolution` and
        `desired_bandwidth` must be provided.
    desired_bandwidth : None|tuple
        The desired ImpRespBW (Row, Col) tuple. Exactly one of `desired_resolution`
        and `desired_bandwidth` must be provided.
    kwargs
        Keyword arguments passed through to :func:`sarpy.processing.normalize_sicd.sicd_degrade_reweight`

    Returns
    -------
    None|FlatSICDReader
        No return if `output_file` is provided, otherwise the returns the in-memory
        reader object.
    """

    return quality_degrade(
        reader, index=index, output_file=output_file,
        desired_resolution=desired_resolution, desired_bandwidth=desired_bandwidth,
        **kwargs)


def quality_degrade_noise(
        reader, index=0, output_file=None, desired_nesz=None, **kwargs):
    """
    Create a degraded quality SICD based on INCREASING the noise to the desired
    Noise Equivalent Sigma Zero value. The produced SICD will have uniform weighting.

    .. warning::

        This will fail for a SICD which is not fully Radiometrically calibrated,
        with ABSOLUTE noise type.

    Parameters
    ----------
    reader : str|SICDTypeReader
    index : int
        The reader index to be used.
    output_file : None|str
        If `None`, an in-memory SICD reader instance will be returned. Otherwise,
        this is the path for the produced output SICD file.
    desired_nesz : None|float
        The desired noise equivalent sigma zero value.
    kwargs
        Keyword arguments passed through to :func:`sarpy.processing.normalize_sicd.sicd_degrade_reweight`

    Returns
    -------
    None|FlatSICDReader
        No return if `output_file` is provided, otherwise the returns the in-memory
        reader object.
    """

    return quality_degrade(reader, index=index, output_file=output_file, desired_nesz=desired_nesz, **kwargs)


def quality_degrade_rniirs(
        reader, index=0, output_file=None, desired_rniirs=None, alpha=0, **kwargs):
    r"""
    Create a degraded quality SICD based on the desired estimated RNIIRS value.
    The produced SICD will have uniform weighting.

    The sicd degradation will be performed as follows:

    - The current information density/current rniirs **with respect to uniform weighting**
      will be found.

    - The information density required to produce the desired rniirs will be found.

    - This will be used, along with the :math:`\alpha` value, will be used in
      :func:`get_bandwidth_noise_distribution` to determine the best feasible
      multipliers for bandwidth and noise values.

    - These desired bandwidth and noise values will then be used in conjunction
      with :func:`sicd_degrade_reweight`.

    .. warning::

        This will fail for a SICD which is not fully Radiometrically calibrated,
        with `'ABSOLUTE'` noise type.

    Parameters
    ----------
    reader : str|SICDTypeReader
    index : int
        The reader index to be used.
    output_file : None|str
        If `None`, an in-memory SICD reader instance will be returned. Otherwise,
        this is the path for the produced output SICD file.
    desired_rniirs : None|float
        The desired rniirs value, according to the RGIQE methodology.
    alpha : float
        This must be a number in the interval [0, 1] defining the (geometric)
        distribution of variability between required influence from increasing
        noise and require influence of decreasing bandwidth.
    kwargs
        Keyword arguments passed through to :func:`sarpy.processing.normalize_sicd.sicd_degrade_reweight`

    Returns
    -------
    None|FlatSICDReader
        No return if `output_file` is provided, otherwise the returns the in-memory
        reader object.
    """

    if desired_rniirs is None:
        return quality_degrade(reader, index=index, output_file=output_file, **kwargs)

    reader, index = _validate_reader(reader, index)
    sicd = reader.get_sicds_as_tuple()[index]
    current_noise = numpy.exp(numpy.log(10)*0.1*sicd.Radiometric.NoiseLevel.NoisePoly[0, 0])

    bandwidth_multiplier, noise_multiplier = get_bandwidth_noise_distribution(
        sicd, alpha, desired_rniirs=desired_rniirs)

    desired_bandwidth = (
        sicd.Grid.Row.ImpRespBW*bandwidth_multiplier[0],
        sicd.Grid.Col.ImpRespBW*bandwidth_multiplier[1])
    add_noise = (noise_multiplier - 1)*current_noise

    if alpha == 0 or add_noise <= 0:
        add_noise = None
    row_aperture, row_bw_factor, column_aperture, column_bw_factor = _map_bandwidth_parameters(
        sicd, desired_bandwidth=desired_bandwidth)
    row_weighting, column_weighting = _get_uniform_weight_dicts(sicd)

    return sicd_degrade_reweight(
        reader, output_file=output_file, index=index,
        row_aperture=row_aperture, row_weighting=row_weighting,
        column_aperture=column_aperture, column_weighting=column_weighting,
        add_noise=add_noise, **kwargs)
