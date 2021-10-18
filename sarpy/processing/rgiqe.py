"""
Radar Generalized Image Quality Equation (RGIQE) calculation(s) and tools for
application to SICD structures and files.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging

import numpy
from scipy.optimize import minimize_scalar

from sarpy.io.complex.base import SICDTypeReader
from sarpy.processing.windows import get_hamming_broadening_factor
from sarpy.processing.normalize_sicd import sicd_degrade_reweight, is_uniform_weight, \
    get_resultant_noise
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


def _get_sigma0_deweighted(sicd):
    """
    Calculate the Sigma Zero SF for a deweighted version of the sicd.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    float
    """

    sigma0_sf = sicd.Radiometric.RCSSFPoly[0, 0] * \
                (sicd.Grid.Row.ImpRespBW * sicd.Grid.Col.ImpRespBW) / \
                numpy.cos(numpy.deg2rad(sicd.SCPCOA.SlopeAng))
    return sigma0_sf


def get_sigma0_noise_with_uniform_weighting(sicd):
    """
    Gets both the current nesz and nesz after uniform weighting **in power units**.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    current_nesz : float
        In power units.
    reweighted_nesz : float
        In power units.
    """

    _verify_sicd_with_noise(sicd)
    current_noise, reweighted_noise = _get_current_and_reweighted_noise(sicd)
    sigma0_sf = _get_sigma0_deweighted(sicd)
    return current_noise*sigma0_sf, reweighted_noise*sigma0_sf


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


def get_bandwidth_noise_distribution(bandwidth_area, snr, alpha, desired_information_density):
    r"""
    This is a function which defines a distribution of SICD degradation
    parameters to achieve the desired information density/rniirs. This gets the
    relative multipliers for the bandwidth and nesz values, given the current
    bandwidth area, the current snr, a distribution parameter, and the desired
    information density.

    The information density is defined as

    .. math::

        inf\_density = bw\_area\cdot\log_2(1 + snr).

    If we use sub-aperture degradation to reduce the bandwidth area and add noise
    (before the sub-aperture degradation), we get

    .. math::

        inf\_density\_new = bw\_area \cdot bw\_mult^2\cdot\log_2\left(1 + \frac{snr}{bw\_mult^2\cdot noise\_mult}\right)

    The resultant noise (in unit of power) AFTER processing will be

    .. math::

        new\_noise & = noise\cdot bw\_mult^2\cdot noise\_mult \text{(in power)}\\
        new\_nesz &= nesz + 2\log_{10}(bw\_mult) + \log_{10}(noise\_mult) \text{(in dB)}

    Holding the value for :math:`inf\_density\_new` constant defines a distribution
    on :math:`bw\_mult` and :math:`noise\_mult` to obtain the desired information
    density/rniirs. There is significant interplay between the values for :math:`bw\_mult` and
    :math:`noise\_mult`, but it can be simplified to a one parameter distribution
    based on :math:`\alpha \in [0, 1]`. There is a natural continuum of consideration,
    which we define as :math:`\alpha=0`, we add no additional noise
    (:math:`noise\_mult=1`) and use purely sub-aperture degradation. On the other
    end, which we define as :math:`\alpha=1`, we have :math:`bw\_mult=1`, so that

    .. math::

        inf\_density\_new &= bw\_area\cdot\log_2\left(1 + \frac{snr}{max\_noise\_mult}\right) \\
        max\_noise\_mult & = \frac{snr}{2^{inf\_density\_new/bw\_area} - 1}

    Then, given :math:`\alpha \in [0, 1]`, our distribution is defined by setting
    :math:`noise\_mult = (max\_noise\_mult - 1)\cdot\alpha + 1` and then determining
    the unique :math:`bw\_mult` which fulfills our requirement

    .. math::

        inf\_density\_new = bw\_area \cdot bw\_mult^2\cdot\log_2\left(1 + \frac{snr}{bw\_mult^2\cdot noise\_mult}\right)

    To visualize the noise and bandwidth area as functions of :math:`\alpha` determined
    by this function, then just use the fundamental definitions.

    .. math::

        new\_bandwidth\_area(\alpha) &= bandwidth\_area\cdot (bw\_mult(\alpha))^2 \\
        new\_noise(\alpha) &= noise\cdot (bw\_mult(\alpha))^2\cdot noise\_mult(\alpha) \text{(in power)}\\
        new\_nesz(\alpha) &= nesz + 2\log_{10}(bw\_mult(\alpha)) + \log_{10}(noise\_mult(\alpha)) \text{(in dB)}


    Parameters
    ----------
    bandwidth_area : float
    snr : float
    alpha : float
    desired_information_density : float

    Returns
    -------
    (float, float)
        The bandwidth multiplier and noise multiplier
    """

    # todo: this is not the swiftest implementation - is it worth trying to speed up?

    def minimize_function(bandwidth_mult):
        sq_factor = bandwidth_mult*bandwidth_mult
        new_bw_area = bandwidth_area*sq_factor
        new_snr = snr/(sq_factor*noise_multiplier)
        diff = desired_information_density - get_information_density(new_bw_area, new_snr, 1)
        return diff*diff

    current_inf_density = get_information_density(bandwidth_area, snr, 1)
    if desired_information_density >= current_inf_density:
        raise ValueError(
            'current information density is {}, this SICD can not be degraded\n\t'
            'to obtain a result with information density {}'.format(
                current_inf_density, desired_information_density))

    if not (0 <= alpha <= 1):
        raise ValueError('alpha must be in the interval [0, 1], got {}'.format(alpha))

    max_signal_multiplier = snr/(2**(desired_information_density/bandwidth_area) - 1)

    noise_multiplier = (max_signal_multiplier - 1.0)*alpha + 1.0

    res = minimize_scalar(
        minimize_function,
        bounds=(1e-6, 1),
        method='bounded',
        options={'xatol': 1e-8, 'maxiter': 1000, 'disp': 0})
    if not res.success:
        raise ValueError('information density value search for bandwidth multiplier failed')
    bw_multiplier = res.x

    if alpha == 0:
        return bw_multiplier, 1
    elif alpha == 1:
        return 1, noise_multiplier
    else:
        return bw_multiplier, noise_multiplier


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


def _determine_additional_noise_amount(desired_nesz, total_bw_factor, current_noise, reweighted_noise, sicd):
    """
    Determine the additional amount of noise (in units of pixel power) required
    to achieve the desired Noise Equivalent Sigma Zero.

    Except in trivial cases, this will raise an exception if the SICD structure
    does not have full radiometric and noise calibration.

    Parameters
    ----------
    desired_nesz : None|float
        The desired noise equivalent sigma zero value.
    total_bw_factor : float
        The relative size of the bandwidth area after implied sup-aperture degradation.
    current_noise : float
    reweighted_noise : float
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        The sicd structure.

    Returns
    -------
    None|float
        The amount of noise to add, before re-weighting

    Raises
    ------
    ValueError
    """

    if desired_nesz is None:
        return None

    desired_nesz = float(desired_nesz)
    current_nesz = reweighted_noise*sicd.Radiometric.SigmaZeroSFPoly[0, 0]
    required_nesz = desired_nesz/total_bw_factor
    additional_factor = (required_nesz - current_nesz)/current_nesz
    if additional_factor < 0:
        raise ValueError('The desired NESZ is too small to be achieved using the given sub-aperture processing scheme')
    elif additional_factor <= 1e-5:
        return None
    else:
        return current_noise*additional_factor


def _get_current_and_reweighted_noise(sicd):
    """
    Gets both the current noise in units of power, and the noise after uniform
    weighting in units of power.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    noise : float
    reweighted_noise : float
    """

    current_noise = numpy.exp(numpy.log(10)*0.1*sicd.Radiometric.NoiseLevel.NoisePoly[0, 0])  # in power
    reweighted_noise = get_resultant_noise(
        sicd,
        row_weight_funct=numpy.ones((32, ), dtype='float32'),
        column_weight_funct=numpy.ones((32, ), dtype='float32'))
    return current_noise, numpy.exp(numpy.log(10)*0.1*reweighted_noise)


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


#########################
# SICD quality degradation functions

def quality_degrade(reader, index=0, output_file=None, desired_resolution=None,
                    desired_bandwidth=None, desired_nesz=None, **kwargs):
    r"""
    Create a degraded quality SICD based on the desired resolution (impulse response width)
    or bandwidth (impulse response bandwidth), and the desired Noise Equivalent
    Sigma Zero value. The produced SICD will have **uniform weighting**.

    No more than one of `desired_resolution` and `desired_bandwidth` can be provided.
    If None of `desired_resolution`, `desired_bandwidth`, or `desired_nesz` are provided,
    then the SICD will be re-weighted with uniform weighting - even this will
    change the noise and RNIIRS values slightly.

    .. note::

        The de-weighting and sub-aperture degradation involved in setting the
        desired resolution or bandwidth naturally changes the magnitude of the
        per pixel noise. A SICD produced from such processing will naturally
        have a different nesz value (and slightly pink/colored noise), even if it
        is not augmented by the addition of extra noise.

    The current `reweighted_nesz` in power units can be obtained from
    :func:`get_sigma0_noise_with_uniform_weighting`.

    Assuming that the current Row/Col ImpRespBW is given in tuple `(row_bw, col_bw)`,
    the desired Row/Col ImpRespBW is provided as `(new_row_bw, new_col_bw)`,
    and the current reweighted noise equivalent sigma zero value is given as `reweighted_nesz`,
    then the resultant noise will be given as

    .. math::

        resultant\_nesz = reweighted\_nesz*\frac{new\_row\_bw}{row\_bw}*\frac{new\_col\_bw}{col\_bw}

    This is the default resultant noise, and the lower bound for `desired_nesz`.

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
        raise ValueError('Both desire_resolution and desired_bandwidth cannot be supplied.')

    sicd = reader.get_sicds_as_tuple()[index]
    row_weighting, column_weighting = _get_uniform_weight_dicts(sicd)

    row_aperture, row_bw_factor, column_aperture, column_bw_factor = _map_bandwidth_parameters(
        sicd, desired_resolution=desired_resolution, desired_bandwidth=desired_bandwidth)
    current_noise, reweighted_noise = _get_current_and_reweighted_noise(sicd)
    add_noise = _determine_additional_noise_amount(desired_nesz, row_bw_factor*column_bw_factor, current_noise, reweighted_noise, sicd)

    return sicd_degrade_reweight(
        reader, output_file=output_file, index=index,
        row_aperture=row_aperture, row_weighting=row_weighting,
        column_aperture=column_aperture, column_weighting=column_weighting,
        add_noise=add_noise, **kwargs)


def quality_degrade_resolution(reader, index=0, output_file=None,
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


def quality_degrade_noise(reader, index=0, output_file=None, desired_nesz=None, **kwargs):
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
      :func:`get_bandwidth_noise_distribution` to determine the multipliers for
      bandwidth and nesz values.

    - These desired bandwidth and noise values will then be used in conjunction
      with :func:`sicd_degrade_reweight`.

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

    def find_inf_density():
        res = minimize_scalar(
            lambda x: (desired_rniirs - get_rniirs(x))**2,
            bounds=(0, current_inf_density),
            method='bounded')
        if not res.success:
            raise ValueError('RNIIRS value search for information density failed')
        return float(res.x)

    if desired_rniirs is None:
        return quality_degrade(reader, index=index, output_file=output_file, **kwargs)

    reader, index = _validate_reader(reader, index)
    sicd = reader.get_sicds_as_tuple()[index]

    bandwidth_area = get_bandwidth_area(sicd)
    signal = get_default_signal_estimate(sicd)
    current_noise, reweighted_noise = _get_current_and_reweighted_noise(sicd)
    sigma0_sf = _get_sigma0_deweighted(sicd)
    nesz = reweighted_noise*sigma0_sf

    current_inf_density = get_information_density(bandwidth_area, signal, nesz)
    current_rniirs = get_rniirs(current_inf_density)

    if current_rniirs < desired_rniirs:
        raise ValueError(
            'The current rniirs (after uniform weighting) is {},\n\t'
            'and the desired rniirs is {}'.format(current_rniirs, desired_rniirs))

    # find the information density for the required RNIIRS
    desired_inf_density = find_inf_density()
    alpha = float(alpha)
    bw_multiplier, nesz_multiplier = get_bandwidth_noise_distribution(
        bandwidth_area, signal/nesz, alpha, desired_inf_density)

    desired_bandwidth = (sicd.Grid.Row.ImpRespBW*bw_multiplier, sicd.Grid.Col.ImpRespBW*bw_multiplier)
    if nesz_multiplier < (1 - 1e-5):
        raise ValueError('Got negative required additional noise.')
    elif nesz_multiplier < (1 + 1e-5):
        add_noise = None  # to overcome imprecision nonsense
    else:
        add_noise = current_noise*(nesz_multiplier - 1)

    row_weighting, column_weighting = _get_uniform_weight_dicts(sicd)

    row_aperture, row_bw_factor, column_aperture, column_bw_factor = _map_bandwidth_parameters(
        sicd, desired_bandwidth=desired_bandwidth)

    if alpha == 0:
        # signal and noise remain constant, and we vary only bandwidth (ImpRespBW)
        return sicd_degrade_reweight(
            reader, output_file=output_file, index=index,
            row_aperture=row_aperture, row_weighting=row_weighting,
            column_aperture=column_aperture, column_weighting=column_weighting,
            add_noise=None, **kwargs)
    elif alpha == 1:
        # bandwidth area remains constant, we vary only the noise
        return sicd_degrade_reweight(
            reader, output_file=output_file, index=index,
            row_weighting=row_weighting,
            column_weighting=column_weighting,
            add_noise=add_noise, **kwargs)
    else:
        return sicd_degrade_reweight(
            reader, output_file=output_file, index=index,
            row_aperture=row_aperture, row_weighting=row_weighting,
            column_aperture=column_aperture, column_weighting=column_weighting,
            add_noise=add_noise, **kwargs)
