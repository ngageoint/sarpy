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
from sarpy.processing.normalize_sicd import sicd_degrade_reweight, is_uniform_weight
from sarpy.io.complex.converter import open_complex

logger = logging.getLogger(__name__)

RNIIRS_FIT_PARAMETERS = numpy.array([3.7555, .3960], dtype='float64')
"""
The RNIIRS calculation parameters determined by empirical fit 
"""


#####################
# methods for extracting necessary information from the sicd structure

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
    noise = sicd.Radiometric.NoiseLevel.NoisePoly[0, 0]  # this is in db
    noise = 10**(0.1*noise)  # this is absolute

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
    (float, float)
        The information_density and estimated RNIIRS
    """

    information_density = get_information_density(bandwidth_area, signal, noise)
    rniirs = get_rniirs(information_density)
    return information_density, rniirs


def populate_rniirs_for_sicd(sicd, signal=None, noise=None, override=False):
    """
    This populates the value(s) for RNIIRS and information density in the SICD
    structure, according to the RGIQE.

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


def get_bandwidth_noise_distribution(alpha, snr, information_density_ratio):
    r"""
    From the perspective of achieving the desired RNIIRS based on downgrading
    the SICD. This gets the relative multipliers for the bandwidth and nesz values,
    given the distribution parameter, the current snr, and the ratio between the
    desired and current information density.

    The information density is defined as

    .. math::

        inf\_density = bw\_area*\log_2(1 + snr),

    which naturally splits multiplicatively into a piece which can be varied
    based on bandwidth, and a piece which can be varied based purely on varying
    snr (nesz).

    Then, we have

    .. math::

        inf\_density\_ratio & = \frac{desired\_inf\_density}{current\_inf\_density} \\
                                  & = \frac{desired\_bw\_area}{current\_bw\_area} \cdot
                                      \frac{\log_2(1 + desired\_snr)}{\log_2(1 + snr)} \\
                                  & = (bw\_multiplier)^2 \cdot \frac{\log_2(1 + snr/noise\_multiplier)}{\log_2(1 + snr)}

    A one parameter distribution, with free parameter :math:`\alpha \in [0, 1]`, can
    naturally be formed here by setting

    .. math::

        inf\_density\_ratio & = (inf\_density\_ratio)^{1-\alpha}\cdot (inf\_density\_ratio)^{\alpha} \\
        inf\_density\_ratio^{1-\alpha} & = (bw\_multiplier)^2 \\
        inf\_density\_ratio^{\alpha} &= \frac{\log_2(1 + snr/noise\_multiplier)}{\log_2(1 + snr)},

    This allows us to uniquely determine that

    .. math::

        bw\_multiplier &= (inf\_density\_ratio)^{(1-\alpha)/2} \\
        noise\_multiplier &= snr/\left((1 + snr)^{inf\_density\_ratio^{\alpha}} - 1\right)

    which allows for unique determination for values `bw_multiplier` and `noise_multiplier`.

    This establishes a specific distribution of bandwidth and nesz values all providing
    identical RNIIRS value. On one end, at :math:`\alpha = 0`, the bandwidth is decreased
    while noise remains static, and the other extreme end, at :math:`\alpha = 1`,
    the bandwidth is static and the noise is increased.

    Parameters
    ----------
    alpha : float
    snr : float
    information_density_ratio : float

    Returns
    -------
    (float, float)
        The bw_multiplier and noise_multiplier (for nesz).
    """

    def find_bw_multiplier(multiplier):
        return float(numpy.sqrt(multiplier))

    def find_nesz_multiplier(multiplier):
        return snr/(numpy.power(1 + snr, multiplier) - 1)

        # res = minimize_scalar(
        #     lambda x: (numpy.log(1 + snr*x) - multiplier*numpy.log(1 + snr))**2,
        #     bounds=(0, 1),
        #     method='bounded')
        # if not res.success:
        #     raise ValueError('RNIIRS value search for nesz failed')
        # return 1./res.x

    alpha = float(alpha)
    if not (0 <= alpha <= 1):
        raise ValueError('alpha must be in the interval [0, 1], got {}'.format(alpha))

    if alpha == 0:
        return find_bw_multiplier(information_density_ratio), 1.0
    elif alpha == 1:
        return 1.0, find_nesz_multiplier(information_density_ratio)
    else:
        bw_ratio = numpy.power(information_density_ratio, 1-alpha)
        noise_ratio = information_density_ratio/bw_ratio
        return find_bw_multiplier(bw_ratio), find_nesz_multiplier(noise_ratio)


#########################
# helpers for quality degradation function

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
    None|tuple
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
                'After mapping form Desired {} ImpRespWid considering uniform weighting,\n\t'
                'the equivalent desired ImpRespBW is {},\n\t'
                'but the current ImpRespBW is {}'.format(direction, use_bandwidth, current_imp_resp_bw))
        else:
            raise ValueError(
                'Desired {} ImpRespBW is given as {},\n\t'
                'but the current ImpRespBW is {}'.format(direction, use_bandwidth, current_imp_resp_bw))
    elif use_bandwidth == current_imp_resp_bw:
        return None
    else:
        oversample = max(1., 1./(sample_size*use_bandwidth))
        ap_size = round(direction_size/oversample)
        start_ind = int(numpy.floor(0.5*(direction_size - ap_size)))
        return start_ind, start_ind+ap_size


def _determine_additional_noise_amount(desired_nesz, sicd):
    """
    Determine the additional amount of noise (in units of pixel power) required
    to achieve the desired Noise Equivalent Sigma Zero.

    Except in trivial cases, this will raise an exception if the SICd structure
    does not have full radiometric and noise calibration.

    Parameters
    ----------
    desired_nesz : None|float
        The desired noise equivalent sigma zero value.
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        The sicd structure.

    Returns
    -------
    None|float
    """

    if desired_nesz is None:
        return None

    desired_nesz = float(desired_nesz)
    current_nesz = get_sigma0_noise(sicd)  # of the form sigma0*<pixel power>
    # this will have raises an exception if the sicd is improper
    if desired_nesz < current_nesz:
        raise ValueError('The current NESZ is {}, while the desired value is {}'.format(current_nesz, desired_nesz))
    elif desired_nesz == current_nesz:
        return None
    else:
        return (desired_nesz - current_nesz)/sicd.Radiometric.SigmaZeroSFPoly[0, 0]


#########################
# SICD quality degradation functions

def quality_degrade(reader, index=0, output_file=None, desired_resolution=None,
                    desired_bandwidth=None, desired_nesz=None, **kwargs):
    """
    Create a degraded quality SICD based on the desired resolution (impulse response width)
    or bandwidth (impulse response bandwidth), and the desired Noise Equivalent
    Sigma Zero value. The produced SICD will have uniform weighting.

    No more than one of `desired_resolution` and `desired_bandwidth` can be provided.
    If None of `desired_resolution`, `desired_bandwidth`, or `desired_nesz` are provided,
    then the SICD will be re-weighted with uniform weighting.

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
        The desired Noise Equivalent Sigma Zero value.
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
    row_weighting = None if is_uniform_weight(sicd, 0) else \
        {'WindowName': 'UNIFORM', 'WgtFunct': numpy.ones((32,), dtype='float64')}
    column_weighting = None if is_uniform_weight(sicd, 1) else \
        {'WindowName': 'UNIFORM', 'WgtFunct': numpy.ones((32,), dtype='float64')}

    if desired_resolution is not None:
        # get the broadening factor for uniform weighting
        broadening_factor = get_hamming_broadening_factor(1.0)
        row_aperture = _map_desired_resolution_to_aperture(
            sicd.Grid.Row.ImpRespBW, sicd.Grid.Row.SS, 'Row', sicd.ImageData.NumRows,
            desired_resolution=desired_resolution[0], broadening_factor=broadening_factor)
        column_aperture = _map_desired_resolution_to_aperture(
            sicd.Grid.Col.ImpRespBW, sicd.Grid.Col.SS, 'Col', sicd.ImageData.NumCols,
            desired_resolution=desired_resolution[1], broadening_factor=broadening_factor)
    elif desired_bandwidth is not None:
        row_aperture = _map_desired_resolution_to_aperture(
            sicd.Grid.Row.ImpRespBW, sicd.Grid.Row.SS, 'Row', sicd.ImageData.NumRows,
            desired_bandwidth=desired_bandwidth[0])
        column_aperture = _map_desired_resolution_to_aperture(
            sicd.Grid.Col.ImpRespBW, sicd.Grid.Col.SS, 'Col', sicd.ImageData.NumCols,
            desired_bandwidth=desired_bandwidth[1])
    else:
        row_aperture = None
        column_aperture = None

    add_noise = _determine_additional_noise_amount(desired_nesz, sicd)

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
        reader, index=0, output_file=None, desired_rniirs=None,
        alpha=0, **kwargs):
    r"""
    Create a degraded quality SICD based on the desired estimated RNIIRS value.
    The produced SICD will have uniform weighting.

    The information density is defined as
    :math:`inf\_density = bandwidth\_area*\log(1 + signal/nesz)`, which naturally
    splits multiplicatively into a piece which can be varied based on bandwidth,
    and a piece which can be varied based on nesz.

    The icd degradation will be performed as follows.
    - The current information density/current rniirs will be found.
    - The information density required to produce the desired rniirs will be found.
    - The ratio between the two information densities will be calculated,
      :math:`ratio = \frac{required\_inf\_density}{current\_inf\_density}`.
    - This will be used to determine the multipliers for bandwidth and nesz values
    using :func:`get_bandwidth_noise_distribution`.

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

    nesz = get_sigma0_noise(sicd)
    signal = get_default_signal_estimate(sicd)
    bandwidth_area = get_bandwidth_area(sicd)
    current_inf_density = get_information_density(bandwidth_area, signal, nesz)
    current_rniirs = get_rniirs(current_inf_density)

    if current_rniirs < desired_rniirs:
        raise ValueError(
            'The current rniirs is {}, and the desired rniirs is {}'.format(current_rniirs, desired_rniirs))

    # find the information density for the required RNIIRS
    desired_inf_density = find_inf_density()
    the_ratio = desired_inf_density/current_inf_density
    alpha = float(alpha)
    bw_multiplier, nesz_multiplier = get_bandwidth_noise_distribution(alpha, signal/nesz, the_ratio)

    desired_bandwidth = (sicd.Grid.Row.ImpRespBW*bw_multiplier, sicd.Grid.Col.ImpRespBW*bw_multiplier)
    desired_nesz = nesz*nesz_multiplier

    if alpha == 0:
        # signal and noise remain constant, and we vary only bandwidth (ImpRespBW)
        return quality_degrade(
            reader, index=index, output_file=output_file,
            desired_bandwidth=desired_bandwidth, **kwargs)
    elif alpha == 1:
        # bandwidth area remains constant, we vary only the noise
        return quality_degrade(
            reader, index=index, output_file=output_file,
            desired_nesz=desired_nesz, **kwargs)
    else:
        return quality_degrade(
            reader, index=index, output_file=output_file,
            desired_bandwidth=desired_bandwidth, desired_nesz=desired_nesz, **kwargs)
