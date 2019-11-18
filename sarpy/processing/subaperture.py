"""Subaperture processing methods"""

import numpy as np

# Default parameters for a variety of methods
_DEFAULT_FRAMES = 9
_DEFAULT_APFRACTION = 0.2
_DEFAULT_METHOD = 'FULLPIXEL'
_DEFAULT_DIR = 'RIGHT'
_DEFAULT_DIM = 1
_DEFAULT_FILL = 1
_DEFAULT_OFFSETPCT = None
_DEFAULT_SELECTEDFRAMES = None


__classification__ = "UNCLASSIFIED"


# TODO: MEDIUM - overall high level question: where would we get complex data without sicd metadata?
#   It really seems like we should have a CLASS which extracts sicd metadata and provides
#   relevant reading functionality. This mirrors the h5py functionality.
#   We could speedify things using simple cython. This comment probably belongs in io.complex package...


def mem(ci,
        frames=_DEFAULT_FRAMES,
        apfraction=_DEFAULT_APFRACTION,
        method=_DEFAULT_METHOD,
        platformdir=_DEFAULT_DIR,
        dim=_DEFAULT_DIM,
        fill=_DEFAULT_FILL,
        offset_pct=_DEFAULT_OFFSETPCT,
        selected_frames=_DEFAULT_SELECTEDFRAMES):
    """
    Calculates the subaperture-processed image on complex data that is held in memory.

    :param ci: Complex data. ASSUMED TO BE DE-SKEWED.
    :param frames: Number of frames.
    :param apfraction: Fraction of aperture for each subaperture.
    :param method: One of ['NORMAL', 'FULLPIXEL', 'MINIMAL'].
    :param platformdir: Platform direction one of ['RIGHT', 'LEFT'].
    :param dim: Dimension over which to split subaperture.
    :param fill: Fill factor.
    :param offset_pct: Float in [0, 1]. Only used for single frame at a certain aperture offset.
    :param selected_frames: Iterable subcollection of [0,...,`frames`-1] determining subset of frames to be used.
    :return: List of frame data

    .. Note: All desired frames will be held in memory at once.
    """

    # TODO: HIGH - unit test. This naming scheme continues to be undesirable.
    #   Why not just have an optional argument of mem_ph for doing the shift mumbo jumbo?

    return mem_ph(np.fft.fftshift(np.fft.fft(ci, axis=dim), axes=dim),
                  frames=frames,
                  apfraction=apfraction,
                  method=method,
                  platformdir=platformdir,
                  dim=dim,
                  fill=fill,
                  offset_pct=offset_pct,
                  selected_frames=selected_frames)


def mem_ph(ph,
           frames=_DEFAULT_FRAMES,
           apfraction=_DEFAULT_APFRACTION,
           method=_DEFAULT_METHOD,
           platformdir=_DEFAULT_DIR,
           dim=_DEFAULT_DIM,
           fill=_DEFAULT_FILL,
           offset_pct=_DEFAULT_OFFSETPCT,
           selected_frames=_DEFAULT_SELECTEDFRAMES):
    """
    Calculates the subaperture-processed image on phase history data that is held in memory.

    :param ph: phase history data. Assumes an FFTSHIFT has been applied, so that DC is in the center, not at index 0.
    :param frames: number of frames
    :param apfraction: fraction of aperture for each subaperture
    :param method: one of ['normal', 'fullpixel', 'minimal']
    :param platformdir: platform direction one of ['right', 'left']
    :param dim: dimension over which to split subaperture
    :param fill: fill factor
    :param offset_pct: float in [0, 1]. Only used for single frame at a certain aperture offset.
    :param selected_frames: iterable subcollection of [0,...,`frames`-1] determining subset of frames to be used.
    :return: list of frame data

    .. Note: All desired frames will be held in memory at once.
    """

    # TODO: HIGH - unit test. This naming scheme continues to be undesirable.
    #   Is it worth returning a generator to alleviate the everything in memory at once scenario? What is the use case?

    # Parse and validate arguments if not already done
    if not np.iscomplexobj(ph) or ph.ndim > 2:
        raise(TypeError('Input image must be a single complex image.'))

    # Setup for the subaperture processing
    if dim == 0:  # simple
        ph = np.transpose(ph)  # TODO: HIGH - this is gross

    if selected_frames is None or frames == 1:
        selected_frames = range(0, frames)  # TODO: HIGH - this is a generator in Python3
    elif type(selected_frames) != list:
        selected_frames = list(selected_frames)  # TODO: MEDIUM - why do this?

    nxfftOrig = ph.shape[1]
    left_edge = int(np.round(nxfftOrig * (1 - (1 / max(fill, 1))) / 2))  # Where left edge starts
    nxfft = nxfftOrig - (2*left_edge)
    if method.upper() == 'MINIMAL':
        output_res = np.ceil(nxfft / frames)
    elif method.upper() == 'FULLPIXEL':
        output_res = nxfftOrig
    else:
        # TODO: LOW - it's probably worth fleshing out the full list, unless we are absolutely positive
        #   that it will never change
        output_res = np.ceil(apfraction*nxfftOrig)

    result = []
    num_sa = np.ceil(apfraction * nxfft)
    if frames != 1:
        step = np.floor((nxfft - num_sa) / (frames - 1))
    else:
        step = 0
    if frames == 1 and offset_pct is not None:  # Use offset_pct, rather than frames
        if dim == 1 and platformdir.upper() == 'RIGHT':
            offset_pct = 1 - offset_pct
        offset = left_edge + np.floor((nxfft - num_sa) * offset_pct)
    else:
        offset = left_edge + np.floor((nxfft - (step * (frames - 1) + num_sa)) / 2)

    # Run the subaperture processing.  Done a frame at a time instead of by line -- either works
    for f in selected_frames:
        if dim == 1 and platformdir.upper() == 'RIGHT':
            frame_num = frames - f - 1
        else:
            frame_num = f
        result.append(np.fft.ifft(ph[:, int(np.round(offset + (step * frame_num))):
                                     int(np.round(offset + (step * frame_num) + num_sa))],
                      n=int(output_res)))
        if dim == 0:
            result[-1] = result[-1].transpose()

    return result


def mem_sicd(ci, sicd_meta,
             frames=_DEFAULT_FRAMES,
             apfraction=_DEFAULT_APFRACTION,
             method=_DEFAULT_METHOD,
             dim=_DEFAULT_DIM,
             offset_pct=_DEFAULT_OFFSETPCT,
             selected_frames=_DEFAULT_SELECTEDFRAMES):
    """
    Compute subaperture processing parameters, extracting fill and platform direction from SICD metadata structure.
    :param ci: complex data. ASSUMED TO BE DE-SKEWED.
    :param sicd_meta: the sicd metadata structure.
    :param frames: number of frames.
    :param apfraction: fraction of aperture for each subaperture.
    :param method: one of ['normal', 'fullpixel', 'minimal'].
    :param dim: dimension over which to split subaperture.
    :param offset_pct: float in [0, 1]. Only used for single frame at a certain aperture offset.
    :param selected_frames: iterable subcollection of [0,...,`frames`-1] determining subset of frames to be used.
    :return: All desired frames will be held in memory at once.

    .. Note: All desired frames will be held in memory at once.
    """
    # TODO: HIGH - This should not exist.
    #   `sicd_meta` should be an keyword argument of mem_ph, and fill/platformdir overridden if not `None`

    if dim == 1:
        fill = 1/(sicd_meta.Grid.Col.SS * sicd_meta.Grid.Col.ImpRespBW)
    else:
        fill = 1/(sicd_meta.Grid.Row.SS * sicd_meta.Grid.Row.ImpRespBW)
    if sicd_meta.SCPCOA.SideOfTrack == 'L':
        platformdir = 'left'
    else:
        platformdir = 'right'

    return mem(ci,
               frames=frames,
               apfraction=apfraction,
               method=method,
               dim=dim,
               offset_pct=offset_pct,
               selected_frames=selected_frames,
               fill=fill,
               platformdir=platformdir)
