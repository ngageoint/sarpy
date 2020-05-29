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

__author__ = ('Clayton Williams', 'Wade Schwartzkopf', 'Tom Braun')
__classification__ = "UNCLASSIFIED"


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

    Parameters
    ----------
    ci : numpy.ndarray
        the complex data array
    frames : int
        number of frames
    apfraction : float
        Fraction of aperture for each subaperture.
    method : str
        One of ['NORMAL', 'FULLPIXEL', 'MINIMAL'].
    platformdir : str
        Platform direction one of ['RIGHT', 'LEFT'].
    dim : int
        Dimension over which to split subaperture.
    fill : int|float
        Fill factor.
    offset_pct : float
        In the range [0, 1]. Only used for single frame at a certain aperture offset.
    selected_frames : list|tuple
        subcollection of [0,...,`frames`-1] determining subset of frames to be used.

    Returns
    -------
    list
        the list of frame data

    .. Note:: All desired frames will be held in memory at once.
    .. Warning:: The input array is assumed to have been de-skewed already.
    """

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

    Parameters
    ----------
    ph: numpy.ndarray
        the complex data array
    frames : int
        number of frames
    apfraction : float
        Fraction of aperture for each subaperture.
    method : str
        One of ['NORMAL', 'FULLPIXEL', 'MINIMAL'].
    platformdir : str
        Platform direction one of ['RIGHT', 'LEFT'].
    dim : int
        Dimension over which to split subaperture.
    fill : int|float
        Fill factor.
    offset_pct : float
        In the range [0, 1]. Only used for single frame at a certain aperture offset.
    selected_frames : list|tuple
        subcollection of [0,...,`frames`-1] determining subset of frames to be used.

    Returns
    -------
    list
        the list of frame data

    .. Note:: All desired frames will be held in memory at once.
    .. Warning:: Assumes that the data has had an FFTSHIFT applied to it -
        DC is in the center, not at index 0.
    """

    # Parse and validate arguments if not already done
    if not np.iscomplexobj(ph) or ph.ndim > 2:
        raise(TypeError('Input image must be a single complex image.'))

    # Setup for the subaperture processing
    if dim == 0:  # simple
        ph = ph.T  # this is just a view

    if selected_frames is None or frames == 1:
        selected_frames = list(range(0, frames))
    elif not isinstance(selected_frames, (list, tuple)):
        selected_frames = list(selected_frames)

    nxfftOrig = ph.shape[1]
    left_edge = int(np.round(nxfftOrig * (1 - (1 / max(fill, 1))) / 2))  # Where left edge starts
    nxfft = nxfftOrig - (2*left_edge)
    if method.upper() == 'MINIMAL':
        output_res = np.ceil(nxfft / frames)
    elif method.upper() == 'FULLPIXEL':
        output_res = nxfftOrig
    else:
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
        result.append(np.fft.ifft(ph[:, int(np.round(offset + (step * frame_num))):int(np.round(offset + (step * frame_num) + num_sa))],
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
    Calculates the subaperture-processed image on complex data that is held in memory, with the
    fill and platform direction from SICD metadata structure.

    Parameters
    ----------
    ci : numpy.array
    sicd_meta : sarpy.io.complex.sicd_elements.SICD.SICDType
    frames : int
        number of frames
    apfraction : float
        Fraction of aperture for each subaperture.
    method : str
        One of ['NORMAL', 'FULLPIXEL', 'MINIMAL'].
    dim : int
        Dimension over which to split subaperture.
    offset_pct : float
        In the range [0, 1]. Only used for single frame at a certain aperture offset.
    selected_frames : list|tuple
        subcollection of [0,...,`frames`-1] determining subset of frames to be used.

    Returns
    -------
    list
        the list of frame data

    .. Warning: The input array is passed staright through to mem, and so is
        assumed to have been de-skewed already.
    """

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