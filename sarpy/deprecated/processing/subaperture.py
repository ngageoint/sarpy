'''Module for doing subaperture processing.'''

# Imports
import numpy as np

# Define these so all varieties of function use the same defaults
_DEFAULT_FRAMES = 9
_DEFAULT_APFRACTION = 0.2
_DEFAULT_METHOD = 'fullpixel'
_DEFAULT_DIR = 'right'
_DEFAULT_DIM = 1
_DEFAULT_FILL = 1
_DEFAULT_OFFSETPCT = None
_DEFAULT_SELECTEDFRAMES = None

__author__ = ['Clayton Williams', 'Wade Schwartzkopf', 'Tom Braun']
__classification__ = 'UNCLASSIFIED'
__email__ = 'Clayton.S.Williams@aero.org'


def mem(ci, frames=_DEFAULT_FRAMES, apfraction=_DEFAULT_APFRACTION, method=_DEFAULT_METHOD,
        platformdir=_DEFAULT_DIR, dim=_DEFAULT_DIM, fill=_DEFAULT_FILL,
        offset_pct=_DEFAULT_OFFSETPCT, selected_frames=_DEFAULT_SELECTEDFRAMES):
    '''Calculates the subaperture-processed image on complex data that is held in memory.

           Keyword           Description
           frames            number of frames (default = 9)
           apfraction        fraction of aperture for each subaperture (default = .2)
           method            'normal', 'fullpixel' (default), or 'minimal'
           platformdir       platform direction, 'right' (default) or 'left'
           dim               dimension over which to split subaperture (default = 1)
           fill              fill factor (default = 1)
           offset_pct        used when only a single frame is requested at a certain aperture
                                offset. Float from 0-1.
           selected_frames   used when a subset of frames is desired rather than the entire
                                sequence. Caution: The first frame starts at 0, so the standard
                                set would be [0,1,2,3,4,5,6,7,8]. None implies all frames are desired.

    Output is stored in a list, where each element of the array is a frame.

    Limitation: Assumes complex data and all desired frames can be held in
    memory at once. Assumes complex data is de-skewed.
    '''
    return mem_ph(np.fft.fftshift(np.fft.fft(ci, axis=dim), axes=dim),
                  frames, apfraction, method, platformdir, dim, fill, offset_pct, selected_frames)


def mem_ph(ph, frames=_DEFAULT_FRAMES, apfraction=_DEFAULT_APFRACTION, method=_DEFAULT_METHOD,
           platformdir=_DEFAULT_DIR, dim=_DEFAULT_DIM, fill=_DEFAULT_FILL,
           offset_pct=_DEFAULT_OFFSETPCT, selected_frames=_DEFAULT_SELECTEDFRAMES):
    '''Calculates the subaperture-processed image on phase history data that is held in memory.

    Assumes the data has had an FFTSHIFT applied to it (DC is in the center, not at index 0.)

    See mem function for documentation of keyword parameters.
    '''
    # Parse and validate arguments if not already done
    if not np.iscomplexobj(ph) or ph.ndim > 2:
        raise(TypeError('Input image must be a single complex image.'))

    # Setup for the subaperture processing
    if dim == 0:  # simple
        ph = np.transpose(ph)

    if selected_frames is None or frames == 1:
        selected_frames = range(0, frames)
    elif type(selected_frames) != list:
        selected_frames = list(selected_frames)

    nxfftOrig = ph.shape[1]
    left_edge = int(np.round(nxfftOrig * (1 - (1 / max(fill, 1))) / 2))  # Where left edge starts
    nxfft = nxfftOrig - (2*left_edge)
    if method == 'minimal':
        output_res = np.ceil(nxfft / frames)
    elif method == 'fullpixel':
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
        if dim == 1 and platformdir == 'right':
            offset_pct = 1 - offset_pct
        offset = left_edge + np.floor((nxfft - num_sa) * offset_pct)
    else:
        offset = left_edge + np.floor((nxfft - (step * (frames - 1) + num_sa)) / 2)

    # Run the subaperture processing.  Done a frame at a time instead of by line -- either works
    for f in selected_frames:
        if dim == 1 and platformdir == 'right':
            frame_num = frames - f - 1
        else:
            frame_num = f
        result.append(np.fft.ifft(ph[:, int(np.round(offset + (step * frame_num))):
                                     int(np.round(offset + (step * frame_num) + num_sa))],
                      n=int(output_res)))
        if dim == 0:
            result[-1] = result[-1].transpose()

    return result


def mem_sicd(ci, sicd_meta, frames=_DEFAULT_FRAMES, apfraction=_DEFAULT_APFRACTION,
             method=_DEFAULT_METHOD, dim=_DEFAULT_DIM, offset_pct=_DEFAULT_OFFSETPCT,
             selected_frames=_DEFAULT_SELECTEDFRAMES):
    '''Demonstrates how to compute subaperture processing parameters from SICD metadata structure.

    See mem function for documentation of keyword parameters.
    '''
    if dim == 1:
        fill = 1/(sicd_meta.Grid.Col.SS * sicd_meta.Grid.Col.ImpRespBW)
    else:
        fill = 1/(sicd_meta.Grid.Row.SS * sicd_meta.Grid.Row.ImpRespBW)
    if sicd_meta.SCPCOA.SideOfTrack == 'L':
        platformdir = 'left'
    else:
        platformdir = 'right'

    return mem(ci, frames=frames, apfraction=apfraction, method=method, dim=dim,
               offset_pct=offset_pct, selected_frames=selected_frames, fill=fill,
               platformdir=platformdir)
