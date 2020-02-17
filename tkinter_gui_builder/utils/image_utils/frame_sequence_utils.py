import PIL.Image
from numpy import ndarray


def save_numpy_frame_sequence_to_animated_gif(frame_sequence,  # type: [ndarray]
                                              fname,  # type: str
                                              fps=15,  # type: float
                                              loop_animation=True
                                              ):
    duration = (1 / fps) * 1000
    pil_frame_sequence = []
    for frame in frame_sequence:
        pil_frame_sequence.append(PIL.Image.fromarray(frame))
    if loop_animation:
        pil_frame_sequence[0].save(fname,
                                   save_all=True,
                                   append_images=pil_frame_sequence[1:],
                                   optimize=True,
                                   duration=duration,
                                   loop=0)
    else:
        pil_frame_sequence[0].save(fname,
                                   save_all=True,
                                   append_images=pil_frame_sequence[1:],
                                   optimize=True,
                                   duration=duration)
