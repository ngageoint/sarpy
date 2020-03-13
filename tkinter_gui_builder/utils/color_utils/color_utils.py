import numpy as np
import math
import tkinter_gui_builder.utils.color_utils.color_converter as color_converter


def get_full_rgb_palette(rgb_palette,           # type: [[float]]
                         n_colors=None,         # type: int
                         ):                     # type: (...) -> [[float]]
    if n_colors is None:
        n_colors = len(rgb_palette)
    color_array = []
    n_color_bins = len(rgb_palette)
    indices = np.linspace(0, n_colors, n_colors)
    for i in indices:
        index = i / n_colors * (n_color_bins - 1)
        low = int(index)
        high = int(math.ceil(index))
        interp_float = index - low
        color_array.append(list(np.array(rgb_palette[low]) * (1 - interp_float) + np.array(rgb_palette[high]) * interp_float))
    return color_array


def get_full_hex_palette(hex_palette,       # type: []
                         n_colors=None,     # type: int
                         ):
    rgb_palette = color_converter.hex_list_to_rgb_list(hex_palette)
    rgb_full = get_full_rgb_palette(rgb_palette, n_colors)
    hex_full = color_converter.rgb_list_to_hex_list(rgb_full)
    return hex_full
