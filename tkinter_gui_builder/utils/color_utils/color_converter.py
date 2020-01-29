import matplotlib


def hex_to_rgb(hex_color_value):
    return matplotlib.colors.to_rgb(hex_color_value)


def rgb_to_hex(rgb_color_value):
    return matplotlib.colors.to_hex(rgb_color_value)


def hex_list_to_rgb_list(hex_list):
    rgb_list = []
    for hex_value in hex_list:
        rgb_list.append(hex_to_rgb(hex_value))
    return rgb_list


def rgb_list_to_hex_list(rgb_list):
    hex_list = []
    for rgb_value in rgb_list:
        hex_list.append(rgb_to_hex(rgb_value))
    return hex_list
