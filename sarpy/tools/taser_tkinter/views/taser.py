#!/usr/bin/env python

import json


import os
from sarpy.tools.taser_tkinter.utils.frame_generator import FrameGenerator


cam = FrameGenerator()


# @mb.route('/taser/update_image_path', methods=['POST'])
def set_image_path(image_fname, tnx, tny):

    image_path = os.path.normpath(image_fname)

    nx, ny = cam.set_image_path(image_path, tnx, tny)

    return  nx, ny

#
# # @mb.route('/taser/update_image_content', methods=['POST'])
# def crop_image():
#
#     minx = int(round(float((request.values.get('minx', '')))))
#     maxx = int(round(float((request.values.get('maxx', '')))))
#     miny = int(round(float((request.values.get('miny', '')))))
#     maxy = int(round(float((request.values.get('maxy', '')))))
#     tnx = int(round(float((request.values.get('tnx', '')))))
#     tny = int(round(float((request.values.get('tny', '')))))
#
#     cam.crop_image(minx, miny, maxx, maxy, tnx, tny)
#
#     return ''
#
#
# # @mb.route('/taser/ortho_image', methods=['POST'])
# def ortho_image():
#
#     output_image_path = os.path.normpath(request.values.get('input', ''))
#     cam.ortho_image(output_image_path)
#
#     return ''
#
#
# # @mb.route('/taser/get_frame', methods=['POST'])
# def get_image():
#     return cam.get_frame()
