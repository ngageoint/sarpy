#!/usr/bin/env python

import json

from . import mb
from flask import render_template
from flask import request

import os
from algorithm_toolkit import app

from sarpy.deprecated.tools import FrameGenerator
from sarpy.deprecated.tools import FrameForm

cam = FrameGenerator()


@app.route('/aperture_tool/')
def index():
    form = FrameForm()
    """Image Blending home page."""
    return render_template('index.html', form=form)


@mb.route('/aperture_tool/update_image_path', methods=['POST'])
def set_image_path():

    image_path = os.path.normpath(request.values.get('image_path', ''))
    tnx = int(request.values.get('tnx', ''))
    tny = int(request.values.get('tny', ''))

    nx, ny = cam.set_image_path(image_path, tnx, tny)

    return json.dumps({'nx': nx, 'ny': ny})


@mb.route('/aperture_tool/update_image_content', methods=['POST'])
def crop_image():

    minx = int(request.values.get('minx', ''))
    maxx = int(request.values.get('maxx', ''))
    miny = int(request.values.get('miny', ''))
    maxy = int(request.values.get('maxy', ''))
    tnx = int(request.values.get('tnx', ''))
    tny = int(request.values.get('tny', ''))

    cam.crop_image(minx, miny, maxx, maxy, tnx, tny)

    return ''


@mb.route('/aperture_tool/ortho_image', methods=['POST'])
def ortho_image():

    output_image_path = os.path.normpath(request.values.get('input', ''))
    cam.ortho_image(output_image_path)

    return ''


@mb.route('/aperture_tool/get_frame', methods=['POST'])
def get_image():
    return cam.get_frame()
