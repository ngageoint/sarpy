#!/usr/bin/env python

import time
import json

from . import mb
from flask import render_template, Response
from flask import request

import os
from algorithm_toolkit import app

from sarpy.atk.atk_interactive.utils.frame_generator import FrameGenerator
from sarpy.atk.atk_interactive.forms import FrameForm

cam = FrameGenerator()


@app.route('/taser/')
def index():
    form = FrameForm()
    """Image Blending home page."""
    return render_template('index.html', form=form)


@mb.route('/taser/update_image_path', methods=['POST'])
def set_image_path():

    image_path = os.path.normpath(request.values.get('input', ''))

    nx, ny = cam.set_image_path(image_path)

    return str([nx,ny])


@mb.route('/taser/update_decimation', methods=['POST'])
def set_decimation():

    dec_val = int(request.values.get('input', ''))

    cam.set_decimation(dec_val)

    return ''


@mb.route('/taser/crop_image', methods=['POST'])
def crop_image():

    minx = int(request.values.get('minx', ''))
    maxx = int(request.values.get('maxx', ''))
    miny = int(request.values.get('miny', ''))
    maxy = int(request.values.get('maxy', ''))

    cam.crop_image(minx, miny, maxx, maxy)

    return json.dumps({'hello world': 'i am a computer'})


@mb.route('/taser/ortho_image', methods=['POST'])
def ortho_image():

    output_image_path = os.path.normpath(request.values.get('input', ''))
    cam.ortho_image(output_image_path)

    return ''


def gen(camera):
    while True:
        time.sleep(.1)
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


def image_feed():
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
