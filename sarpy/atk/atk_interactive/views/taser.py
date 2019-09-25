#!/usr/bin/env python

import time
from . import mb
from sarpy.atk.atk_interactive.forms import FrameForm

from flask import render_template, Response
from flask import request

from algorithm_toolkit import app

from sarpy.atk.atk_interactive.frame_generator import FrameGenerator

cam = FrameGenerator()


@app.route('/taser/')
def index():
    form = FrameForm()
    """Image Blending home page."""
    return render_template('index.html', form=form)


@mb.route('/taser/update_decimation', methods=['POST'])
def set_decimation():
    dec_val = int(request.values.get('input', ''))

    cam.set_decimation(dec_val)

    return str(dec_val)


def gen(camera):
    while True:
        time.sleep(.1)
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


@mb.route('/taser/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
