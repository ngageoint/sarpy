#!/usr/bin/env python

from . import mb
from sarpy.atk.atk_interactive.forms import FrameForm

from flask import render_template, Response
from flask import request
import numpy as np

from algorithm_toolkit import app

from sarpy.web_streaming.frame_generator import FrameGenerator

cam = FrameGenerator()


@app.route('/taser/')
def index():
    form = FrameForm()
    """Image Blending home page."""
    return render_template('index.html', form=form)


@mb.route('/taser/frame_val', methods=['POST'])
def get_frame_value():
    slider_val = request.values.get('input', '')
    frame_val = int(np.round((int(slider_val))/100))
    print(frame_val)
    cam.set_frame_num(frame_val)
    return slider_val


@mb.route('/taser/slider_val', methods=['POST'])
def get_slider_value():
    slider_val = request.values.get('input', '')
    print(slider_val)
    float_slide_val = float(slider_val)/100.0
    cam.set_slider_val(float_slide_val)
    return slider_val


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_mem_jpg(camera):
    while True:
        frame = camera.get_mem_jpg()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def blend_mem_png(camera):
    while True:
        frame = camera.blend_mem_png()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


@mb.route('/taser/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(blend_mem_png(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
