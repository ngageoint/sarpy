#!/usr/bin/env python
import os
from flask import Flask, render_template, Response, send_file
from flask import request
import numpy as np

from camera_slider import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)

SLIDER_VAL = 0
imgs = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]


cam = Camera()


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/slider_val', methods=['POST'])
def get_slider_value():
    slider_val = request.values.get('input', '')
    frame_val = int(np.floor((int(slider_val))/100))
    print(frame_val)
    cam.set_frame_num(frame_val)
    return slider_val


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_mem_jpg(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_mem_jpg()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_mem_jpg(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
