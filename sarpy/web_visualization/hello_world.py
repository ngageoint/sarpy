from flask import Flask, send_file

import numpy as np
from PIL import Image
import io


app = Flask(__name__)


@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<body>
<canvas id="canvas" width="300" height="300" style="border:1px solid #000000;">
</canvas>
<script>
var sun = new Image();
var moon = new Image();
var earth = new Image();
function init() {
  sun.src = 'https://mdn.mozillademos.org/files/1456/Canvas_sun.png';
  moon.src = 'https://mdn.mozillademos.org/files/1443/Canvas_moon.png';
  earth.src = 'https://mdn.mozillademos.org/files/1429/Canvas_earth.png';
  window.requestAnimationFrame(draw);
}

function draw() {
  var ctx = document.getElementById('canvas').getContext('2d');

  ctx.globalCompositeOperation = 'destination-over';
  ctx.clearRect(0, 0, 300, 300); // clear canvas

  ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
  ctx.strokeStyle = 'rgba(0, 153, 255, 0.4)';
  ctx.save();
  ctx.translate(150, 150);

  // Earth
  var time = new Date();
  ctx.rotate(((2 * Math.PI) / 60) * time.getSeconds() + ((2 * Math.PI) / 60000) * time.getMilliseconds());
  ctx.translate(105, 0);
  ctx.fillRect(0, -12, 40, 24); // Shadow
  ctx.drawImage(earth, -12, -12);

  // Moon
  ctx.save();
  ctx.rotate(((2 * Math.PI) / 6) * time.getSeconds() + ((2 * Math.PI) / 6000) * time.getMilliseconds());
  ctx.translate(0, 28.5);
  ctx.drawImage(moon, -3.5, -3.5);
  ctx.restore();

  ctx.restore();
  
  ctx.beginPath();
  ctx.arc(150, 150, 105, 0, Math.PI * 2, false); // Earth orbit
  ctx.stroke();
 
  ctx.drawImage(sun, 0, 0, 300, 300);

  window.requestAnimationFrame(draw);
}

init();
</script>
</body>
</html>
'''


@app.route('/api/b')
def array():
    '''
    generate image from numpy.array using PIL.Image
    and send without saving on disk using io.BytesIO'''

    # arr = np.array([
    #     [255, 255,   0,   0,   0,   0,   0,   0, 255, 255],
    #     [255,   0, 255, 255, 255, 255, 255, 255,   0, 255],
    #     [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
    #     [  0, 255, 255,   0, 255, 255,   0, 255, 255,   0],
    #     [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
    #     [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
    #     [  0, 255,   0, 255, 255, 255, 255,   0, 255,   0],
    #     [  0, 255, 255,   0,   0,   0,   0, 255, 255,   0],
    #     [255,   0, 255, 255, 255, 255, 255, 255,   0, 255],
    #     [255, 255,   0,   0,   0,   0,   0,   0, 255, 255],
    # ])


    arr = np.array([
        [255, 255,   0,   0,   0,   0,   0,   0, 255, 255],
        [255,   0, 255, 255, 255, 255, 255, 255,   0, 255],
        [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
        [  0, 255, 255,   0, 255, 255,   0, 255, 255,   0],
        [  0, 0, 0, 0, 0, 0, 0, 0, 0,   0],
        [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
        [  0, 255,   0, 255, 255, 255, 255,   0, 255,   0],
        [  0, 255, 255,   0,   0,   0,   0, 255, 255,   0],
        [255,   0, 255, 255, 255, 255, 255, 255,   0, 255],
        [255, 255,   0,   0,   0,   0,   0,   0, 255, 255],
    ])


    img = Image.fromarray(arr.astype('uint8')) # convert arr to image

    file_object = io.BytesIO()   # create file in memory 
    img.save(file_object, 'PNG') # save as PNG in file in memory
    file_object.seek(0)          # move to beginning of file
                                 # so send_file() will read data from beginning of file

    return send_file(file_object,  mimetype='image/png')


if __name__ == "__main__":
    app.run()