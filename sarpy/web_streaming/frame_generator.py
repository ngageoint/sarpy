import imageio
from PIL import Image
import io
import numpy as np


class FrameGenerator(object):
    def __init__(self):
        self.frame_num = 0
        self.slider_val = 0.0
        self.imgs = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]
        im1_data = imageio.imread("1.jpg")
        im2_data = imageio.imread("2.jpg")
        im3_data = imageio.imread("3.jpg")
        self.numpy_data = [im1_data, im2_data, im3_data]

    def get_frame(self):
        """Return the current camera frame."""
        # wait for a signal from the camera thread
        return self.imgs[self.frame_num]

    def get_mem_jpg(self):
        im_data = self.numpy_data[self.frame_num]
        img = Image.fromarray(im_data.astype('uint8'))  # convert arr to image

        file_object = io.BytesIO()  # create file in memory
        img.save(file_object, format='jpeg')  # save as jpg in file in memory
        file_object.seek(0)  # move to beginning of file

        imdata2 = file_object.read()
        return imdata2

    def blend_mem_png(self):
        floating_slider_val = self.slider_val
        upper_image_index = int(np.floor(self.slider_val))
        lower_image_index = int(np.ceil(self.slider_val))
        im_data1 = self.numpy_data[upper_image_index]
        im_data2 = self.numpy_data[lower_image_index]

        im_data2_percent = floating_slider_val - upper_image_index
        im_data1_percent = 1 - im_data2_percent

        blended_image = im_data2 * im_data2_percent + im_data1 * im_data1_percent

        img = Image.fromarray(blended_image.astype('uint8'))  # convert arr to image

        file_object = io.BytesIO()  # create file in memory
        img.save(file_object, format='png')  # save as jpg in file in memory
        file_object.seek(0)  # move to beginning of file

        png_data = file_object.read()

        return png_data

    def set_frame_num(self, frame_num):
        self.frame_num = frame_num

    def set_slider_val(self, slider_val):
        self.slider_val = slider_val
