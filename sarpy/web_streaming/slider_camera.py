import imageio
import tempfile
from PIL import Image
import io

class SliderCamera(object):

    def __init__(self):
        self.frame_num = 0
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


    def set_frame_num(self, frame_num):
        self.frame_num = frame_num
