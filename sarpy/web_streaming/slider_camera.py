class SliderCamera(object):

    def __init__(self):
        self.frame_num = 0
        self.imgs = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]

    def get_frame(self):
        """Return the current camera frame."""
        # wait for a signal from the camera thread
        return self.imgs[self.frame_num]

        return SliderCamera.frame

    def set_frame_num(self, frame_num):
        self.frame_num = frame_num
