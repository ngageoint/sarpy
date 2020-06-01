import tkinter
from tkinter_gui_builder.widgets.image_canvas import ImageCanvas


class ImageCanvasPanel(tkinter.LabelFrame):
    def __init__(self,
                 master,
                 ):
        tkinter.LabelFrame.__init__(self, master)

        self.canvas = ImageCanvas(self)
        self.canvas.pack()
