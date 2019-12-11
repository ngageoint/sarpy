import PIL.Image
from PIL import ImageTk
import tkinter as tk
from sarpy.tkinter_gui_builder.widget_utils.basic_widgets import Canvas
import numpy as np


class ImageCanvas(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.x = 0
        self.y = 0

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = Canvas(self, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.pack()

        self.sbarv=tk.Scrollbar(self, orient=tk.VERTICAL)
        self.sbarh=tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.sbarv.grid(row=0, column=1, stick=tk.N+tk.S)
        self.sbarh.grid(row=1, column=0, sticky=tk.E+tk.W)

        self.image_id = None        # type: int
        self.rect_id = None         # type: int
        self.line_id = None         # type: int

        self.rect_coords = None

        self.start_x = None
        self.start_y = None

        self.nx_pix = None      # type: int
        self.ny_pix = None      # type: int

        self.tk_im = None           # type: ImageTk.PhotoImage
        self.image_data = None      # type: np.ndarray

        self.set_image_from_numpy_array(np.random.random((self.canvas_width, self.canvas_height)), scale_dynamic_range=True)

    def set_image_from_pil_image_object(self, pil_image):
        numpy_data = np.array(pil_image)
        self.set_image_from_numpy_array(numpy_data)

    def set_image_from_fname(self,
                             fname,         # type: str
                             ):
        im = PIL.Image.open(fname)
        self.set_image_from_pil_image_object(im)

    def set_image_from_numpy_array(self,
                                   numpy_data,                      # type: np.ndarray
                                   scale_dynamic_range=False,       # type: bool
                                   ):
        """
        This is the default way to set and display image data.  All other methods to update images should
        ultimately call this.

        :param numpy_data:
        :param scale_dynamic_range:
        :return:
        """
        if scale_dynamic_range:
            dynamic_range = numpy_data.max() - numpy_data.min()
            numpy_data = numpy_data - numpy_data.min()
            numpy_data = numpy_data / dynamic_range
            numpy_data = numpy_data * 255
            numpy_data = np.asanyarray(numpy_data, dtype=np.int8)
        pil_image = PIL.Image.fromarray(numpy_data)

        self.nx_pix, self.ny_pix = pil_image.size
        self.canvas.config(scrollregion=(0, 0, self.nx_pix, self.ny_pix))
        self.tk_im = ImageTk.PhotoImage(pil_image)
        self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.canvas.tag_lower(self.image_id)

        self.image_data = numpy_data

    def set_canvas_size(self,
                        width_npix,          # type: int
                        height_npix,         # type: int
                        ):
        self.canvas_width = width_npix
        self.canvas_height = height_npix
        self.canvas.config(width=width_npix, height=height_npix)

    def event_initiate_rect(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if not self.rect_id:
            self.rect_id = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def event_drag_rect(self, event):
        event_x_pos = self.canvas.canvasx(event.x)
        event_y_pos = self.canvas.canvasy(event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event_x_pos, event_y_pos)
        self.rect_coords = [self.start_y, self.start_x, event_y_pos, event_x_pos]

    def event_initiate_line(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if not self.line_id:
            self.line_id = self.canvas.create_line(self.x, self.y, 1, 1, fill='blue')

    def event_drag_line(self, event):
        event_x_pos = self.canvas.canvasx(event.x)
        event_y_pos = self.canvas.canvasy(event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.line_id, self.start_x, self.start_y, event_x_pos, event_y_pos)

    def get_data_in_rect(self):
        print(self.rect_coords)
        y_ul = int(self.rect_coords[0])
        x_ul = int(self.rect_coords[1])
        y_br = int(self.rect_coords[2])
        x_br = int(self.rect_coords[3])
        selected_image_data = self.image_data[y_ul: y_br, x_ul:x_br]
        return selected_image_data