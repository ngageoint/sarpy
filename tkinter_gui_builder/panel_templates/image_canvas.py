import PIL.Image
from PIL import ImageTk
import tkinter as tk
from tkinter_gui_builder.widgets import basic_widgets
import numpy as np


class AppVariables:
    def __init__(self):
        self.rect_border_width = 2
        self.line_width = 2
        self.point_size = 3

        self.foreground_color = "red"

        self.current_object_id = None
        self.object_ids = []            # type: [int]


class ImageCanvas(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.variables = AppVariables()

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = basic_widgets.Canvas(self, width=self.canvas_width, height=self.canvas_height)
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
        self.point_ids = []         # type: [int]

        self.nx_pix = None      # type: int
        self.ny_pix = None      # type: int

        self.tk_im = None           # type: ImageTk.PhotoImage
        self.image_data = None      # type: np.ndarray

        self.set_image_from_numpy_array(np.random.random((self.canvas_width, self.canvas_height)), scale_dynamic_range=True)
        self.set_image_from_numpy_array(np.random.random((self.canvas_width, self.canvas_height)) * 0 + 255, scale_dynamic_range=False)

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
        start_x = self.canvas.canvasx(event.x)
        start_y = self.canvas.canvasy(event.y)

        print(self.variables.current_object_id)
        # create rectangle if not yet exist
        if self.variables.current_object_id in self.variables.object_ids:
            self.canvas.coords(self.variables.current_object_id, start_x, start_y, start_x+1, start_y+1)
        else:
            rect_id = self.canvas.create_rectangle(start_x, start_y, start_x+1, start_y+1,
                                                   outline=self.variables.foreground_color,
                                                   width=self.variables.rect_border_width)
            self.variables.object_ids.append(rect_id)
            self.variables.current_object_id = rect_id

    def event_drag_rect(self, event):
        event_x_pos = self.canvas.canvasx(event.x)
        event_y_pos = self.canvas.canvasy(event.y)

        coords = self.canvas.coords(self.variables.current_object_id)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.variables.current_object_id, coords[0], coords[1], event_x_pos, event_y_pos)

    def event_initiate_line(self, event):
        # save mouse drag start position
        start_x = self.canvas.canvasx(event.x)
        start_y = self.canvas.canvasy(event.y)

        # create line if not yet exist
        if self.variables.current_object_id in self.variables.object_ids:
            self.canvas.coords(self.variables.current_object_id, start_x, start_y, start_x + 1, start_y + 1)
        else:
            line_id = self.canvas.create_line(start_x, start_y, start_x + 1, start_y + 1,
                                                   fill=self.variables.foreground_color,
                                                   width=self.variables.line_width)
            self.variables.object_ids.append(line_id)
            self.variables.current_object_id = line_id

    def event_drag_line(self, event):
        event_x_pos = self.canvas.canvasx(event.x)
        event_y_pos = self.canvas.canvasy(event.y)

        coords = self.canvas.coords(self.variables.current_object_id)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.variables.current_object_id, coords[0], coords[1], event_x_pos, event_y_pos)

    def event_draw_point(self, event):
        x1, y1 = (self.canvas.canvasx(event.x) - self.variables.point_size), (self.canvas.canvasy(event.y) - self.variables.point_size)
        x2, y2 = (self.canvas.canvasx(event.x) + self.variables.point_size), (self.canvas.canvasy(event.y) + self.variables.point_size)
        print(self.variables.current_object_id)
        if self.variables.current_object_id in self.variables.object_ids:
            self.canvas.coords(self.variables.current_object_id, x1, y1, x2, y2)
        else:
            point_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.variables.foreground_color)
            self.variables.object_ids.append(point_id)
            self.variables.current_object_id = point_id

    # TODO needs testing
    def get_data_in_rect(self, rect_id):
        coords = self.canvas.coords(rect_id)
        y_ul = int(coords[0])
        x_ul = int(coords[1])
        y_br = int(coords[2])
        x_br = int(coords[3])
        selected_image_data = self.image_data[y_ul: y_br, x_ul:x_br]
        return selected_image_data