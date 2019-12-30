import PIL.Image
from PIL import ImageTk
import tkinter as tk
from tkinter_gui_builder.widgets import basic_widgets
from tkinter_gui_builder.canvas_image_objects.canvas_image import CanvasDisplayImage
import numpy as np


class ToolConstants:
    ZOOM_IN_TOOL = "zoom in"
    ZOOM_OUT_TOOL = "zoom out"
    DRAW_RECT_TOOL = "draw rect"
    DRAW_LINE_TOOL = "draw line"
    DRAW_POINT_TOOL = "draw point"
    DRAW_ARROW_TOOL = "draw arrow"


class AppVariables:
    def __init__(self):
        self.rect_border_width = 2
        self.line_width = 2
        self.point_size = 3

        self.foreground_color = "red"

        self.rect_border_width = 2

        self.image_id = None                # type: int

        self.current_object_id = None
        self.object_ids = []            # type: [int]
        self.object_types_dict = {}
        self.canvas_image_object = None         # type: CanvasDisplayImage
        self.zoom_rect_id = None                # type: int
        self.zoom_rect_color = "blue"
        self.zoom_rect_border_width = 2

        self.animate_zoom = True
        self.n_zoom_animations = 15

        self.select_rect_id = None
        self.select_rect_color = "red"
        self.select_rect_border_width = 2
        self.current_tool = None


class ImageCanvas(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.variables = AppVariables()
        self.constants = ToolConstants()

        self.scale_dynamic_range = False
        self.canvas_height = None
        self.canvas_width = None
        self.canvas = basic_widgets.Canvas(self)
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

        self.tk_im = None               # type: ImageTk.PhotoImage

        self.variables.zoom_rect_id = self.create_new_rect((0, 0, 1, 1), outline=self.variables.zoom_rect_color, width=self.variables.zoom_rect_border_width)
        self.variables.select_rect_id = self.create_new_rect((0, 0, 1, 1), outline=self.variables.select_rect_color, width=self.variables.select_rect_border_width)

        self.canvas.on_left_mouse_click(self.callback_handle_left_mouse_click)
        self.canvas.on_left_mouse_motion(self.callback_handle_left_mouse_motion)
        self.canvas.on_left_mouse_release(self.callback_handle_left_mouse_release)

        self.variables.current_tool = None
        self.variables.current_object_id = None

    def get_object_type(self,
                        object_id,          # type: int
                        ):
        return self.variables.object_types_dict[str(object_id)]

    def hide_shape(self, shape_id):
        if shape_id:
            self.canvas.itemconfigure(shape_id, state="hidden")

    def show_shape(self, shape_id):
        if shape_id:
            self.canvas.itemconfigure(shape_id, state="normal")

    def callback_handle_left_mouse_release(self, event):
        if self.variables.current_tool == self.constants.ZOOM_IN_TOOL:
            rect_coords = self.canvas.coords(self.variables.zoom_rect_id)
            self.zoom_to_selection(rect_coords)
            self.hide_shape(self.variables.zoom_rect_id)
        if self.variables.current_tool == self.constants.ZOOM_OUT_TOOL:
            rect_coords = self.canvas.coords(self.variables.zoom_rect_id)
            x1 = -rect_coords[0]
            x2 = self.canvas_width + rect_coords[2]
            y1 = -rect_coords[1]
            y2 = self.canvas_height + rect_coords[3]
            zoom_rect = (x1, y1, x2, y2)
            self.zoom_to_selection(zoom_rect)
            self.hide_shape(self.variables.zoom_rect_id)

    def callback_handle_left_mouse_click(self, event):
        self.event_initiate_shape(event)

    def callback_handle_left_mouse_motion(self, event):
        self.event_drag_shape(event)

    def set_canvas_image(self,
                         fname,            # type: str
                         ):
        self.variables.canvas_image_object.init_from_fname_and_canvas_size(fname, self.canvas_height, self.canvas_width)
        self.set_image_from_numpy_array(self.variables.canvas_image_object.canvas_display_image)

    def set_image_from_numpy_array(self,
                                   numpy_data,                      # type: np.ndarray
                                   ):
        """
        This is the default way to set and display image data.  All other methods to update images should
        ultimately call this.
        """
        if self.scale_dynamic_range:
            dynamic_range = numpy_data.max() - numpy_data.min()
            numpy_data = numpy_data - numpy_data.min()
            numpy_data = numpy_data / dynamic_range
            numpy_data = numpy_data * 255
            numpy_data = np.asanyarray(numpy_data, dtype=np.int8)
        pil_image = PIL.Image.fromarray(numpy_data)
        self.set_image_from_pil_image(pil_image)

    def set_image_from_pil_image(self, pil_image):
        nx_pix, ny_pix = pil_image.size
        self.canvas.config(scrollregion=(0, 0, nx_pix, ny_pix))
        self.tk_im = ImageTk.PhotoImage(pil_image)
        self.variables.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.canvas.tag_lower(self.variables.image_id)

    def set_canvas_size(self,
                        width_npix,          # type: int
                        height_npix,         # type: int
                        ):
        self.canvas_width = width_npix
        self.canvas_height = height_npix
        self.canvas.config(width=width_npix, height=height_npix)

    def modify_existing_shape(self,
                              shape_id,                   # type: int
                              new_coords,                # type: tuple
                              ):
        self.show_shape(shape_id)
        if self.get_object_type(shape_id) == 'point':
            x1, y1 = (new_coords[2] - self.variables.point_size), (new_coords[3] - self.variables.point_size)
            x2, y2 = (new_coords[2] + self.variables.point_size), (new_coords[3] + self.variables.point_size)
            self.canvas.coords(shape_id, x1, y1, x2, y2)
        else:
            self.canvas.coords(shape_id, new_coords)

    def event_initiate_shape(self, event):
        # save mouse drag start position
        start_x = self.canvas.canvasx(event.x)
        start_y = self.canvas.canvasy(event.y)

        coords = (start_x, start_y, start_x + 1, start_y + 1)

        if self.variables.current_object_id not in self.variables.object_ids:
            if self.variables.current_tool == self.constants.DRAW_LINE_TOOL:
                self.create_new_line(coords)
            elif self.variables.current_tool == self.constants.DRAW_RECT_TOOL:
                self.create_new_rect(coords)
            elif self.variables.current_tool == self.constants.DRAW_POINT_TOOL:
                self.create_new_point(coords)
            elif self.variables.current_tool == self.constants.DRAW_ARROW_TOOL:
                self.create_new_arrow(coords)
            else:
                print("no shape tool selected")
        else:
            if self.variables.current_object_id in self.variables.object_ids:
                self.modify_existing_shape(self.variables.current_object_id, coords)

    def event_initiate_rect(self, event):
        # save current tool and set it back at the end.  This is to handle special cases, such as zoom tools
        tmp_tool = self.variables.current_tool
        self.variables.current_tool = self.constants.DRAW_RECT_TOOL
        self.event_initiate_shape(event)
        self.variables.current_tool = tmp_tool

    def event_initiate_line(self, event):
        self.variables.current_tool = self.constants.DRAW_LINE_TOOL
        self.event_initiate_shape(event)

    def event_initiate_arrow(self, event):
        self.variables.current_tool = self.constants.DRAW_ARROW_TOOL
        self.event_initiate_shape(event)

    def event_draw_point(self, event):
        self.variables.current_tool = self.constants.DRAW_POINT_TOOL
        self.event_initiate_shape(event)

    def event_drag_shape(self, event):
        if self.variables.current_object_id:
            self.show_shape(self.variables.current_object_id)
            event_x_pos = self.canvas.canvasx(event.x)
            event_y_pos = self.canvas.canvasy(event.y)
            coords = self.canvas.coords(self.variables.current_object_id)
            self.modify_existing_shape(self.variables.current_object_id, (coords[0], coords[1], event_x_pos, event_y_pos))
        else:
            pass

    def create_new_rect(self,
                        coords,         # type: (int, int, int, int)
                        **options,
                        ):
        return self.create_new_shape('rect', coords, **options)

    def create_new_arrow(self,
                         coords,
                         **options,
                         ):
        return self.create_new_shape('arrow', coords, **options)

    def create_new_point(self,
                         coords,
                         **options):
        return self.create_new_shape('point', coords, **options)

    def create_new_line(self, coords, **options):
        return self.create_new_shape('line', coords, **options)

    def create_new_shape(self,
                         shape_type,        # type: str
                         coords,            # type: tuple
                         **options,
                         ):
        if shape_type.lower() == 'rect':
            if options == {}:
                shape_id = self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3],
                                                       outline=self.variables.foreground_color,
                                                       width=self.variables.rect_border_width)
            else:
                shape_id = self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3], options)
        if shape_type.lower() == 'line':
            if options == {}:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3],
                                                       fill=self.variables.foreground_color,
                                                       width=self.variables.line_width)
            else:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3], options)
        if shape_type.lower() == 'arrow':
            if options == {}:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3],
                                                       fill=self.variables.foreground_color,
                                                       width=self.variables.line_width,
                                                       arrow=tk.LAST)
            else:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3], options, arrow=tk.LAST)
        if shape_type.lower() == 'point':
            x1, y1 = (coords[0] - self.variables.point_size), (coords[1] - self.variables.point_size)
            x2, y2 = (coords[0] + self.variables.point_size), (coords[1] + self.variables.point_size)
            if options == {}:
                shape_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.variables.foreground_color)
            else:
                shape_id = self.canvas.create_oval(x1, y1, x2, y2, options)

        self.variables.object_ids.append(shape_id)
        self.variables.current_object_id = shape_id
        self.variables.object_types_dict[str(shape_id)] = shape_type.lower()
        return shape_id

    def get_point_xy_center(self,
                            point_id,               # type: int
                            ):
        point_coords = self.canvas.coords(point_id)
        point_x = (point_coords[0] + point_coords[2]) / 2.0
        point_y = (point_coords[1] + point_coords[3]) / 2.0
        return point_x, point_y

    def get_data_in_rect(self, rect_id):
        coords = self.canvas.coords(rect_id)
        return self.variables.canvas_image_object.get_image_data_in_canvas_rect(coords)

    def zoom_to_selection(self, canvas_rect):
        background_image = self.variables.canvas_image_object.canvas_display_image
        self.variables.canvas_image_object.update_canvas_display_image_from_canvas_rect(canvas_rect)
        new_image = PIL.Image.fromarray(self.variables.canvas_image_object.canvas_display_image)
        if self.variables.animate_zoom is True:
            n_animations = self.variables.n_zoom_animations
            background_image = background_image / 2
            canvas_x1, canvas_y1, canvas_x2, canvas_y2 = canvas_rect
            display_x_ul = min(canvas_x1, canvas_x2)
            display_x_br = max(canvas_x1, canvas_x2)
            display_y_ul = min(canvas_y1, canvas_y2)
            display_y_br = max(canvas_y1, canvas_y2)
            x_diff = new_image.width - (display_x_br - display_x_ul)
            y_diff = new_image.height - (display_y_br - display_y_ul)
            new_display_image = PIL.Image.fromarray(background_image)
            for i in range(n_animations):
                new_x_ul = int(display_x_ul * (1 - i/(n_animations-1)))
                new_y_ul = int(display_y_ul * (1 - i/(n_animations-1)))
                new_size_x = int((display_x_br - display_x_ul) + x_diff * (i/(n_animations-1)))
                new_size_y = int((display_y_br - display_y_ul) + y_diff * (i/(n_animations-1)))
                resized_zoom_image = new_image.resize((new_size_x, new_size_y))
                new_display_image.paste(resized_zoom_image, (new_x_ul, new_y_ul))
                self.set_image_from_pil_image(new_display_image)
                self.canvas.update()
        self.set_image_from_numpy_array(self.variables.canvas_image_object.canvas_display_image)
        self.canvas.update()

    def set_current_tool_to_zoom_out(self):
        self.variables.current_object_id = self.variables.zoom_rect_id
        self.variables.current_tool = self.constants.ZOOM_OUT_TOOL

    def set_current_tool_to_zoom_in(self):
        self.variables.current_object_id = self.variables.zoom_rect_id
        self.variables.current_tool = self.constants.ZOOM_IN_TOOL

    def set_current_tool_to_draw_rect(self, rect_id=None):
        self.variables.current_object_id = rect_id
        self.variables.current_tool = self.constants.DRAW_RECT_TOOL
        self.show_shape(rect_id)

    def set_current_tool_to_draw_line(self, line_id=None):
        self.variables.current_object_id = line_id
        self.variables.current_tool = self.constants.DRAW_LINE_TOOL
        self.show_shape(line_id)

    def set_current_tool_to_draw_arrow(self, arrow_id=None):
        self.variables.current_object_id = arrow_id
        self.variables.current_tool = self.constants.DRAW_ARROW_TOOL
        self.show_shape(arrow_id)

    def set_current_tool_to_draw_point(self, point_id=None):
        self.variables.current_object_id = point_id
        self.variables.current_tool = self.constants.DRAW_POINT_TOOL
        self.show_shape(point_id)
