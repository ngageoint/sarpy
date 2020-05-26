import PIL.Image
from PIL import ImageTk
from tkinter_gui_builder.widgets import basic_widgets
from tkinter_gui_builder.utils.color_utils.hex_color_palettes import SeabornHexPalettes
import tkinter_gui_builder.utils.color_utils.color_utils as color_utils
import platform
import numpy
import time
import tkinter
import tkinter.colorchooser as colorchooser

from numpy import ndarray
from typing import Union
import PIL.Image
import numpy as np
from tkinter_gui_builder.image_readers.image_reader import ImageReader

from .tool_constants import ShapePropertyConstants as SHAPE_PROPERTIES
from .tool_constants import ShapeTypeConstants as SHAPE_TYPES
from .tool_constants import ToolConstants as TOOLS

if platform.system() == "Linux":
    import pyscreenshot as ImageGrab
else:
    from PIL import ImageGrab


class AppVariables:
    def __init__(self):
        self.rect_border_width = 2
        self.line_width = 2
        self.point_size = 3
        self.poly_border_width = 2
        self.poly_fill = None

        self.foreground_color = "red"

        self.image_id = None                # type: int

        self.current_shape_id = None
        self.current_shape_canvas_anchor_point_xy = None
        self.pan_anchor_point_xy = None
        self.shape_ids = []            # type: [int]
        self.shape_properties = {}
        self.canvas_image_object = None         # type: CanvasImage
        self.zoom_rect_id = None                # type: int
        self.zoom_rect_color = "cyan"
        self.zoom_rect_border_width = 2

        self.animate_zoom = True
        self.animate_pan = False
        self.n_zoom_animations = 5
        self.animation_time_in_seconds = 0.3

        self.select_rect_id = None
        self.select_rect_color = "red"
        self.select_rect_border_width = 2
        self.active_tool = None
        self.current_tool = None

        self.pan_anchor_point_xy = (0, 0)

        self.vertex_selector_pixel_threshold = 10.0         # type: float

        self.the_canvas_is_currently_zooming = False        # type: bool
        self.mouse_wheel_zoom_percent_per_event = 1.5

        self.actively_drawing_shape = False

        self.shape_drag_xy_limits = {}          # type: dict

        self.highlight_color_palette = SeabornHexPalettes.blues
        self.highlight_n_colors_cycle = 10

        self.tmp_points = None              # type: [int]

        self.tmp_closest_coord_index = 0        # type: int

    # @property
    # def canvas_image_object(self):  # type: () -> AbstractCanvasImage
    #     return self._canvas_image_object
    #
    # @canvas_image_object.setter
    # def canvas_image_object(self, value):
    #     if value is None:
    #         self._canvas_image_object = None
    #         return
    #
    #     if not isinstance(value, AbstractCanvasImage):
    #         raise TypeError('Requires instance of AbstractCanvasImage, got {}'.format(type(value)))
    #     self._canvas_image_object = value


# class SpecificAppVariables(AppVariables):
#     @property
#     def canvas_image_object(self):  # type: () -> object
#         return self._canvas_image_object
#
#     @canvas_image_object.setter
#     def canvas_image_object(self, value):
#         if value is None:
#             self._canvas_image_object = None
#             return
#
#         if not isinstance(value, object):
#             raise TypeError('Requires instance of AbstractCanvasImage, got {}'.format(type(value)))
#         self._canvas_image_object = value


class ImageCanvas(tkinter.LabelFrame):
    def __init__(self,
                 master,
                 ):
        tkinter.LabelFrame.__init__(self, master)

        self.SHAPE_PROPERTIES = SHAPE_PROPERTIES
        self.SHAPE_TYPES = SHAPE_TYPES
        self.TOOLS = TOOLS

        self.variables = AppVariables()

        self.scale_dynamic_range = False
        self.canvas_height = 200            # default width
        self.canvas_width = 300             # default height
        self.canvas = basic_widgets.Canvas(self)
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        self.canvas.pack()

        self.canvas.grid(row=0, column=0, sticky=tkinter.N+tkinter.S+tkinter.E+tkinter.W)
        self.sbarv = None         # type: tkinter.Scrollbar
        self.sbarh = None         # type: tkinter.Scrollbar

        self.variables.zoom_rect_id = self.create_new_rect((0, 0, 1, 1), outline=self.variables.zoom_rect_color, width=self.variables.zoom_rect_border_width)
        self.variables.select_rect_id = self.create_new_rect((0, 0, 1, 1), outline=self.variables.select_rect_color, width=self.variables.select_rect_border_width)

        # hide the shapes we initialize
        self.hide_shape(self.variables.select_rect_id)
        self.hide_shape(self.variables.zoom_rect_id)

        self.canvas.on_left_mouse_click(self.callback_handle_left_mouse_click)
        self.canvas.on_left_mouse_motion(self.callback_handle_left_mouse_motion)
        self.canvas.on_left_mouse_release(self.callback_handle_left_mouse_release)
        self.canvas.on_right_mouse_click(self.callback_handle_right_mouse_click)
        self.canvas.on_mouse_motion(self.callback_handle_mouse_motion)

        self.canvas.on_mouse_wheel(self.callback_mouse_zoom)

        self.variables.active_tool = None
        self.variables.current_shape_id = None

        self.zoom_on_wheel = True

        self._tk_im = None               # type: ImageTk.PhotoImage

        self.rescale_image_to_fit_canvas = True

    def set_image_reader(self,
                         image_reader,  # type: ImageReader
                         ):
        self.variables.canvas_image_object = CanvasImage(image_reader, self.canvas_width, self.canvas_height)
        if self.rescale_image_to_fit_canvas:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.display_image)
        else:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.canvas_decimated_image)

    def set_labelframe_text(self, label):
        self.config(text=label)

    def get_canvas_line_length(self, line_id):
        line_coords = self.canvas.coords(line_id)
        x1 = line_coords[0]
        y1 = line_coords[1]
        x2 = line_coords[2]
        y2 = line_coords[3]
        length = numpy.sqrt(numpy.square(x2-x1) + numpy.square(y2-y1))
        return length

    def get_image_line_length(self, line_id):
        canvas_line_length = self.get_canvas_line_length(line_id)
        return canvas_line_length * self.variables.canvas_image_object.decimation_factor

    def get_shape_type(self,
                       shape_id,  # type: int
                       ):
        return self._get_shape_property(shape_id, SHAPE_PROPERTIES.SHAPE_TYPE)

    def hide_shape(self, shape_id):
        if shape_id:
            self.canvas.itemconfigure(shape_id, state="hidden")

    def show_shape(self, shape_id):
        if shape_id:
            self.canvas.itemconfigure(shape_id, state="normal")

    def callback_mouse_zoom(self, event):
        if self.zoom_on_wheel:
            delta = event.delta
            single_delta = 120

            # handle case where platform is linux:
            if platform.system() == "Linux":
                delta = single_delta
                if event.num == 5:
                    delta = delta*-1

            zoom_in_box_half_width = int(self.canvas_width / self.variables.mouse_wheel_zoom_percent_per_event / 2)
            zoom_out_box_half_width = int(self.canvas_width * self.variables.mouse_wheel_zoom_percent_per_event / 2)
            zoom_in_box_half_height = int(self.canvas_height / self.variables.mouse_wheel_zoom_percent_per_event / 2)
            zoom_out_box_half_height = int(self.canvas_height * self.variables.mouse_wheel_zoom_percent_per_event / 2)

            x = event.x
            y = event.y

            after_zoom_x_offset = (self.canvas_width/2 - x)/self.variables.mouse_wheel_zoom_percent_per_event
            after_zoom_y_offset = (self.canvas_height/2 - y)/self.variables.mouse_wheel_zoom_percent_per_event

            x_offset_point = x + after_zoom_x_offset
            y_offset_point = y + after_zoom_y_offset

            zoom_in_box = [x_offset_point - zoom_in_box_half_width,
                           y_offset_point - zoom_in_box_half_height,
                           x_offset_point + zoom_in_box_half_width,
                           y_offset_point + zoom_in_box_half_height]

            zoom_out_box = [x_offset_point - zoom_out_box_half_width,
                            y_offset_point - zoom_out_box_half_height,
                            x_offset_point + zoom_out_box_half_width,
                            y_offset_point + zoom_out_box_half_height]

            if self.variables.the_canvas_is_currently_zooming:
                pass
            else:
                if delta > 0:
                    self.zoom_to_selection(zoom_in_box, self.variables.animate_zoom)
                else:
                    self.zoom_to_selection(zoom_out_box, self.variables.animate_zoom)
        else:
            pass

    def animate_with_numpy_frame_sequence(self,
                                          numpy_frame_sequence,  # type: [numpy.ndarray]
                                          frames_per_second=15,  # type: float
                                          ):
        sleep_time = 1/frames_per_second
        for animation_frame in numpy_frame_sequence:
            tic = time.time()
            self.set_image_from_numpy_array(animation_frame)
            self.canvas.update()
            toc = time.time()
            frame_generation_time = toc-tic
            if frame_generation_time < sleep_time:
                new_sleep_time = sleep_time - frame_generation_time
                time.sleep(new_sleep_time)
            else:
                pass

    def animate_with_pil_frame_sequence(self,
                                        pil_frame_sequence,  # type: [PIL.Image]
                                        frames_per_second=15,  # type: float
                                        ):
        sleep_time = 1/frames_per_second
        for animation_frame in pil_frame_sequence:
            tic = time.time()
            self._set_image_from_pil_image(animation_frame)
            self.canvas.update()
            toc = time.time()
            frame_generation_time = toc-tic
            if frame_generation_time < sleep_time:
                new_sleep_time = sleep_time - frame_generation_time
                time.sleep(new_sleep_time)
            else:
                pass

    def callback_handle_left_mouse_click(self, event):
        if self.variables.active_tool == TOOLS.PAN_TOOL:
            self.variables.pan_anchor_point_xy = event.x, event.y
            self.variables.tmp_anchor_point = event.x, event.y
        elif self.variables.active_tool == TOOLS.TRANSLATE_SHAPE_TOOL:
            self.variables.translate_anchor_point_xy = event.x, event.y
            self.variables.tmp_anchor_point = event.x, event.y
        elif self.variables.active_tool == TOOLS.EDIT_SHAPE_COORDS_TOOL:
            closest_coord_index = self.find_closest_shape_coord(self.variables.current_shape_id, event.x, event.y)
            self.variables.tmp_closest_coord_index = closest_coord_index
        elif self.variables.active_tool == TOOLS.SELECT_CLOSEST_SHAPE_TOOL:
            closest_shape_id = self.find_closest_shape(event.x, event.y)
            self.variables.current_shape_id = closest_shape_id
            self.highlight_existing_shape(self.variables.current_shape_id)
        else:
            start_x = self.canvas.canvasx(event.x)
            start_y = self.canvas.canvasy(event.y)

            self.variables.current_shape_canvas_anchor_point_xy = (start_x, start_y)

            if self.variables.current_shape_id not in self.variables.shape_ids:
                coords = (start_x, start_y, start_x + 1, start_y + 1)
                if self.variables.active_tool == TOOLS.DRAW_LINE_BY_DRAGGING:
                    self.create_new_line(coords)
                elif self.variables.active_tool == TOOLS.DRAW_LINE_BY_CLICKING:
                    self.create_new_line(coords)
                    self.variables.actively_drawing_shape = True
                elif self.variables.active_tool == TOOLS.DRAW_ARROW_BY_DRAGGING:
                    self.create_new_arrow(coords)
                elif self.variables.active_tool == TOOLS.DRAW_ARROW_BY_CLICKING:
                    self.create_new_arrow(coords)
                    self.variables.actively_drawing_shape = True
                elif self.variables.active_tool == TOOLS.DRAW_RECT_BY_DRAGGING:
                    self.create_new_rect(coords)
                elif self.variables.active_tool == TOOLS.DRAW_RECT_BY_CLICKING:
                    self.create_new_rect(coords)
                    self.variables.actively_drawing_shape = True
                elif self.variables.active_tool == TOOLS.DRAW_POINT_BY_CLICKING:
                    self.create_new_point((start_x, start_y))
                elif self.variables.active_tool == TOOLS.DRAW_POLYGON_BY_CLICKING:
                    self.create_new_polygon(coords)
                    self.variables.actively_drawing_shape = True
                else:
                    print("no tool selected")
            else:
                if self.variables.current_shape_id in self.variables.shape_ids:
                    if self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.POINT:
                        self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id,
                                                                       (start_x, start_y))
                    elif self.variables.active_tool == TOOLS.DRAW_LINE_BY_CLICKING:
                        self.event_click_line(event)
                    elif self.variables.active_tool == TOOLS.DRAW_ARROW_BY_CLICKING:
                        self.event_click_line(event)
                    elif self.variables.active_tool == TOOLS.DRAW_POLYGON_BY_CLICKING:
                        self.event_click_polygon(event)
                    elif self.variables.active_tool == TOOLS.DRAW_RECT_BY_CLICKING:
                        if self.variables.actively_drawing_shape:
                            self.variables.actively_drawing_shape = False
                        else:
                            self.variables.actively_drawing_shape = True

    def callback_handle_left_mouse_release(self, event):
        if self.variables.active_tool == TOOLS.PAN_TOOL:
            self._pan(event)
        if self.variables.active_tool == TOOLS.ZOOM_IN_TOOL:
            rect_coords = self.canvas.coords(self.variables.zoom_rect_id)
            self.zoom_to_selection(rect_coords, self.variables.animate_zoom)
            self.hide_shape(self.variables.zoom_rect_id)
        if self.variables.active_tool == TOOLS.ZOOM_OUT_TOOL:
            rect_coords = self.canvas.coords(self.variables.zoom_rect_id)
            x1 = -rect_coords[0]
            x2 = self.canvas_width + rect_coords[2]
            y1 = -rect_coords[1]
            y2 = self.canvas_height + rect_coords[3]
            zoom_rect = (x1, y1, x2, y2)
            self.zoom_to_selection(zoom_rect, self.variables.animate_zoom)
            self.hide_shape(self.variables.zoom_rect_id)

    def callback_handle_mouse_motion(self, event):
        if self.variables.actively_drawing_shape:
            if self.variables.active_tool == TOOLS.DRAW_LINE_BY_CLICKING:
                self.event_drag_multipoint_line(event)
            elif self.variables.active_tool == TOOLS.DRAW_ARROW_BY_CLICKING:
                self.event_drag_multipoint_line(event)
            elif self.variables.active_tool == TOOLS.DRAW_POLYGON_BY_CLICKING:
                self.event_drag_multipoint_polygon(event)
            elif self.variables.active_tool == TOOLS.DRAW_RECT_BY_CLICKING:
                self.event_drag_line(event)
        elif self.variables.current_tool == TOOLS.EDIT_SHAPE_TOOL:
            if self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.RECT:
                select_x1, select_y1, select_x2, select_y2 = self.get_shape_canvas_coords(
                    self.variables.current_shape_id)
                select_xul = min(select_x1, select_x2)
                select_xlr = max(select_x1, select_x2)
                select_yul = min(select_y1, select_y2)
                select_ylr = max(select_y1, select_y2)

                distance_to_ul = numpy.sqrt(numpy.square(event.x - select_xul) + numpy.square(event.y - select_yul))
                distance_to_ur = numpy.sqrt(numpy.square(event.x - select_xlr) + numpy.square(event.y - select_yul))
                distance_to_lr = numpy.sqrt(numpy.square(event.x - select_xlr) + numpy.square(event.y - select_ylr))
                distance_to_ll = numpy.sqrt(numpy.square(event.x - select_xul) + numpy.square(event.y - select_ylr))

                if distance_to_ul < self.variables.vertex_selector_pixel_threshold:
                    self.canvas.config(cursor="top_left_corner")
                    self.variables.active_tool = TOOLS.EDIT_SHAPE_COORDS_TOOL
                elif distance_to_ur < self.variables.vertex_selector_pixel_threshold:
                    self.canvas.config(cursor="top_right_corner")
                    self.variables.active_tool = TOOLS.EDIT_SHAPE_COORDS_TOOL
                elif distance_to_lr < self.variables.vertex_selector_pixel_threshold:
                    self.canvas.config(cursor="bottom_right_corner")
                    self.variables.active_tool = TOOLS.EDIT_SHAPE_COORDS_TOOL
                elif distance_to_ll < self.variables.vertex_selector_pixel_threshold:
                    self.canvas.config(cursor="bottom_left_corner")
                    self.variables.active_tool = TOOLS.EDIT_SHAPE_COORDS_TOOL
                elif select_xul < event.x < select_xlr and select_yul < event.y < select_ylr:
                    self.canvas.config(cursor="fleur")
                    self.variables.active_tool = TOOLS.TRANSLATE_SHAPE_TOOL
                else:
                    self.canvas.config(cursor="arrow")
                    self.variables.active_tool = None

    def callback_handle_left_mouse_motion(self, event):
        if self.variables.active_tool == TOOLS.PAN_TOOL:
            x_dist = event.x - self.variables.tmp_anchor_point[0]
            y_dist = event.y - self.variables.tmp_anchor_point[1]
            self.canvas.move(self.variables.image_id, x_dist, y_dist)
            self.variables.tmp_anchor_point = event.x, event.y
        elif self.variables.active_tool == TOOLS.TRANSLATE_SHAPE_TOOL:
            x_dist = event.x - self.variables.tmp_anchor_point[0]
            y_dist = event.y - self.variables.tmp_anchor_point[1]
            new_x1 = min(self.get_shape_canvas_coords(self.variables.current_shape_id)[0] + x_dist,
                         self.get_shape_canvas_coords(self.variables.current_shape_id)[2] + x_dist)
            new_x2 = max(self.get_shape_canvas_coords(self.variables.current_shape_id)[0] + x_dist,
                         self.get_shape_canvas_coords(self.variables.current_shape_id)[2] + x_dist)
            new_y1 = min(self.get_shape_canvas_coords(self.variables.current_shape_id)[1] + y_dist,
                         self.get_shape_canvas_coords(self.variables.current_shape_id)[3] + y_dist)
            new_y2 = max(self.get_shape_canvas_coords(self.variables.current_shape_id)[1] + y_dist,
                         self.get_shape_canvas_coords(self.variables.current_shape_id)[3] + y_dist)
            width = new_x2 - new_x1
            height = new_y2 - new_y1

            if str(self.variables.current_shape_id) in self.variables.shape_drag_xy_limits.keys():
                drag_x_lim_1, drag_y_lim_1, drag_x_lim_2, drag_y_lim_2 = self.variables.shape_drag_xy_limits[str(self.variables.current_shape_id)]
                if self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.RECT:
                    if new_x1 < drag_x_lim_1:
                        new_x1 = drag_x_lim_1
                        new_x2 = new_x1 + width
                    if new_x2 > drag_x_lim_2:
                        new_x2 = drag_x_lim_2
                        new_x1 = new_x2 - width
                    if new_y1 < drag_y_lim_1:
                        new_y1 = drag_y_lim_1
                        new_y2 = new_y1 + height
                    if new_y2 > drag_y_lim_2:
                        new_y2 = drag_y_lim_2
                        new_y1 = new_y2 - height
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (new_x1, new_y1, new_x2, new_y2), update_pixel_coords=True)
            self.variables.tmp_anchor_point = event.x, event.y
        elif self.variables.active_tool == TOOLS.EDIT_SHAPE_COORDS_TOOL:
            previous_coords = self.get_shape_canvas_coords(self.variables.current_shape_id)
            coord_x_index = self.variables.tmp_closest_coord_index*2
            coord_y_index = coord_x_index + 1
            new_coords = list(previous_coords)
            new_coords[coord_x_index] = event.x
            new_coords[coord_y_index] = event.y
            if str(self.variables.current_shape_id) in self.variables.shape_drag_xy_limits.keys():
                drag_x_lim_1, drag_y_lim_1, drag_x_lim_2, drag_y_lim_2 = self.variables.shape_drag_xy_limits[str(self.variables.current_shape_id)]
                if new_coords[coord_x_index] < drag_x_lim_1:
                    new_coords[coord_x_index] = drag_x_lim_1
                if new_coords[coord_x_index] > drag_x_lim_2:
                    new_coords[coord_x_index] = drag_x_lim_2
                if new_coords[coord_y_index] < drag_y_lim_1:
                    new_coords[coord_y_index] = drag_y_lim_1
                if new_coords[coord_y_index] > drag_y_lim_2:
                    new_coords[coord_y_index] = drag_y_lim_2

            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, tuple(new_coords))
        elif self.variables.active_tool == TOOLS.ZOOM_IN_TOOL:
            self.event_drag_line(event)
        elif self.variables.active_tool == TOOLS.ZOOM_OUT_TOOL:
            self.event_drag_line(event)
        elif self.variables.active_tool == TOOLS.SELECT_TOOL:
            self.event_drag_line(event)
        elif self.variables.active_tool == TOOLS.DRAW_RECT_BY_DRAGGING:
            self.event_drag_line(event)
        elif self.variables.active_tool == TOOLS.DRAW_LINE_BY_DRAGGING:
            self.event_drag_line(event)
        elif self.variables.active_tool == TOOLS.DRAW_ARROW_BY_DRAGGING:
            self.event_drag_line(event)
        elif self.variables.active_tool == TOOLS.DRAW_POINT_BY_CLICKING:
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (event.x, event.y))

    def highlight_existing_shape(self, shape_id):
        original_color = self._get_shape_property(shape_id, SHAPE_PROPERTIES.COLOR)
        colors = color_utils.get_full_hex_palette(self.variables.highlight_color_palette, self.variables.highlight_n_colors_cycle)
        for color in colors:
            self.change_shape_color(shape_id, color)
            time.sleep(0.001)
            self.canvas.update()
        colors.reverse()
        for color in colors:
            self.change_shape_color(shape_id, color)
            time.sleep(0.001)
            self.canvas.update()
        self.change_shape_color(shape_id, original_color)

    def callback_handle_right_mouse_click(self, event):
        if self.variables.active_tool == TOOLS.DRAW_LINE_BY_CLICKING:
            self.variables.actively_drawing_shape = False
        elif self.variables.active_tool == TOOLS.DRAW_ARROW_BY_CLICKING:
            self.variables.actively_drawing_shape = False
        elif self.variables.active_tool == TOOLS.DRAW_POLYGON_BY_CLICKING:
            self.variables.actively_drawing_shape = False

    def set_image_from_numpy_array(self,
                                   numpy_data,                      # type: numpy.ndarray
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
            numpy_data = numpy.asanyarray(numpy_data, dtype=numpy.int8)
        pil_image = PIL.Image.fromarray(numpy_data)
        self._set_image_from_pil_image(pil_image)

    def set_canvas_size(self,
                        width_npix,          # type: int
                        height_npix,         # type: int
                        ):
        self.canvas_width = width_npix
        self.canvas_height = height_npix
        self.canvas.config(width=width_npix, height=height_npix)

    def modify_existing_shape_using_canvas_coords(self,
                                                  shape_id,  # type: int
                                                  new_coords,  # type: tuple
                                                  update_pixel_coords=True,         # type: bool
                                                  ):
        if self.get_shape_type(shape_id) == SHAPE_TYPES.POINT:
            point_size = self._get_shape_property(shape_id, SHAPE_PROPERTIES.POINT_SIZE)
            x1, y1 = (new_coords[0] - point_size), (new_coords[1] - point_size)
            x2, y2 = (new_coords[0] + point_size), (new_coords[1] + point_size)
            canvas_drawing_coords = (x1, y1, x2, y2)
        else:
            canvas_drawing_coords = tuple(new_coords)
        self.canvas.coords(shape_id, canvas_drawing_coords)
        self.set_shape_canvas_coords(shape_id, new_coords)
        if update_pixel_coords:
            self.set_shape_pixel_coords_from_canvas_coords(shape_id)

    def modify_existing_shape_using_image_coords(self,
                                                  shape_id,  # type: int
                                                  image_coords,  # type: tuple
                                                  ):
        self.set_shape_pixel_coords(shape_id, image_coords)
        canvas_coords = self.image_coords_to_canvas_coords(shape_id)
        self.modify_existing_shape_using_canvas_coords(shape_id, canvas_coords, update_pixel_coords=False)

    def event_drag_multipoint_line(self, event):
        if self.variables.current_shape_id:
            self.show_shape(self.variables.current_shape_id)
            event_x_pos = self.canvas.canvasx(event.x)
            event_y_pos = self.canvas.canvasy(event.y)
            coords = self.canvas.coords(self.variables.current_shape_id)
            new_coords = list(coords[0:-2]) + [event_x_pos, event_y_pos]
            if self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.ARROW or self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.LINE:
                self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, new_coords)
        else:
            pass

    def event_drag_multipoint_polygon(self, event):
        if self.variables.current_shape_id:
            self.show_shape(self.variables.current_shape_id)
            event_x_pos = self.canvas.canvasx(event.x)
            event_y_pos = self.canvas.canvasy(event.y)
            coords = self.canvas.coords(self.variables.current_shape_id)
            new_coords = list(coords[0:-2]) + [event_x_pos, event_y_pos]
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, new_coords)
        else:
            pass

    def event_drag_line(self, event):
        if self.variables.current_shape_id:
            self.show_shape(self.variables.current_shape_id)
            event_x_pos = self.canvas.canvasx(event.x)
            event_y_pos = self.canvas.canvasy(event.y)
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (self.variables.current_shape_canvas_anchor_point_xy[0], self.variables.current_shape_canvas_anchor_point_xy[1], event_x_pos, event_y_pos))

    def event_drag_rect(self, event):
        if self.variables.current_shape_id:
            self.show_shape(self.variables.current_shape_id)
            event_x_pos = self.canvas.canvasx(event.x)
            event_y_pos = self.canvas.canvasy(event.y)
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (self.variables.current_shape_canvas_anchor_point_xy[0], self.variables.current_shape_canvas_anchor_point_xy[1], event_x_pos, event_y_pos))

    def event_click_line(self, event):
        if self.variables.actively_drawing_shape:
            old_coords = self.get_shape_canvas_coords(self.variables.current_shape_id)
            new_coords = tuple(list(old_coords) + [event.x, event.y])
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, new_coords)
        else:
            new_coords = (event.x, event.y, event.x+1, event.y+1)
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, new_coords)
            self.variables.actively_drawing_shape = True

    def delete_shape(self, shape_id):
        self.variables.shape_ids.remove(shape_id)
        self.canvas.delete(shape_id)
        if shape_id == self.variables.current_shape_id:
            self.variables.current_shape_id = None

    def event_click_polygon(self, event):
        if self.variables.actively_drawing_shape:
            old_coords = self.get_shape_canvas_coords(self.variables.current_shape_id)
            new_coords = tuple(list(old_coords) + [event.x, event.y])
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, new_coords)
        # re-initialize shape if we're not actively drawing
        else:
            new_coords = (event.x, event.y, event.x+1, event.y+1)
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, new_coords)
            self.variables.actively_drawing_shape = True

    def create_new_rect(self,
                        coords,         # type: (int, int, int, int)
                        **options
                        ):
        if options == {}:
            shape_id = self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3],
                                                    outline=self.variables.foreground_color,
                                                    width=self.variables.rect_border_width)
        else:
            shape_id = self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3], options)

        self.variables.shape_ids.append(shape_id)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.SHAPE_TYPE, SHAPE_TYPES.RECT)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.COLOR, self.variables.foreground_color)
        self.set_shape_canvas_coords(shape_id, coords)
        self.set_shape_pixel_coords_from_canvas_coords(shape_id)
        self.variables.current_shape_id = shape_id
        return shape_id

    def create_new_polygon(self,
                           coords,  # type: (int, int, int, int)
                           **options
                           ):
        if options == {}:
            shape_id = self.canvas.create_polygon(coords[0], coords[1], coords[2], coords[3],
                                                  outline=self.variables.foreground_color,
                                                  width=self.variables.poly_border_width,
                                                  fill='')
        else:
            shape_id = self.canvas.create_polygon(coords[0], coords[1], coords[2], coords[3], options)

        self.variables.shape_ids.append(shape_id)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.SHAPE_TYPE, SHAPE_TYPES.POLYGON)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.COLOR, self.variables.foreground_color)
        self.set_shape_canvas_coords(shape_id, coords)
        self.set_shape_pixel_coords_from_canvas_coords(shape_id)
        self.variables.current_shape_id = shape_id
        return shape_id

    def create_new_arrow(self,
                         coords,
                         **options
                         ):
        if options == {}:
            shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3],
                                               fill=self.variables.foreground_color,
                                               width=self.variables.line_width,
                                               arrow=tkinter.LAST)
        else:
            shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3], options, arrow=tkinter.LAST)
        self.variables.shape_ids.append(shape_id)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.SHAPE_TYPE, SHAPE_TYPES.ARROW)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.COLOR, self.variables.foreground_color)
        self.set_shape_canvas_coords(shape_id, coords)
        self.set_shape_pixel_coords_from_canvas_coords(shape_id)
        self.variables.current_shape_id = shape_id
        return shape_id

    def create_new_line(self, coords, **options):
        if options == {}:
            shape_id = self.canvas.create_line(coords,
                                               fill=self.variables.foreground_color,
                                               width=self.variables.line_width)
        else:
            shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3], options)
        self.variables.shape_ids.append(shape_id)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.SHAPE_TYPE, SHAPE_TYPES.LINE)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.COLOR, self.variables.foreground_color)
        self.set_shape_canvas_coords(shape_id, coords)
        self.set_shape_pixel_coords_from_canvas_coords(shape_id)
        self.variables.current_shape_id = shape_id
        return shape_id

    def create_new_point(self,
                         coords,
                         **options):
        x1, y1 = (coords[0] - self.variables.point_size), (coords[1] - self.variables.point_size)
        x2, y2 = (coords[0] + self.variables.point_size), (coords[1] + self.variables.point_size)
        if options == {}:
            shape_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.variables.foreground_color)
        else:
            shape_id = self.canvas.create_oval(x1, y1, x2, y2, options)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.POINT_SIZE, self.variables.point_size)

        self.variables.shape_ids.append(shape_id)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.SHAPE_TYPE, self.SHAPE_TYPES.POINT)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.COLOR, self.variables.foreground_color)
        self.set_shape_canvas_coords(shape_id, coords)
        self.set_shape_pixel_coords_from_canvas_coords(shape_id)
        self.variables.current_shape_id = shape_id
        return shape_id

    def change_shape_color(self,
                           shape_id,        # type: int
                           color,           # type: str
                           ):
        shape_type = self.get_shape_type(shape_id)
        if shape_type == SHAPE_TYPES.RECT:
            self.canvas.itemconfig(shape_id, outline=color)
        elif shape_type == SHAPE_TYPES.POLYGON:
            self.canvas.itemconfig(shape_id, outline=color)
        else:
            self.canvas.itemconfig(shape_id, fill=color)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.COLOR, color)

    def set_shape_canvas_coords(self,
                                shape_id,
                                coords):
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.CANVAS_COORDS, coords)

    def set_shape_pixel_coords_from_canvas_coords(self, shape_id):
        if self.variables.canvas_image_object is None:
            self._set_shape_property(shape_id, SHAPE_PROPERTIES.IMAGE_COORDS, None)
        else:
            image_coords = self.canvas_shape_coords_to_image_coords(shape_id)
            self._set_shape_property(shape_id, SHAPE_PROPERTIES.IMAGE_COORDS, image_coords)

    def set_shape_pixel_coords(self,
                               shape_id,            # type: int
                               image_coords,        # type: list
                               ):
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.IMAGE_COORDS, image_coords)

    def canvas_shape_coords_to_image_coords(self, shape_id):
        canvas_coords = self.get_shape_canvas_coords(shape_id)
        return self.variables.canvas_image_object.canvas_coords_to_full_image_yx(canvas_coords)

    def get_shape_canvas_coords(self, shape_id):
        return self._get_shape_property(shape_id, SHAPE_PROPERTIES.CANVAS_COORDS)

    def get_shape_image_coords(self, shape_id):
        return self._get_shape_property(shape_id, SHAPE_PROPERTIES.IMAGE_COORDS)

    def image_coords_to_canvas_coords(self, shape_id):
        image_coords = self.get_shape_image_coords(shape_id)
        return self.variables.canvas_image_object.full_image_yx_to_canvas_coords(image_coords)

    def get_image_data_in_canvas_rect_by_id(self, rect_id, decimation=None):
        image_coords = self.get_shape_image_coords(rect_id)
        if image_coords[0] > image_coords[2]:
            tmp = image_coords[0]
            image_coords[0] = image_coords[2]
            image_coords[2] = tmp
        if image_coords[1] > image_coords[3]:
            tmp = image_coords[1]
            image_coords[1] = image_coords[3]
            image_coords[3] = tmp
        if decimation is None:
            decimation = self.variables.canvas_image_object.get_decimation_factor_from_full_image_rect(image_coords)
        image_data_in_rect = self.variables.canvas_image_object.get_decimated_image_data_in_full_image_rect(image_coords, decimation)
        return image_data_in_rect

    def zoom_to_selection(self, canvas_rect, animate=False):
        self.variables.the_canvas_is_currently_zooming = True
        # fill up empty canvas space due to inconsistent ratios between the canvas rect and the canvas dimensions
        image_coords = self.variables.canvas_image_object.canvas_coords_to_full_image_yx(canvas_rect)

        zoomed_image_height = image_coords[2] - image_coords[0]
        zoomed_image_width = image_coords[3] - image_coords[1]

        canvas_height_width_ratio = self.canvas_height / self.canvas_width
        zoomed_image_height_width_ratio = zoomed_image_height / zoomed_image_width

        new_image_width = zoomed_image_height / canvas_height_width_ratio
        new_image_height = zoomed_image_width * canvas_height_width_ratio

        if zoomed_image_height_width_ratio > canvas_height_width_ratio:
            image_zoom_point_center = (image_coords[3] + image_coords[1]) / 2
            image_coords[1] = image_zoom_point_center - new_image_width/2
            image_coords[3] = image_zoom_point_center + new_image_width/2
        else:
            image_zoom_point_center = (image_coords[2] + image_coords[0]) / 2
            image_coords[0] = image_zoom_point_center - new_image_height / 2
            image_coords[2] = image_zoom_point_center + new_image_height / 2

        # keep the rect within the image bounds
        image_y_ul = max(image_coords[0], 0)
        image_x_ul = max(image_coords[1], 0)
        image_y_br = min(image_coords[2], self.variables.canvas_image_object.image_reader.full_image_ny)
        image_x_br = min(image_coords[3], self.variables.canvas_image_object.image_reader.full_image_nx)

        # re-adjust if we ran off one of the edges
        if image_x_ul == 0:
            image_coords[3] = new_image_width
        if image_x_br == self.variables.canvas_image_object.image_reader.full_image_nx:
            image_coords[1] = self.variables.canvas_image_object.image_reader.full_image_nx - new_image_width
        if image_y_ul == 0:
            image_coords[2] = new_image_height
        if image_y_br == self.variables.canvas_image_object.image_reader.full_image_ny:
            image_coords[0] = self.variables.canvas_image_object.image_reader.full_image_ny - new_image_height

        # keep the rect within the image bounds
        image_y_ul = max(image_coords[0], 0)
        image_x_ul = max(image_coords[1], 0)
        image_y_br = min(image_coords[2], self.variables.canvas_image_object.image_reader.full_image_ny)
        image_x_br = min(image_coords[3], self.variables.canvas_image_object.image_reader.full_image_nx)

        new_canvas_rect = self.variables.canvas_image_object.full_image_yx_to_canvas_coords((image_y_ul, image_x_ul, image_y_br, image_x_br))
        new_canvas_rect = (int(new_canvas_rect[0]), int(new_canvas_rect[1]), int(new_canvas_rect[2]), int(new_canvas_rect[3]))

        background_image = self.variables.canvas_image_object.display_image
        self.variables.canvas_image_object.update_canvas_display_image_from_canvas_rect(new_canvas_rect)
        if self.rescale_image_to_fit_canvas:
            new_image = PIL.Image.fromarray(self.variables.canvas_image_object.display_image)
        else:
            new_image = PIL.Image.fromarray(self.variables.canvas_image_object.canvas_decimated_image)
        if animate is True:
            #create frame sequence
            n_animations = self.variables.n_zoom_animations
            background_image = background_image / 2
            background_image = np.asarray(background_image, dtype=np.uint8)
            canvas_x1, canvas_y1, canvas_x2, canvas_y2 = new_canvas_rect
            display_x_ul = min(canvas_x1, canvas_x2)
            display_x_br = max(canvas_x1, canvas_x2)
            display_y_ul = min(canvas_y1, canvas_y2)
            display_y_br = max(canvas_y1, canvas_y2)
            x_diff = new_image.width - (display_x_br - display_x_ul)
            y_diff = new_image.height - (display_y_br - display_y_ul)
            pil_background_image = PIL.Image.fromarray(background_image)
            frame_sequence = []
            for i in range(n_animations):
                new_x_ul = int(display_x_ul * (1 - i/(n_animations-1)))
                new_y_ul = int(display_y_ul * (1 - i/(n_animations-1)))
                new_size_x = int((display_x_br - display_x_ul) + x_diff * (i/(n_animations-1)))
                new_size_y = int((display_y_br - display_y_ul) + y_diff * (i/(n_animations-1)))
                resized_zoom_image = new_image.resize((new_size_x, new_size_y))
                animation_image = pil_background_image.copy()
                animation_image.paste(resized_zoom_image, (new_x_ul, new_y_ul))
                frame_sequence.append(animation_image)
            fps = n_animations / self.variables.animation_time_in_seconds
            self.animate_with_pil_frame_sequence(frame_sequence, frames_per_second=fps)
        if self.rescale_image_to_fit_canvas:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.display_image)
        else:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.canvas_decimated_image)
        self.canvas.update()
        self.redraw_all_shapes()
        self.variables.the_canvas_is_currently_zooming = False

    def update_current_image(self):
        rect = (0, 0, self.canvas_width, self.canvas_height)
        self.variables.canvas_image_object.update_canvas_display_image_from_canvas_rect(rect)
        self.set_image_from_numpy_array(self.variables.canvas_image_object.display_image)
        self.canvas.update()

    def redraw_all_shapes(self):
        for shape_id in self.variables.shape_ids:
            pixel_coords = self._get_shape_property(shape_id, SHAPE_PROPERTIES.IMAGE_COORDS)
            if pixel_coords:
                new_canvas_coords = self.image_coords_to_canvas_coords(shape_id)
                self.modify_existing_shape_using_canvas_coords(shape_id, new_canvas_coords, update_pixel_coords=False)

    def set_current_tool_to_select_closest_shape(self):
        self.variables.active_tool = TOOLS.SELECT_CLOSEST_SHAPE_TOOL
        self.variables.current_tool = TOOLS.SELECT_CLOSEST_SHAPE_TOOL

    def set_current_tool_to_zoom_out(self):
        self.variables.current_shape_id = self.variables.zoom_rect_id
        self.variables.active_tool = TOOLS.ZOOM_OUT_TOOL
        self.variables.current_tool = TOOLS.ZOOM_OUT_TOOL

    def set_current_tool_to_zoom_in(self):
        self.variables.current_shape_id = self.variables.zoom_rect_id
        self.variables.active_tool = TOOLS.ZOOM_IN_TOOL
        self.variables.current_tool = TOOLS.ZOOM_IN_TOOL

    def set_current_tool_to_draw_rect(self, rect_id=None):
        self.variables.current_shape_id = rect_id
        self.show_shape(rect_id)
        self.variables.active_tool = TOOLS.DRAW_RECT_BY_DRAGGING
        self.variables.current_tool = TOOLS.DRAW_RECT_BY_DRAGGING

    def set_current_tool_to_draw_rect_by_clicking(self, rect_id=None):
        self.variables.current_shape_id = rect_id
        self.show_shape(rect_id)
        self.variables.active_tool = TOOLS.DRAW_RECT_BY_CLICKING
        self.variables.current_tool = TOOLS.DRAW_RECT_BY_CLICKING

    def set_current_tool_to_selection_tool(self):
        self.variables.current_shape_id = self.variables.select_rect_id
        self.variables.active_tool = TOOLS.SELECT_TOOL
        self.variables.current_tool = TOOLS.SELECT_TOOL

    def set_current_tool_to_draw_line_by_dragging(self, line_id=None):
        self.variables.current_shape_id = line_id
        self.show_shape(line_id)
        self.variables.active_tool = TOOLS.DRAW_LINE_BY_DRAGGING
        self.variables.current_tool = TOOLS.DRAW_LINE_BY_DRAGGING

    def set_current_tool_to_draw_line_by_clicking(self, line_id=None):
        self.variables.current_shape_id = line_id
        self.show_shape(line_id)
        self.variables.active_tool = TOOLS.DRAW_LINE_BY_CLICKING
        self.variables.current_tool = TOOLS.DRAW_LINE_BY_CLICKING

    def set_current_tool_to_draw_arrow_by_dragging(self, arrow_id=None):
        self.variables.current_shape_id = arrow_id
        self.show_shape(arrow_id)
        self.variables.active_tool = TOOLS.DRAW_ARROW_BY_DRAGGING
        self.variables.current_tool = TOOLS.DRAW_ARROW_BY_DRAGGING

    def set_current_tool_to_draw_arrow_by_clicking(self, line_id=None):
        self.variables.current_shape_id = line_id
        self.show_shape(line_id)
        self.variables.active_tool = TOOLS.DRAW_ARROW_BY_CLICKING
        self.variables.current_tool = TOOLS.DRAW_ARROW_BY_CLICKING

    def set_current_tool_to_draw_polygon_by_clicking(self, polygon_id=None):
        self.variables.current_shape_id = polygon_id
        self.show_shape(polygon_id)
        self.variables.active_tool = TOOLS.DRAW_POLYGON_BY_CLICKING
        self.variables.current_tool = TOOLS.DRAW_POLYGON_BY_CLICKING

    def set_current_tool_to_draw_point(self, point_id=None):
        self.variables.current_shape_id = point_id
        self.show_shape(point_id)
        self.variables.active_tool = TOOLS.DRAW_POINT_BY_CLICKING
        self.variables.current_tool = TOOLS.DRAW_POINT_BY_CLICKING

    def set_current_tool_to_translate_shape(self):
        self.variables.active_tool = TOOLS.TRANSLATE_SHAPE_TOOL
        self.variables.current_tool = TOOLS.TRANSLATE_SHAPE_TOOL

    def set_current_tool_to_none(self):
        self.variables.active_tool = None
        self.variables.current_tool = None

    def set_current_tool_to_edit_shape(self):
        self.variables.active_tool = TOOLS.EDIT_SHAPE_TOOL
        self.variables.current_tool = TOOLS.EDIT_SHAPE_TOOL

    def set_current_tool_to_edit_shape_coords(self):
        self.variables.active_tool = TOOLS.EDIT_SHAPE_COORDS_TOOL
        self.variables.current_tool = TOOLS.EDIT_SHAPE_COORDS_TOOL

    def set_current_tool_to_pan(self):
        self.variables.active_tool = TOOLS.PAN_TOOL
        self.variables.current_tool = TOOLS.PAN_TOOL

    def _set_image_from_pil_image(self, pil_image):
        nx_pix, ny_pix = pil_image.size
        self.canvas.config(scrollregion=(0, 0, nx_pix, ny_pix))
        self._tk_im = ImageTk.PhotoImage(pil_image)
        # TODO: to center the image in the canvas the following line should be:
        # self.variables.image_id = self.canvas.create_image(int(self.canvas_width/2), int(self.canvas_height/2), anchor="center", image=self._tk_im)
        # TODO: changing to center on the canvas will create downstream book-keeping issues when drawing shapes and converting
        # TODO: between image and canvas coordinates, which will need to be cleaned up.
        self.variables.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self._tk_im)
        self.canvas.tag_lower(self.variables.image_id)

    def _get_shape_property(self,
                            shape_id,  # type: int
                            shape_property,  # type: str
                            ):
        properties = self.variables.shape_properties[str(shape_id)]
        return properties[shape_property]

    def _set_shape_property(self,
                            shape_id,  # type: int
                            shape_property,  # type: str
                            val,
                            ):
        if not str(shape_id) in self.variables.shape_properties.keys():
            self.variables.shape_properties[str(shape_id)] = {}
        self.variables.shape_properties[str(shape_id)][shape_property] = val

    def _update_shape_properties(self,
                                 shape_id,  # type: int
                                 properties,  # type: dict
                                 ):
        for key in properties.keys():
            val = properties[key]
            self._set_shape_property(shape_id, key, val)

    def _pan(self, event):
        new_canvas_x_ul = self.variables.pan_anchor_point_xy[0] - event.x
        new_canvas_y_ul = self.variables.pan_anchor_point_xy[1] - event.y
        new_canvas_x_br = new_canvas_x_ul + self.canvas_width
        new_canvas_y_br = new_canvas_y_ul + self.canvas_height
        canvas_coords = (new_canvas_x_ul, new_canvas_y_ul, new_canvas_x_br, new_canvas_y_br)
        image_coords = self.variables.canvas_image_object.canvas_coords_to_full_image_yx(canvas_coords)
        image_y_ul = image_coords[0]
        image_x_ul = image_coords[1]
        image_y_br = image_coords[2]
        image_x_br = image_coords[3]
        if image_y_ul < 0:
            new_canvas_y_ul = 0
            new_canvas_y_br = self.canvas_height
        if image_x_ul < 0:
            new_canvas_x_ul = 0
            new_canvas_x_br = self.canvas_width
        if image_y_br > self.variables.canvas_image_object.image_reader.full_image_ny:
            image_y_br = self.variables.canvas_image_object.image_reader.full_image_ny
            new_canvas_x_br, new_canvas_y_br = self.variables.canvas_image_object.full_image_yx_to_canvas_coords(
                (image_y_br, image_x_br))
            new_canvas_x_ul, new_canvas_y_ul = int(new_canvas_x_br - self.canvas_width), int(
                new_canvas_y_br - self.canvas_height)
        if image_x_br > self.variables.canvas_image_object.image_reader.full_image_nx:
            image_x_br = self.variables.canvas_image_object.image_reader.full_image_nx
            new_canvas_x_br, new_canvas_y_br = self.variables.canvas_image_object.full_image_yx_to_canvas_coords(
                (image_y_br, image_x_br))
            new_canvas_x_ul, new_canvas_y_ul = int(new_canvas_x_br - self.canvas_width), int(
                new_canvas_y_br - self.canvas_height)

        canvas_rect = (new_canvas_x_ul, new_canvas_y_ul, new_canvas_x_br, new_canvas_y_br)
        self.zoom_to_selection(canvas_rect, self.variables.animate_pan)
        self.hide_shape(self.variables.zoom_rect_id)

    def config_do_not_scale_image_to_fit(self):
        self.sbarv=tkinter.Scrollbar(self, orient=tkinter.VERTICAL)
        self.sbarh=tkinter.Scrollbar(self, orient=tkinter.HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)
        self.sbarv.grid(row=0, column=1, stick=tkinter.N+tkinter.S)
        self.sbarh.grid(row=1, column=0, sticky=tkinter.E+tkinter.W)

    def save_as_png(self,
                    output_fname,           # type: str
                    ):
        # put a sleep in here in case there is a dialog covering the screen before this method is called.
        time.sleep(0.2)
        im = self.save_currently_displayed_canvas_to_numpy_array()
        im.save(output_fname)

    def save_currently_displayed_canvas_to_numpy_array(self):
        x_ul = self.canvas.winfo_rootx() + 1
        y_ul = self.canvas.winfo_rooty() + 1
        x_lr = x_ul + self.canvas_width
        y_lr = y_ul + self.canvas_height
        im = ImageGrab.grab()
        im = im.crop((x_ul, y_ul, x_lr, y_lr))
        return im

    def activate_color_selector(self, event):
        color = colorchooser.askcolor()[1]
        self.variables.foreground_color = color
        self.change_shape_color(self.variables.current_shape_id, color)

    def find_closest_shape_coord(self,
                                 shape_id,          # type: int
                                 canvas_x,          # type: int
                                 canvas_y,          # type: int
                                 ):                 # type: (...) -> int
        shape_type = self.get_shape_type(self.variables.current_shape_id)
        coords = self.get_shape_canvas_coords(shape_id)
        if shape_type == SHAPE_TYPES.RECT:
            select_x1, select_y1, select_x2, select_y2 = coords
            select_xul = min(select_x1, select_x2)
            select_xlr = max(select_x1, select_x2)
            select_yul = min(select_y1, select_y2)
            select_ylr = max(select_y1, select_y2)

            ul = (select_xul, select_yul)
            ur = (select_xlr, select_yul)
            lr = (select_xlr, select_ylr)
            ll = (select_xul, select_ylr)

            rect_coords = [(select_x1, select_y1), (select_x2, select_y2)]

            all_coords = [ul, ur, lr, ll]

            squared_distances = []
            for corner_coord in all_coords:
                coord_x, coord_y = corner_coord
                d = (coord_x - canvas_x)**2 + (coord_y - canvas_y)**2
                squared_distances.append(d)
            closest_coord_index = numpy.where(squared_distances == numpy.min(squared_distances))[0][0]
            closest_coord = all_coords[closest_coord_index]
            if closest_coord not in rect_coords:
                if closest_coord == ul:
                    self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (ul[0], ul[1], lr[0], lr[1]))
                if closest_coord == ur:
                    self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (ur[0], ur[1], ll[0], ll[1]))
                if closest_coord == lr:
                    self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (ul[0], ul[1], lr[0], lr[1]))
                if closest_coord == ll:
                    self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (ll[0], ll[1], ur[0], ur[1]))

            coords = self.get_shape_canvas_coords(shape_id)

        squared_distances = []
        coord_indices = numpy.arange(0, len(coords), step=2)
        for i in coord_indices:
            coord_x, coord_y = coords[i], coords[i+1]
            d = (coord_x - canvas_x)**2 + (coord_y - canvas_y)**2
            squared_distances.append(d)
        closest_coord_index = numpy.where(squared_distances == numpy.min(squared_distances))[0][0]
        return closest_coord_index

    # TODO: improve this.  Right now it finds closest shape just based on distance to corners.  Improvements should
    # TODO: include finding a closest point if the x/y coordinate is inside a polygon, and also finding closest
    # TODO: distance to each line of a polygon, not just the corners.
    def find_closest_shape(self,
                           canvas_x,
                           canvas_y):
        non_tool_shape_ids = self.get_non_tool_shape_ids()
        closest_distances = []
        for shape_id in non_tool_shape_ids:
            coords = self.get_shape_canvas_coords(shape_id)
            squared_distances = []
            coord_indices = numpy.arange(0, len(coords), step=2)
            for i in coord_indices:
                coord_x, coord_y = coords[i], coords[i + 1]
                d = (coord_x - canvas_x) ** 2 + (coord_y - canvas_y) ** 2
                squared_distances.append(d)
            closest_distances.append(numpy.min(squared_distances))
        closest_shape_id = non_tool_shape_ids[numpy.where(closest_distances == numpy.min(closest_distances))[0][0]]
        return closest_shape_id

    def get_non_tool_shape_ids(self):
        all_shape_ids = self.variables.shape_ids
        tool_shape_ids = self.get_tool_shape_ids()
        return list(numpy.setdiff1d(all_shape_ids, tool_shape_ids))

    def get_tool_shape_ids(self):
        tool_shape_ids = [self.variables.zoom_rect_id,
                          self.variables.select_rect_id]
        return tool_shape_ids


class CanvasImage(object):
    image_reader = None                 # type: ImageReader
    canvas_decimated_image = None       # type: ndarray
    display_image = None                # type: ndarray
    decimation_factor = 1               # type: int
    display_rescaling_factor = 1                # type: float
    canvas_full_image_upper_left_yx = (0, 0)    # type: (int, int)
    canvas_ny = None                    # type: int
    canvas_nx = None                    # type: int
    scale_to_fit_canvas = True          # type: bool
    drop_bands = []                     # type: []

    def __init__(self,
                 image_reader,          # type: ImageReader
                 canvas_nx,             # type: int
                 canvas_ny,             # type: int
                 ):
        self.image_reader = image_reader
        self.canvas_nx = canvas_nx
        self.canvas_ny = canvas_ny
        self.update_canvas_display_image_from_full_image()

    def get_decimated_image_data_in_full_image_rect(self,
                                                    full_image_rect,  # type: (int, int, int, int)
                                                    decimation,  # type: int
                                                    ):
        print("decimation")
        print(decimation)
        y_start = full_image_rect[0]
        y_end = full_image_rect[2]
        x_start = full_image_rect[1]
        x_end = full_image_rect[3]

        decimated_data = self.image_reader[y_start:y_end:decimation, x_start:x_end:decimation]
        return decimated_data

    def get_scaled_display_data(self, decimated_image):
        scale_factor = self.compute_display_scale_factor(decimated_image)
        new_nx = int(decimated_image.shape[1] * scale_factor)
        new_ny = int(decimated_image.shape[0] * scale_factor)
        if new_nx > self.canvas_nx:
            new_nx = self.canvas_nx
        if new_ny > self.canvas_ny:
            new_ny = self.canvas_ny
        if self.drop_bands != []:
            zeros_image = numpy.zeros_like(decimated_image[:, :, 0])
            for drop_band in self.drop_bands:
                decimated_image[:, :, drop_band] = zeros_image
        pil_image = PIL.Image.fromarray(decimated_image)
        display_image = pil_image.resize((new_nx, new_ny))
        return np.array(display_image)

    def decimated_image_coords_to_display_image_coords(self,
                                                       decimated_image_yx_cords,  # type: list
                                                       ):
        scale_factor = self.compute_display_scale_factor(self.canvas_decimated_image)
        display_coords = []
        for coord in decimated_image_yx_cords:
            display_coord_y = coord[0] * scale_factor
            display_coord_x = coord[1] * scale_factor
            display_coords.append((display_coord_y, display_coord_x))
        return display_coords

    def display_image_coords_to_decimated_image_coords(self,
                                                       display_image_yx_coords,  # type: list
                                                       ):
        scale_factor = self.compute_display_scale_factor(self.canvas_decimated_image)
        decimated_coords = []
        for coord in display_image_yx_coords:
            display_coord_y = coord[0] / scale_factor
            display_coord_x = coord[1] / scale_factor
            decimated_coords.append((display_coord_y, display_coord_x))
        return decimated_coords

    @staticmethod
    def display_image_coords_to_canvas_coords(display_image_yx_coords,      # type: list
                                              ):
        canvas_coords = []
        for yx in display_image_yx_coords:
            canvas_coords.append((yx[1], yx[0]))
        return canvas_coords

    def compute_display_scale_factor(self, decimated_image):
        decimated_image_nx = decimated_image.shape[1]
        decimated_image_ny = decimated_image.shape[0]
        scale_factor_1 = self.canvas_nx / decimated_image_nx
        scale_factor_2 = self.canvas_ny / decimated_image_ny
        scale_factor = np.min((scale_factor_1, scale_factor_2))
        return scale_factor

    def get_decimated_image_data_in_canvas_rect(self,
                                                canvas_rect,  # type: (int, int, int, int)
                                                decimation=None,  # type: int
                                                ):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        if decimation is None:
            decimation = self.get_decimation_from_canvas_rect(canvas_rect)
        return self.get_decimated_image_data_in_full_image_rect(full_image_rect, decimation)

    def update_canvas_display_image_from_full_image(self):
        full_image_rect = (0, 0, self.image_reader.full_image_ny, self.image_reader.full_image_nx)
        self.update_canvas_display_image_from_full_image_rect(full_image_rect)

    def update_canvas_display_image_from_full_image_rect(self, full_image_rect):
        self.set_decimation_from_full_image_rect(full_image_rect)
        decimated_image_data = self.get_decimated_image_data_in_full_image_rect(full_image_rect, self.decimation_factor)
        self.update_canvas_display_from_numpy_array(decimated_image_data)
        self.canvas_full_image_upper_left_yx = (full_image_rect[0], full_image_rect[1])

    def update_canvas_display_image_from_canvas_rect(self, canvas_rect):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        full_image_rect = (int(round(full_image_rect[0])),
                           int(round(full_image_rect[1])),
                           int(round(full_image_rect[2])),
                           int(round(full_image_rect[3])))
        self.update_canvas_display_image_from_full_image_rect(full_image_rect)

    def update_canvas_display_from_numpy_array(self,
                                               image_data,  # type: ndarray
                                               ):
        if self.drop_bands != []:
            zeros_image = numpy.zeros_like(image_data[:, :, 0])
            for drop_band in self.drop_bands:
                image_data[:, :, drop_band] = zeros_image
        self.canvas_decimated_image = image_data
        if self.scale_to_fit_canvas:
            scale_factor = self.compute_display_scale_factor(image_data)
            self.display_rescaling_factor = scale_factor
            self.display_image = self.get_scaled_display_data(image_data)
        else:
            self.display_image = image_data

    def get_decimation_factor_from_full_image_rect(self, full_image_rect):
        ny = full_image_rect[2] - full_image_rect[0]
        nx = full_image_rect[3] - full_image_rect[1]
        decimation_y = ny / self.canvas_ny
        decimation_x = nx / self.canvas_nx
        decimation_factor = max(decimation_y, decimation_x)
        decimation_factor = int(decimation_factor)
        if decimation_factor < 1:
            decimation_factor = 1
        return decimation_factor

    def get_decimation_from_canvas_rect(self, canvas_rect):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        return self.get_decimation_factor_from_full_image_rect(full_image_rect)

    def set_decimation_from_full_image_rect(self, full_image_rect):
        decimation_factor = self.get_decimation_factor_from_full_image_rect(full_image_rect)
        self.decimation_factor = decimation_factor

    def canvas_coords_to_full_image_yx(self,
                                       canvas_coords,       # type: [int]
                                       ):
        x_coords = canvas_coords[0::2]
        y_coords = canvas_coords[1::2]
        xy_coords = zip(x_coords, y_coords)
        image_yx_coords = []
        for xy in xy_coords:
            decimation_factor = self.decimation_factor
            if self.scale_to_fit_canvas:
                decimation_factor = decimation_factor / self.display_rescaling_factor
            image_x = xy[0] * decimation_factor + self.canvas_full_image_upper_left_yx[1]
            image_y = xy[1] * decimation_factor + self.canvas_full_image_upper_left_yx[0]
            image_yx_coords.append(image_y)
            image_yx_coords.append(image_x)
        return image_yx_coords

    def canvas_rect_to_full_image_rect(self,
                                       canvas_rect,  # type: (int, int, int, int)
                                       ):            # type: (...) ->[float]
        image_y1, image_x1 = self.canvas_coords_to_full_image_yx((canvas_rect[0], canvas_rect[1]))
        image_y2, image_x2 = self.canvas_coords_to_full_image_yx((canvas_rect[2], canvas_rect[3]))

        if image_x1 < 0:
            image_x1 = 0
        if image_y1 < 0:
            image_y1 = 0
        if image_x2 > self.image_reader.full_image_nx:
            image_x2 = self.image_reader.full_image_nx
        if image_y2 > self.image_reader.full_image_ny:
            image_y2 = self.image_reader.full_image_ny

        return image_y1, image_x1, image_y2, image_x2

    def full_image_yx_to_canvas_coords(self,
                                       full_image_yx,           # type: Union[(int, int), list]
                                       ):                       # type: (...) -> Union[(int, int), list]
        y_coords = full_image_yx[0::2]
        x_coords = full_image_yx[1::2]
        image_yx_coords = zip(y_coords, x_coords)
        canvas_xy_coords = []
        decimation_factor = self.decimation_factor
        if self.scale_to_fit_canvas:
            decimation_factor = decimation_factor / self.display_rescaling_factor
        for image_yx in image_yx_coords:
            canvas_x = (image_yx[1] - self.canvas_full_image_upper_left_yx[1]) / decimation_factor
            canvas_y = (image_yx[0] - self.canvas_full_image_upper_left_yx[0]) / decimation_factor
            canvas_xy_coords.append(canvas_x)
            canvas_xy_coords.append(canvas_y)
        return canvas_xy_coords
