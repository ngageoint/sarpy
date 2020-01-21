import PIL.Image
from PIL import ImageTk
import tkinter as tk
from tkinter_gui_builder.widgets import basic_widgets
from tkinter_gui_builder.canvas_image_objects.abstract_canvas_image import AbstractCanvasImage
from tkinter_gui_builder.canvas_image_objects.numpy_canvas_image import NumpyCanvasDisplayImage
import platform
import numpy as np
import time
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

        self.foreground_color = "red"
        self.rect_border_width = 2

        self.image_id = None                # type: int

        self.current_shape_id = None
        self.current_shape_canvas_anchor_point_xy = None
        self.pan_anchor_point_xy = None
        self.shape_ids = []            # type: [int]
        self.shape_properties = {}
        self.canvas_image_object = None         # type: AbstractCanvasImage
        self.zoom_rect_id = None                # type: int
        self.zoom_rect_color = "cyan"
        self.zoom_rect_border_width = 2

        self.animate_zoom = True
        self.animate_pan = False
        self.n_zoom_animations = 50
        self.animation_time_in_seconds = 0.3

        self.select_rect_id = None
        self.select_rect_color = "red"
        self.select_rect_border_width = 2
        self.current_tool = None

        self.pan_anchor_point_xy = (0, 0)


class ImageCanvas(tk.LabelFrame):

    def __init__(self, master):
        tk.LabelFrame.__init__(self, master)

        self.SHAPE_PROPERTIES = SHAPE_PROPERTIES
        self.SHAPE_TYPES = SHAPE_TYPES
        self.TOOLS = TOOLS

        self.variables = AppVariables()

        self.scale_dynamic_range = False
        self.canvas_height = 200            # default width
        self.canvas_width = 300             # default height
        self.canvas = basic_widgets.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.pack()

        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.sbarv = None         # type: tk.Scrollbar
        self.sbarh = None         # type: tk.Scrollbar

        self.variables.zoom_rect_id = self.create_new_rect((0, 0, 1, 1), outline=self.variables.zoom_rect_color, width=self.variables.zoom_rect_border_width)
        self.variables.select_rect_id = self.create_new_rect((0, 0, 1, 1), outline=self.variables.select_rect_color, width=self.variables.select_rect_border_width)

        # hide the shapes we initialize
        self.hide_shape(self.variables.select_rect_id)
        self.hide_shape(self.variables.zoom_rect_id)

        self.canvas.on_left_mouse_click(self.callback_handle_left_mouse_click)
        self.canvas.on_left_mouse_motion(self.callback_handle_left_mouse_motion)
        self.canvas.on_left_mouse_release(self.callback_handle_left_mouse_release)

        self.canvas.on_mouse_wheel(self.callback_mouse_zoom)

        self.variables.current_tool = None
        self.variables.current_shape_id = None

        self.zoom_on_wheel = True

        self._tk_im = None               # type: ImageTk.PhotoImage

        self.rescale_image_to_fit_canvas = True

    def init_with_fname(self,
                        fname,  # type: str
                        ):
        self.variables.canvas_image_object.init_from_fname_and_canvas_size(fname, self.canvas_height, self.canvas_width, scale_to_fit_canvas=self.rescale_image_to_fit_canvas)
        self.variables.canvas_image_object.scale_to_fit_canvas = self.rescale_image_to_fit_canvas
        if self.rescale_image_to_fit_canvas:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.display_image)
        else:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.canvas_decimated_image)

    def init_with_numpy_image(self,
                              numpy_array,      # type: np.ndarray
                              ):
        self.variables.canvas_image_object = NumpyCanvasDisplayImage()
        self.variables.canvas_image_object.scale_to_fit_canvas = self.rescale_image_to_fit_canvas
        self.variables.canvas_image_object.init_from_numpy_array_and_canvas_size(numpy_array, self.canvas_height, self.canvas_width)
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
        length = np.sqrt(np.square(x2-x1) + np.square(y2-y1))
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
        # TODO: This kind of works.  Change it so that the zoom location moves to the mouse, rather than the center
        # TODO: of the image.  This will also require a refactor to make the image fit to the canvas, rather than
        # TODO: go off the edge and use scroll bars.  Could also be made smoother, handle multiple events for a single
        # TODO: long mouse scroll

        if self.zoom_on_wheel:
            delta = event.delta
            single_delta = 120

            # handle case where platform is linux:
            if platform.system() == "Linux":
                delta = single_delta
                if event.num == 5:
                    delta = delta*-1

            box_percent = 1.20

            zoom_in_box_half_width = int(self.canvas_width / box_percent / 2)
            zoom_out_box_half_width = int(self.canvas_width * box_percent / 2)
            zoom_in_box_half_height = int(self.canvas_height / box_percent / 2)
            zoom_out_box_half_height = int(self.canvas_height * box_percent / 2)

            x = event.x
            y = event.y

            after_zoom_x_offset = (self.canvas_width/2 - x)/box_percent
            after_zoom_y_offset = (self.canvas_height/2 - y)/box_percent

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

            if delta > 0:
                self.zoom_to_selection(zoom_in_box, self.variables.animate_zoom)
            else:
                self.zoom_to_selection(zoom_out_box, self.variables.animate_zoom)
        else:
            pass

    def animate_with_frame_sequence(self,
                                    frame_sequence,         # type: np.ndarray
                                    frames_per_second=15,      # type: float
                                    ):
        sleep_time = 1/frames_per_second
        for animation_frame in frame_sequence:
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

    def callback_handle_left_mouse_release(self, event):
        if self.variables.current_tool == TOOLS.PAN_TOOL:
            self._pan(event)
        if self.variables.current_tool == TOOLS.ZOOM_IN_TOOL:
            rect_coords = self.canvas.coords(self.variables.zoom_rect_id)
            self.zoom_to_selection(rect_coords, self.variables.animate_zoom)
            self.hide_shape(self.variables.zoom_rect_id)
        if self.variables.current_tool == TOOLS.ZOOM_OUT_TOOL:
            rect_coords = self.canvas.coords(self.variables.zoom_rect_id)
            x1 = -rect_coords[0]
            x2 = self.canvas_width + rect_coords[2]
            y1 = -rect_coords[1]
            y2 = self.canvas_height + rect_coords[3]
            zoom_rect = (x1, y1, x2, y2)
            self.zoom_to_selection(zoom_rect, self.variables.animate_zoom)
            self.hide_shape(self.variables.zoom_rect_id)

    def callback_handle_left_mouse_click(self, event):
        if self.variables.current_tool == TOOLS.PAN_TOOL:
            self.variables.pan_anchor_point_xy = event.x, event.y
            self.variables.tmp_anchor_point = event.x, event.y
        elif self.variables.current_tool == TOOLS.TRANSLATE_SHAPE_TOOL:
            self.variables.translate_anchor_point_xy = event.x, event.y
            self.variables.tmp_anchor_point = event.x, event.y
        else:
            self.event_create_or_reinitialize_shape(event)

    def callback_handle_left_mouse_motion(self, event):
        print(self.variables.current_tool)
        if self.variables.current_tool == TOOLS.PAN_TOOL:
            x_dist = event.x - self.variables.tmp_anchor_point[0]
            y_dist = event.y - self.variables.tmp_anchor_point[1]
            self.canvas.move(self.variables.image_id, x_dist, y_dist)
            self.variables.tmp_anchor_point = event.x, event.y
        elif self.variables.current_tool == TOOLS.TRANSLATE_SHAPE_TOOL:
            x_dist = event.x - self.variables.tmp_anchor_point[0]
            y_dist = event.y - self.variables.tmp_anchor_point[1]
            new_x1 = self.get_shape_canvas_coords(self.variables.current_shape_id)[0] + x_dist
            new_y1 = self.get_shape_canvas_coords(self.variables.current_shape_id)[1] + y_dist
            new_x2 = self.get_shape_canvas_coords(self.variables.current_shape_id)[2] + x_dist
            new_y2 = self.get_shape_canvas_coords(self.variables.current_shape_id)[3] + y_dist
            self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (new_x1, new_y1, new_x2, new_y2), update_pixel_coords=True)
            self.variables.tmp_anchor_point = event.x, event.y
        else:
            self.event_drag_shape(event)

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

    def event_create_or_reinitialize_shape(self, event):
        # save mouse drag start position
        start_x = self.canvas.canvasx(event.x)
        start_y = self.canvas.canvasy(event.y)

        coords = (start_x, start_y, start_x + 1, start_y + 1)
        self.variables.current_shape_canvas_anchor_point_xy = (start_x, start_y)

        if self.variables.current_shape_id not in self.variables.shape_ids:
            if self.variables.current_tool == TOOLS.DRAW_LINE_TOOL:
                self.create_new_line(coords)
            elif self.variables.current_tool == TOOLS.DRAW_RECT_TOOL:
                self.create_new_rect(coords)
            elif self.variables.current_tool == TOOLS.DRAW_ARROW_TOOL:
                self.create_new_arrow(coords)
            elif self.variables.current_tool == TOOLS.DRAW_POINT_TOOL:
                self.create_new_point((start_x, start_y))
            else:
                print("no shape tool selected")
        else:
            if self.variables.current_shape_id in self.variables.shape_ids:
                if self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.POINT:
                    self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id,
                                                                   (start_x, start_y))
                else:
                    self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, coords)

    def event_drag_shape(self, event):
        if self.variables.current_shape_id:
            self.show_shape(self.variables.current_shape_id)
            event_x_pos = self.canvas.canvasx(event.x)
            event_y_pos = self.canvas.canvasy(event.y)
            coords = self.canvas.coords(self.variables.current_shape_id)
            if self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.POINT:
                self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (event_x_pos, event_y_pos))
            elif self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.ARROW or self.get_shape_type(self.variables.current_shape_id) == SHAPE_TYPES.LINE:
                self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (self.variables.current_shape_canvas_anchor_point_xy[0], self.variables.current_shape_canvas_anchor_point_xy[1], event_x_pos, event_y_pos))
            else:
                min_x = coords[0]
                min_y = coords[1]
                max_x = coords[2]
                max_y = coords[3]
                # determine which corner we're dragging from
                left_right_corner = None
                top_bottom_corner = None

                # do nothing if the event is at one of the rect boundaries:
                if event_x_pos == min_x or event_x_pos == max_x or event_y_pos == min_y or event_y_pos == max_y:
                    pass
                else:
                    if event_x_pos > min_x and event_x_pos > max_x:
                        left_right_corner = "right"
                    if event_x_pos < min_x and event_x_pos < max_x:
                        left_right_corner = "left"
                    if min_x < event_x_pos < max_x:
                        left_dist = abs(min_x - event_x_pos)
                        right_dist = abs(max_x - event_x_pos)
                        if left_dist < right_dist:
                            left_right_corner = "left"
                        else:
                            left_right_corner = "right"
                    if left_right_corner == "right":
                        max_x = event_x_pos
                        min_x = self.variables.current_shape_canvas_anchor_point_xy[0]
                    else:
                        min_x = event_x_pos
                        max_x = self.variables.current_shape_canvas_anchor_point_xy[0]

                    if event_y_pos > min_y and event_y_pos > max_y:
                        top_bottom_corner = "bottom"
                    if event_y_pos < min_y and event_y_pos < max_y:
                        top_bottom_corner = "top"
                    if min_y < event_y_pos < max_y:
                        top_dist = abs(min_y - event_y_pos)
                        bottom_dist = abs(max_y - event_y_pos)
                        if top_dist < bottom_dist:
                            top_bottom_corner = "top"
                        else:
                            top_bottom_corner = "bottom"
                    if top_bottom_corner == "bottom":
                        max_y = event_y_pos
                        min_y = self.variables.current_shape_canvas_anchor_point_xy[1]
                    else:
                        min_y = event_y_pos
                        max_y = self.variables.current_shape_canvas_anchor_point_xy[1]
                    self.modify_existing_shape_using_canvas_coords(self.variables.current_shape_id, (min_x, min_y, max_x, max_y))
        else:
            pass

    def create_new_rect(self,
                        coords,         # type: (int, int, int, int)
                        **options,
                        ):
        return self.create_new_canvas_shape('rect', coords, **options)

    def create_new_arrow(self,
                         coords,
                         **options,
                         ):
        return self.create_new_canvas_shape('arrow', coords, **options)

    def create_new_point(self,
                         coords,
                         **options):
        return self.create_new_canvas_shape('point', coords, **options)

    def create_new_line(self, coords, **options):
        return self.create_new_canvas_shape('line', coords, **options)

    def create_new_canvas_shape(self,
                                shape_type,  # type: str
                                coords,  # type: tuple
                                **options,
                                ):
        if shape_type == SHAPE_TYPES.RECT:
            if options == {}:
                shape_id = self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3],
                                                       outline=self.variables.foreground_color,
                                                       width=self.variables.rect_border_width)
            else:
                shape_id = self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3], options)
        if shape_type == SHAPE_TYPES.LINE:
            if options == {}:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3],
                                                       fill=self.variables.foreground_color,
                                                       width=self.variables.line_width)
            else:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3], options)
        if shape_type == SHAPE_TYPES.ARROW:
            if options == {}:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3],
                                                       fill=self.variables.foreground_color,
                                                       width=self.variables.line_width,
                                                       arrow=tk.LAST)
            else:
                shape_id = self.canvas.create_line(coords[0], coords[1], coords[2], coords[3], options, arrow=tk.LAST)
        if shape_type == SHAPE_TYPES.POINT:
            x1, y1 = (coords[0] - self.variables.point_size), (coords[1] - self.variables.point_size)
            x2, y2 = (coords[0] + self.variables.point_size), (coords[1] + self.variables.point_size)
            if options == {}:
                shape_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.variables.foreground_color)
            else:
                shape_id = self.canvas.create_oval(x1, y1, x2, y2, options)
            self._set_shape_property(shape_id, SHAPE_PROPERTIES.POINT_SIZE, self.variables.point_size)

        self.variables.shape_ids.append(shape_id)
        self._set_shape_property(shape_id, SHAPE_PROPERTIES.SHAPE_TYPE, shape_type)
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
        else:
            self.canvas.itemconfig(shape_id, fill=color)

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

    def get_image_data_in_canvas_rect_by_id(self, rect_id):
        image_coords = self._get_shape_property(rect_id, SHAPE_PROPERTIES.IMAGE_COORDS)
        decimation_factor = self.variables.canvas_image_object.get_decimation_factor_from_full_image_rect(image_coords)
        image_data_in_rect = self.variables.canvas_image_object.get_decimated_image_data_in_full_image_rect(image_coords, decimation_factor)
        return image_data_in_rect

    def zoom_to_selection(self, canvas_rect, animate=False):
        # TODO: change this so that the canvas rect stays within the image bounds
        background_image = self.variables.canvas_image_object.canvas_decimated_image
        self.variables.canvas_image_object.update_canvas_display_image_from_canvas_rect(canvas_rect)
        if self.rescale_image_to_fit_canvas:
            new_image = PIL.Image.fromarray(self.variables.canvas_image_object.display_image)
        else:
            new_image = PIL.Image.fromarray(self.variables.canvas_image_object.canvas_decimated_image)
        if animate is True:
            #create frame sequence
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
            frame_sequence = []
            for i in range(n_animations):
                new_x_ul = int(display_x_ul * (1 - i/(n_animations-1)))
                new_y_ul = int(display_y_ul * (1 - i/(n_animations-1)))
                new_size_x = int((display_x_br - display_x_ul) + x_diff * (i/(n_animations-1)))
                new_size_y = int((display_y_br - display_y_ul) + y_diff * (i/(n_animations-1)))
                resized_zoom_image = new_image.resize((new_size_x, new_size_y))
                new_display_image.paste(resized_zoom_image, (new_x_ul, new_y_ul))
                animation_frame = np.array(new_display_image)
                frame_sequence.append(animation_frame)
            fps = n_animations / self.variables.animation_time_in_seconds
            self.animate_with_frame_sequence(frame_sequence, frames_per_second=fps)
        if self.rescale_image_to_fit_canvas:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.display_image)
        else:
            self.set_image_from_numpy_array(self.variables.canvas_image_object.canvas_decimated_image)
        self.canvas.update()
        self.redraw_all_shapes()

    def redraw_all_shapes(self):
        for shape_id in self.variables.shape_ids:
            pixel_coords = self._get_shape_property(shape_id, SHAPE_PROPERTIES.IMAGE_COORDS)
            if pixel_coords:
                new_canvas_coords = self.image_coords_to_canvas_coords(shape_id)
                self.modify_existing_shape_using_canvas_coords(shape_id, new_canvas_coords, update_pixel_coords=False)

    def set_current_tool_to_zoom_out(self):
        self.variables.current_shape_id = self.variables.zoom_rect_id
        self.variables.current_tool = TOOLS.ZOOM_OUT_TOOL

    def set_current_tool_to_zoom_in(self):
        self.variables.current_shape_id = self.variables.zoom_rect_id
        self.variables.current_tool = TOOLS.ZOOM_IN_TOOL

    def set_current_tool_to_draw_rect(self, rect_id=None):
        self.variables.current_shape_id = rect_id
        self.variables.current_tool = TOOLS.DRAW_RECT_TOOL
        self.show_shape(rect_id)

    def set_current_tool_to_selection_tool(self):
        self.variables.current_shape_id = self.variables.select_rect_id
        self.variables.current_tool = TOOLS.SELECT_TOOL

    def set_current_tool_to_draw_line(self, line_id=None):
        self.variables.current_shape_id = line_id
        self.variables.current_tool = TOOLS.DRAW_LINE_TOOL
        self.show_shape(line_id)

    def set_current_tool_to_draw_arrow(self, arrow_id=None):
        self.variables.current_shape_id = arrow_id
        self.variables.current_tool = TOOLS.DRAW_ARROW_TOOL
        self.show_shape(arrow_id)

    def set_current_tool_to_draw_point(self, point_id=None):
        self.variables.current_shape_id = point_id
        self.variables.current_tool = TOOLS.DRAW_POINT_TOOL
        self.show_shape(point_id)

    def set_current_tool_to_translate_shape(self):
        self.variables.current_tool = TOOLS.TRANSLATE_SHAPE_TOOL

    def set_current_tool_to_pan(self):
        self.variables.current_tool = TOOLS.PAN_TOOL

    def _set_image_from_pil_image(self, pil_image):
        nx_pix, ny_pix = pil_image.size
        self.canvas.config(scrollregion=(0, 0, nx_pix, ny_pix))
        self._tk_im = ImageTk.PhotoImage(pil_image)
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
        if image_y_br > self.variables.canvas_image_object.full_image_ny:
            image_y_br = self.variables.canvas_image_object.full_image_ny
            new_canvas_x_br, new_canvas_y_br = self.variables.canvas_image_object.full_image_yx_to_canvas_coords(
                (image_y_br, image_x_br))
            new_canvas_x_ul, new_canvas_y_ul = int(new_canvas_x_br - self.canvas_width), int(
                new_canvas_y_br - self.canvas_height)
        if image_x_br > self.variables.canvas_image_object.full_image_nx:
            image_x_br = self.variables.canvas_image_object.full_image_nx
            new_canvas_x_br, new_canvas_y_br = self.variables.canvas_image_object.full_image_yx_to_canvas_coords(
                (image_y_br, image_x_br))
            new_canvas_x_ul, new_canvas_y_ul = int(new_canvas_x_br - self.canvas_width), int(
                new_canvas_y_br - self.canvas_height)

        canvas_rect = (new_canvas_x_ul, new_canvas_y_ul, new_canvas_x_br, new_canvas_y_br)
        self.zoom_to_selection(canvas_rect, self.variables.animate_pan)
        self.hide_shape(self.variables.zoom_rect_id)

    def config_do_not_scale_image_to_fit(self):
        self.sbarv=tk.Scrollbar(self, orient=tk.VERTICAL)
        self.sbarh=tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)
        self.sbarv.grid(row=0, column=1, stick=tk.N+tk.S)
        self.sbarh.grid(row=1, column=0, sticky=tk.E+tk.W)

    def save_as_png(self,
                    output_fname,           # type: str
                    ):
        # put a sleep in here in case there is a dialog covering the screen before this method is called.
        time.sleep(0.2)
        im = self.save_to_numpy_array()
        im.save(output_fname)

    def save_to_numpy_array(self):
        x_ul = self.canvas.winfo_rootx() + 1
        y_ul = self.canvas.winfo_rooty() + 1
        x_lr = x_ul + self.canvas_width
        y_lr = y_ul + self.canvas_height
        im = ImageGrab.grab()
        im = im.crop((x_ul, y_ul, x_lr, y_lr))
        return im
