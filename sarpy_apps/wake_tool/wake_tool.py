import tkinter
from sarpy_apps.wake_tool.panels.side_panel import SidePanel
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
import tkinter.colorchooser as colorchooser
from sarpy_apps.sarpy_app_helper_utils.sarpy_canvas_image import SarpyCanvasDisplayImage
import sarpy.geometry.point_projection as point_projection
import sarpy.geometry.geocoords as geocoords
import numpy as np
import math


class AppVariables:
    def __init__(self):
        self.image_fname = "None"       # type: str
        self.sicd_metadata = None
        self.current_tool_selection = None      # type: str

        self.arrow_id = None             # type: int
        self.point_id = None            # type: int
        self.horizontal_line_id = None      # type: int

        self.line_color = "red"
        self.line_width = 3
        self.horizontal_line_width = 2
        self.horizontal_line_color = "green"


class WakeTool:
    def __init__(self, master):
        # set the master frame
        master_frame = tkinter.Frame(master)
        self.app_variables = AppVariables()

        # define panels widget_wrappers in master frame
        self.side_panel = SidePanel(master_frame)
        self.side_panel.set_spacing_between_buttons(0)
        self.image_canvas = ImageCanvas(master_frame)
        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()     # type: SarpyCanvasDisplayImage
        self.image_canvas.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.side_panel.pack(side="bottom")
        self.image_canvas.pack(side="top")
        master_frame.pack()

        # set up event listeners
        self.side_panel.buttons.line_draw.on_left_mouse_click(self.callback_press_line_button)
        self.side_panel.buttons.point_draw.on_left_mouse_click(self.callback_press_point_button)
        self.side_panel.buttons.foreground_color.on_left_mouse_click(self.callback_select_color)
        self.side_panel.buttons.zoom_in.on_left_mouse_click(self.callback_set_to_zoom_in)
        self.side_panel.buttons.zoom_out.on_left_mouse_click(self.callback_set_to_zoom_out)

        self.image_canvas.variables.line_width = self.app_variables.line_width
        self.image_canvas.canvas.on_left_mouse_click(self.callback_handle_left_mouse_click)
        self.image_canvas.canvas.on_left_mouse_motion(self.callback_on_left_mouse_motion)

        self.side_panel.file_selector.set_fname_filters(["*.NITF", ".nitf"])
        self.side_panel.file_selector.select_file.on_left_mouse_click(self.callback_select_file)

    def callback_select_file(self, event):
        self.side_panel.file_selector.event_select_file(event)
        if self.side_panel.file_selector.fname:
            self.app_variables.image_fname = self.side_panel.file_selector.fname
        self.image_canvas.init_with_fname(self.app_variables.image_fname)

    def callback_press_line_button(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.line_draw)
        self.image_canvas.set_current_tool_to_draw_arrow()
        self.image_canvas.variables.current_shape_id = self.app_variables.arrow_id

    def callback_press_point_button(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.point_draw)
        self.image_canvas.set_current_tool_to_draw_point()
        self.image_canvas.variables.current_shape_id = self.app_variables.point_id

    def callback_set_to_zoom_in(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.zoom_in)
        self.image_canvas.set_current_tool_to_zoom_in()

    def callback_set_to_zoom_out(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.zoom_out)
        self.image_canvas.set_current_tool_to_zoom_out()

    def callback_select_color(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.foreground_color)
        color = colorchooser.askcolor()[1]
        self.image_canvas.change_shape_color(self.image_canvas.variables.current_shape_id, color)

    def callback_handle_left_mouse_click(self, event):
        # first do all the normal mouse click functionality of the canvas
        self.image_canvas.callback_handle_left_mouse_click(event)
        # now set the object ID's accordingly, we do this so we don't draw multiple arrows or points
        if self.image_canvas.variables.current_tool == self.image_canvas.TOOLS.DRAW_ARROW_TOOL:
            self.app_variables.arrow_id = self.image_canvas.variables.current_shape_id
        if self.image_canvas.variables.current_tool == self.image_canvas.TOOLS.DRAW_POINT_TOOL:
            self.app_variables.point_id = self.image_canvas.variables.current_shape_id
        if self.app_variables.point_id is not None and self.app_variables.arrow_id is not None:
            self.update_distance()

    def callback_on_left_mouse_motion(self, event):
        self.image_canvas.callback_handle_left_mouse_motion(event)
        self.update_distance()

    def update_distance(self):
        if self.app_variables.point_id is not None and self.app_variables.arrow_id is not None:
            # calculate horizontal line segment
            point_x, point_y = self.image_canvas.get_shape_canvas_coords(self.app_variables.point_id)
            line_slope, line_intercept = self.get_line_slope_and_intercept()
            end_x = (point_y - line_intercept) / line_slope
            end_y = point_y
            horizontal_line_coords = (point_x, point_y, end_x, end_y)
            # save last object selected on canvas (either the point or the line)
            last_shape_id = self.image_canvas.variables.current_shape_id
            if self.app_variables.horizontal_line_id is None:
                self.app_variables.horizontal_line_id = \
                    self.image_canvas.create_new_line(horizontal_line_coords,
                                                      fill=self.app_variables.horizontal_line_color,
                                                      width=self.app_variables.horizontal_line_width)
            else:
                self.image_canvas.modify_existing_shape_using_canvas_coords(self.app_variables.horizontal_line_id, horizontal_line_coords)
            # set current object ID back to what it was after drawing the horizontal line
            self.image_canvas.variables.current_shape_id = last_shape_id
            canvas_distance = self.image_canvas.get_canvas_line_length(self.app_variables.horizontal_line_id)
            pixel_distance = self.image_canvas.get_image_line_length(self.app_variables.horizontal_line_id)
            geo_distance = self.calculate_wake_distance()
            self.side_panel.info_panel.canvas_distance_val.set_text("{:.2f}".format(canvas_distance))
            self.side_panel.info_panel.pixel_distance_val.set_text("{:.2f}".format(pixel_distance))
            self.side_panel.info_panel.geo_distance_val.set_text("{:.2f}".format(geo_distance))

    def calculate_wake_distance(self):
        horizontal_line_image_coords = self.image_canvas.canvas_shape_coords_to_image_coords(self.app_variables.horizontal_line_id)
        sicd_meta = self.image_canvas.variables.canvas_image_object.reader_object.sicdmeta
        points = np.asarray(np.reshape(horizontal_line_image_coords, (2,2)))
        ecf_ground_points = point_projection.image_to_ground(points, sicd_meta)
        geo_ground_point_1 = geocoords.ecf_to_geodetic(ecf_ground_points[0, 0], ecf_ground_points[0, 1], ecf_ground_points[0, 2])
        geo_ground_point_2 = geocoords.ecf_to_geodetic(ecf_ground_points[1, 0], ecf_ground_points[1, 1], ecf_ground_points[1, 2])
        distance = math.sqrt( (ecf_ground_points[0, 0] - ecf_ground_points[1, 0])**2 +
                              (ecf_ground_points[0, 1] - ecf_ground_points[1, 1])**2 +
                              (ecf_ground_points[0, 2] - ecf_ground_points[1, 2])**2)
        return distance

    def get_line_slope_and_intercept(self):
        line_coords = self.image_canvas.canvas.coords(self.app_variables.arrow_id)

        line_x1, line_x2 = line_coords[0], line_coords[2]
        line_y1, line_y2 = line_coords[1], line_coords[3]

        line_slope = (line_y2 - line_y1)/(line_x2 - line_x1)
        line_intercept = line_y1 - line_slope * line_x1
        return line_slope, line_intercept


if __name__ == '__main__':
    root = tkinter.Tk()
    app = WakeTool(root)
    root.mainloop()
