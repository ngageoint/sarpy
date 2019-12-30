import tkinter
from sarpy_apps.wake_tool.panels.side_panel import SidePanel
from tkinter_gui_builder.panel_templates.image_canvas import ImageCanvas
import tkinter.colorchooser as colorchooser
from sarpy_apps.sarpy_app_helper_utils.sarpy_canvas_image import SarpyCanvasDisplayImage

class AppVariables:
    def __init__(self):
        self.image_fname = "None"       # type: str
        self.current_tool_selection = None      # type: str

        self.arrow_id = None             # type: int
        self.point_id = None            # type: int
        self.horizontal_line_id = None      # type: int

        self.line_color = "red"
        self.line_width = 3
        self.horizontal_line_width = 2
        self.horizontal_line_color = "blue"


class WakeTool:
    def __init__(self, master):
        # set the master frame
        master_frame = tkinter.Frame(master)
        self.app_variables = AppVariables()

        # define panels widget_wrappers in master frame
        self.side_panel = SidePanel(master_frame)
        self.side_panel.set_spacing_between_buttons(0)
        self.image_canvas = ImageCanvas(master_frame)
        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()        # type: SarpyCanvasDisplayImage
        self.image_canvas.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.side_panel.pack(side="bottom")
        self.image_canvas.pack(side="top")
        master_frame.pack()

        # set up event listeners
        self.side_panel.buttons.line_draw.on_left_mouse_click(self.callback_press_line_button)
        self.side_panel.buttons.point_draw.on_left_mouse_click(self.callback_press_point_button)
        self.side_panel.buttons.foreground_color.on_left_mouse_click(self.callback_select_color)
        self.side_panel.buttons.calculate_wake.on_left_mouse_click(self.callback_calculate_wake)

        self.image_canvas.variables.line_width = self.app_variables.line_width
        self.image_canvas.canvas.on_left_mouse_click(self.callback_handle_left_mouse_click)
        self.image_canvas.canvas.on_left_mouse_motion(self.callback_on_left_mouse_motion)

        self.side_panel.file_selector.set_fname_filters(["*.NITF", ".nitf"])
        self.side_panel.file_selector.select_button.on_left_mouse_click(self.callback_select_file)

    def callback_select_file(self, event):
        self.side_panel.file_selector.select_file(event)
        if self.side_panel.file_selector.fname:
            self.app_variables.image_fname = self.side_panel.file_selector.fname
        self.image_canvas.set_canvas_image(self.app_variables.image_fname)

    def callback_press_line_button(self, event):
        self.side_panel.buttons.unpress_all_buttons()
        self.side_panel.buttons.activate_all_buttons()
        self.side_panel.buttons.line_draw.config(state="disabled")
        self.side_panel.buttons.line_draw.config(relief="sunken")
        self.image_canvas.set_current_tool_to_draw_arrow()
        self.image_canvas.variables.current_object_id = self.app_variables.arrow_id

    def callback_press_point_button(self, event):
        self.side_panel.buttons.unpress_all_buttons()
        self.side_panel.buttons.activate_all_buttons()
        self.side_panel.buttons.point_draw.config(state="disabled")
        self.side_panel.buttons.point_draw.config(relief="sunken")
        self.image_canvas.set_current_tool_to_draw_point()
        self.image_canvas.variables.current_object_id = self.app_variables.point_id

    def callback_handle_left_mouse_click(self, event):
        # first do all the normal mouse click functionality of the canvas
        self.image_canvas.callback_handle_left_mouse_click(event)
        # now set the object ID's accordingly
        if self.image_canvas.variables.current_tool == self.image_canvas.constants.DRAW_ARROW_TOOL:
            self.app_variables.arrow_id = self.image_canvas.variables.current_object_id
        if self.image_canvas.variables.current_tool == self.image_canvas.constants.DRAW_POINT_TOOL:
            self.app_variables.point_id = self.image_canvas.variables.current_object_id
        if self.app_variables.point_id is not None and self.app_variables.arrow_id is not None:
            self.draw_horizontal_line()

    def callback_on_left_mouse_motion(self, event):
        self.image_canvas.callback_handle_left_mouse_motion(event)
        self.draw_horizontal_line()

    def draw_horizontal_line(self):
        # draw horizontal line
        if self.app_variables.point_id is not None and self.app_variables.arrow_id is not None:
            point_x, point_y = self.image_canvas.get_point_xy_center(self.app_variables.point_id)

            line_slope, line_intercept = self.get_line_slope_and_intercept()

            end_x = (point_y - line_intercept) / line_slope
            end_y = point_y
            coords = (point_x, point_y, end_x, end_y)
            last_obj_id = self.image_canvas.variables.current_object_id
            if self.app_variables.horizontal_line_id is None:
                self.app_variables.horizontal_line_id = self.image_canvas.create_new_line(coords, fill=self.app_variables.horizontal_line_color, width=self.app_variables.horizontal_line_width)
            else:
                self.image_canvas.modify_existing_shape(self.app_variables.horizontal_line_id, coords)
            self.image_canvas.variables.current_object_id = last_obj_id

    def callback_select_color(self, event):
        self.image_canvas.variables.foreground_color = colorchooser.askcolor()[1]

    def callback_calculate_wake(self, event):
        point_x, point_y = self.image_canvas.get_point_xy_center(self.app_variables.point_id)
        line_slope, line_intercept = self.get_line_slope_and_intercept()

        line_horizontal_x = (point_y - line_intercept)/line_slope

        horizontal_distance = point_x - line_horizontal_x
        print(horizontal_distance)

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
