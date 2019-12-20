import tkinter
from sarpy_apps.wake_tool.panels.side_panel import SidePanel
from tkinter_gui_builder.panel_templates.image_canvas import ImageCanvas
import tkinter.colorchooser as colorchooser

RECT_DRAW_TOOL = "rect_draw_tool"
LINE_DRAW_TOOL = "line_draw_tool"
POINT_DRAW_TOOL = "point_draw_tool"


class AppVariables:
    def __init__(self):
        self.image_fname = "None"       # type: str
        self.current_tool_selection = None      # type: str

        self.first_rect_drawn = False           # type: bool
        self.first_line_drawn = False           # type: bool

        self.rect_id = None             # type: int
        self.line_id = None             # type: int
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
        self.image_canvas.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.side_panel.pack(side="bottom")
        self.image_canvas.pack(side="top")
        master_frame.pack()

        # set up event listeners
        # self.side_panel.buttons.rect_draw.on_left_mouse_click(self.callback_press_rect_button)
        self.side_panel.buttons.line_draw.on_left_mouse_click(self.callback_press_line_button)
        self.side_panel.buttons.point_draw.on_left_mouse_click(self.callback_press_point_button)
        self.side_panel.buttons.foreground_color.on_left_mouse_click(self.callback_select_color)
        self.side_panel.buttons.calculate_wake.on_left_mouse_click(self.callback_calculate_wake)

        self.image_canvas.canvas.on_left_mouse_press(self.callback_handle_canvas_mouse_press_event)
        self.image_canvas.canvas.on_left_mouse_motion(self.callback_handle_canvas_mouse_motion_event)

        self.image_canvas.variables.line_width = self.app_variables.line_width

    def callback_press_rect_button(self, event):
        self.side_panel.buttons.unpress_all_buttons()
        self.side_panel.buttons.activate_all_buttons()
        self.side_panel.buttons.rect_draw.config(state="disabled")
        self.side_panel.buttons.rect_draw.config(relief="sunken")
        self.app_variables.current_tool_selection = RECT_DRAW_TOOL

    def callback_press_line_button(self, event):
        self.side_panel.buttons.unpress_all_buttons()
        self.side_panel.buttons.activate_all_buttons()
        self.side_panel.buttons.line_draw.config(state="disabled")
        self.side_panel.buttons.line_draw.config(relief="sunken")
        self.app_variables.current_tool_selection = LINE_DRAW_TOOL

    def callback_press_point_button(self, event):
        self.side_panel.buttons.unpress_all_buttons()
        self.side_panel.buttons.activate_all_buttons()
        self.side_panel.buttons.point_draw.config(state="disabled")
        self.side_panel.buttons.point_draw.config(relief="sunken")
        self.app_variables.current_tool_selection = POINT_DRAW_TOOL

    def create_rect(self, event):
        self.image_canvas.variables.foreground_color = "red"
        if self.app_variables.rect_id is None:
            self.image_canvas.variables.current_object_id = None
            self.image_canvas.event_initiate_rect(event)
            self.app_variables.rect_id = self.image_canvas.variables.current_object_id
        else:
            self.image_canvas.variables.current_object_id = self.app_variables.rect_id
            self.image_canvas.event_initiate_rect(event)

    def create_line(self, event):
        self.image_canvas.variables.foreground_color = self.app_variables.line_color
        if self.app_variables.line_id is None:
            self.image_canvas.variables.current_object_id = None
            self.image_canvas.event_initiate_line(event)
            self.app_variables.line_id = self.image_canvas.variables.current_object_id
        else:
            self.image_canvas.variables.current_object_id = self.app_variables.line_id
            self.image_canvas.event_initiate_line(event)
            print(self.image_canvas.canvas.coords(self.app_variables.line_id))

    def create_point(self, event):
        if self.app_variables.line_id is None:
            print("draw a line first")
        else:
            if self.app_variables.point_id is None:
                self.image_canvas.variables.current_object_id = None
                self.image_canvas.event_draw_point(event)
                self.app_variables.point_id = self.image_canvas.variables.current_object_id
            else:
                self.image_canvas.variables.current_object_id = self.app_variables.point_id
                self.image_canvas.event_draw_point(event)
            self.draw_horizontal_line()
            print(self.get_point_xy())

    def draw_horizontal_line(self):
        # draw horizontal line
        point_x, point_y = self.get_point_xy()

        line_slope, line_intercept = self.get_line_slope_and_intercept()

        end_x = (point_y - line_intercept) / line_slope
        end_y = point_y
        if self.app_variables.horizontal_line_id is None:
            self.image_canvas.variables.current_object_id = None
            horizontal_line_id = self.image_canvas.canvas.create_line(point_x, point_y, end_x, end_y,
                                                                      fill=self.app_variables.horizontal_line_color,
                                                                      width=self.app_variables.horizontal_line_width)
            self.app_variables.horizontal_line_id = horizontal_line_id
            self.image_canvas.variables.object_ids.append(horizontal_line_id)
        else:
            self.image_canvas.canvas.coords(self.app_variables.horizontal_line_id, point_x, point_y, end_x, end_y)

    def callback_handle_canvas_mouse_press_event(self, event):
        current_fg_color = self.image_canvas.variables.foreground_color
        if self.app_variables.current_tool_selection is RECT_DRAW_TOOL:
            self.create_rect(event)

        if self.app_variables.current_tool_selection is LINE_DRAW_TOOL:
            self.create_line(event)

        if self.app_variables.current_tool_selection is POINT_DRAW_TOOL:
            self.create_point(event)

        # change the fg color back to what it was
        self.image_canvas.variables.foreground_color = current_fg_color

    def callback_handle_canvas_mouse_motion_event(self, event):
        if self.app_variables.current_tool_selection is RECT_DRAW_TOOL:
            self.image_canvas.event_drag_rect(event)
        if self.app_variables.current_tool_selection is LINE_DRAW_TOOL:
            self.image_canvas.event_drag_line(event)

    def callback_select_color(self, event):
        self.image_canvas.variables.foreground_color = colorchooser.askcolor()[1]

    def get_point_xy(self):
        point_coords = self.image_canvas.canvas.coords(self.app_variables.point_id)
        point_x = (point_coords[0] + point_coords[2]) / 2.0
        point_y = (point_coords[1] + point_coords[3]) / 2.0
        return point_x, point_y

    def callback_calculate_wake(self, event):
        point_x, point_y = self.get_point_xy()
        line_slope, line_intercept = self.get_line_slope_and_intercept()

        line_horizontal_x = (point_y - line_intercept)/line_slope

        horizontal_distance = point_x - line_horizontal_x
        print(horizontal_distance)

    def get_line_slope_and_intercept(self):
        line_coords = self.image_canvas.canvas.coords(self.app_variables.line_id)

        line_x1, line_x2 = line_coords[0], line_coords[2]
        line_y1, line_y2 = line_coords[1], line_coords[3]

        line_slope = (line_y2 - line_y1)/(line_x2 - line_x1)
        line_intercept = line_y1 - line_slope * line_x1
        return line_slope, line_intercept


if __name__ == '__main__':
    root = tkinter.Tk()
    app = WakeTool(root)
    root.mainloop()
