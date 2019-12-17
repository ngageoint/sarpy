import tkinter
from sarpy.tkinter_gui_builder.sarpy_apps.wake_tool.side_panel import SidePanel
from sarpy.tkinter_gui_builder.sarpy_apps.wake_tool.image_canvas import ImageCanvas
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
        self.side_panel.pack(side="left")
        self.image_canvas.pack(side="left")
        master_frame.pack()

        # set up event listeners
        self.side_panel.buttons.rect_draw.on_left_mouse_click(self.callback_press_rect_button)
        self.side_panel.buttons.line_draw.on_left_mouse_click(self.callback_press_line_button)
        self.side_panel.buttons.point_draw.on_left_mouse_click(self.callback_press_point_button)
        self.side_panel.buttons.foreground_color.on_left_mouse_click(self.callback_select_color)

        self.image_canvas.canvas.on_left_mouse_press(self.callback_handle_canvas_mouse_press_event)
        self.image_canvas.canvas.on_left_mouse_motion(self.callback_handle_canvas_mouse_motion_event)

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

    def callback_handle_canvas_mouse_press_event(self, event):
        if self.app_variables.current_tool_selection is RECT_DRAW_TOOL:
            if self.app_variables.rect_id is None:
                self.image_canvas.variables.current_object_id = None
                self.image_canvas.event_initiate_rect(event)
                self.app_variables.rect_id = self.image_canvas.variables.current_object_id
            else:
                self.image_canvas.variables.current_object_id = self.app_variables.rect_id

        if self.app_variables.current_tool_selection is LINE_DRAW_TOOL:
            if self.app_variables.line_id is None:
                self.image_canvas.variables.current_object_id = None
                self.image_canvas.event_initiate_line(event)
                self.app_variables.line_id = self.image_canvas.variables.current_object_id
            else:
                self.image_canvas.variables.current_object_id = self.app_variables.line_id
            self.image_canvas.event_initiate_line(event)

        if self.app_variables.current_tool_selection is POINT_DRAW_TOOL:
            self.image_canvas.event_draw_point(event)

    def callback_handle_canvas_mouse_motion_event(self, event):
        if self.app_variables.current_tool_selection is RECT_DRAW_TOOL:
            self.image_canvas.event_drag_rect(event)
        if self.app_variables.current_tool_selection is LINE_DRAW_TOOL:
            self.image_canvas.event_drag_line(event)

    def callback_select_color(self, event):
        self.image_canvas.variables.foreground_color = colorchooser.askcolor()[1]


if __name__ == '__main__':
    root = tkinter.Tk()
    app = WakeTool(root)
    root.mainloop()
