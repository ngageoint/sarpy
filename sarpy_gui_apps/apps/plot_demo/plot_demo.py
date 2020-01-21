import tkinter
from tkinter_gui_builder.panel_templates.pyplot_panel.main_panel import PyplotPanel
from sarpy_gui_apps.apps.plot_demo.panels.plot_demo_button_panel import ButtonPanel
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel


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


class PlotDemo(AbstractWidgetPanel):
    button_panel = ButtonPanel          # type: ButtonPanel
    pyplot_panel = PyplotPanel      # type: PyplotPanel

    def __init__(self, master):
        # set the master frame
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)
        widget_list = ["pyplot_panel", "button_panel"]
        self.init_w_vertical_layout(widget_list)
        self.variables = AppVariables()

        # need to pack both master frame and self, since this is the main app window.
        master_frame.pack()
        self.pack()

        # set up event listeners
        self.button_panel.single_sine.on_left_mouse_click(self.callback_single_sin)
        self.button_panel.multi_sine.on_left_mouse_click(self.callback_muli_sine)
        self.button_panel.animated_sine.on_left_mouse_click(self.callback_animated_sine)

    def callback_single_sin(self, event):
        stuff = self.winfo_exists()
        print("single sine")

    def callback_muli_sine(self, event):
        stuff = self.winfo_exists()
        print("multi sine")

    def callback_animated_sine(self, event):
        stuff = self.winfo_exists()
        print("animated sine")


if __name__ == '__main__':
    root = tkinter.Tk()
    app = PlotDemo(root)
    root.mainloop()
