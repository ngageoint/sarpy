import tkinter
from tkinter_gui_builder.panel_templates.pyplot_panel.pyplot_canvas import PyplotCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str
        self.remap_type = "density"
        self.selection_rect_id = None           # type: int


class PyplotPanel(AbstractWidgetPanel):
    canvas = PyplotCanvas

    def __init__(self, master):
        AbstractWidgetPanel.__init__(self, master)
        self.variables = AppVariables()

        widget_list = ["canvas"]
        self.init_w_horizontal_layout(widget_list)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = PyplotPanel(root)
    root.mainloop()
