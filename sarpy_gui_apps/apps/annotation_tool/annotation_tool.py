import tkinter
from sarpy_gui_apps.apps.annotation_tool.panels.context_image_panel.context_image_panel import ContextImagePanel
from sarpy_gui_apps.apps.annotation_tool.panels.annotate_panel.annotate_image_panel import AnnotateImagePanel
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import numpy as np
import os


class AppVariables:
    def __init__(self):
        pass


class AnnotationTool(AbstractWidgetPanel):
    context_panel = ContextImagePanel
    annotate_panel = AnnotateImagePanel

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widgets_list = ["context_panel", "annotate_panel"]
        self.init_w_horizontal_layout(widgets_list)
        master_frame.pack()
        self.pack()

        self.app_variables = AppVariables()
        self.context_panel.image_canvas.canvas.on_left_mouse_release(self.callback_handle_context_left_mouse_release)

    def callback_handle_context_left_mouse_release(self, event):
        self.context_panel.callback_handle_left_mouse_release(event)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = AnnotationTool(root)
    root.mainloop()
