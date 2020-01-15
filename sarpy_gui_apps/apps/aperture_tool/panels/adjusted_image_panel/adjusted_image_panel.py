from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import tkinter as tk


class AdjustedViewPanel(tk.LabelFrame):
    image_canvas = ImageCanvas                      # type: ImageCanvas

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        self.image_canvas = ImageCanvas(parent)
        self.image_canvas.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.image_canvas.pack(side="right")

        # set up event listeners
        self.image_canvas.set_labelframe_text("adjusted view")
