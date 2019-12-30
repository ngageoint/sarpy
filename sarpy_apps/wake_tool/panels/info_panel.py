from tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from tkinter_gui_builder.widgets import basic_widgets


class InfoPanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.canvas_distance_label = basic_widgets.Label
        self.canvas_distance_val = basic_widgets.Label
        self.pixel_distance_label = basic_widgets.Label
        self.pixel_distance_entry = basic_widgets.Label

        widget_list = ["canvas_distance_label", "canvas_distance_val",
                       "pixel_distance_label", "pixel_distance_entry"]
        self.init_w_grid_layout(widget_list, n_rows=4, n_widgets_per_row_list=1)
        self.set_label_text("info panel")

        # self.canvas_distance_label.config(text="canvas distance")

