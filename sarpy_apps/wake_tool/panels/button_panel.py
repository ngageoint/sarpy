from tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.rect_draw = basic_widgets.Button
        self.line_draw = basic_widgets.Button
        self.point_draw = basic_widgets.Button
        self.zoom_in = basic_widgets.Button
        self.zoom_out = basic_widgets.Button
        self.foreground_color = basic_widgets.Button

        widget_list = ["line_draw", "point_draw", "foreground_color", "zoom_in", "zoom_out"]
        self.init_w_box_layout(widget_list, 3, column_widths=8, row_heights=2)
        self.set_label_text("wake tool buttons")
