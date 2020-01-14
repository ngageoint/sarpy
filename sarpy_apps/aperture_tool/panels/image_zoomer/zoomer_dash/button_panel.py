from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.rect_draw = basic_widgets.Button
        self.zoom_in = basic_widgets.Button
        self.zoom_out = basic_widgets.Button

        widget_list = ["rect_draw", "zoom_in", "zoom_out"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("wake tool buttons")
