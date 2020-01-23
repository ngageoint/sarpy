from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(AbstractWidgetPanel):
    single_plot = basic_widgets.Button
    multi_plot = basic_widgets.Button
    animated_plot = basic_widgets.Button
    color_button = basic_widgets.Button

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        widget_list = ["single_plot", "multi_plot", "animated_plot", "color_button"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("plot demo buttons")
