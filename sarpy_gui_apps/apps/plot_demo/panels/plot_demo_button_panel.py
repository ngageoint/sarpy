from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(AbstractWidgetPanel):
    single_sine = basic_widgets.Button
    multi_sine = basic_widgets.Button
    animated_sine = basic_widgets.Button

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        widget_list = ["single_sine", "multi_sine", "animated_sine"]
        self.init_w_box_layout(widget_list, 3, column_widths=8)
        self.set_label_text("plot demo buttons")
