from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        self.thing_type = basic_widgets.Combobox  # type: basic_widgets.Combobox

        widget_list = ["thing_type"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("annotate")

        self.thing_type.
