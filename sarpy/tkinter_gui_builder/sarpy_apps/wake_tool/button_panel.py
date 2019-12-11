from sarpy.tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class ButtonPanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.b1 = basic_widgets.Button
        self.b2 = basic_widgets.Button
        self.b3 = basic_widgets.Button
        self.b4 = basic_widgets.Button
        self.b5 = basic_widgets.Button
        self.b6 = basic_widgets.Button
        self.b7 = basic_widgets.Button
        self.b8 = basic_widgets.Button
        self.b9 = basic_widgets.Button
        self.b10 = basic_widgets.Button
        self.b11 = basic_widgets.Button
        self.b12 = basic_widgets.Button
        self.b13 = basic_widgets.Button
        self.b14 = basic_widgets.Button
        self.b15 = basic_widgets.Button
        self.b16 = basic_widgets.Button
        self.b17 = basic_widgets.Button
        self.b18 = basic_widgets.Button
        self.b19 = basic_widgets.Button
        self.b20 = basic_widgets.Button

        buttons_list = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10",
                        "b11", "b12", "b13", "b14", "b15", "b16", "b17", "b18", "b19"]

        self.init_w_box_layout(buttons_list, 5, widget_width=4, widget_height=2)

        self.set_label_text("wake tool buttons")
