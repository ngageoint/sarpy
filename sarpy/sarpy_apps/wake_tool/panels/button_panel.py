from tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.rect_draw = basic_widgets.Button
        self.line_draw = basic_widgets.Button
        self.point_draw = basic_widgets.Button
        self.calculate_wake = basic_widgets.Button
        self.foreground_color = basic_widgets.Button

        self.buttons_list = ["line_draw", "point_draw", ('calculate_wake', 'calc'), ("foreground_color", "fg color")]
        self.init_w_box_layout(self.buttons_list, 2, widget_width=8, widget_height=2)
        self.set_label_text("wake tool buttons")

    def unpress_all_buttons(self):
        for i, widget_and_name in enumerate(self.buttons_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(relief="raised")

    def press_all_buttons(self):
        for i, widget_and_name in enumerate(self.buttons_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(relief="sunken")

    def activate_all_buttons(self):
        for i, widget_and_name in enumerate(self.buttons_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(state="active")