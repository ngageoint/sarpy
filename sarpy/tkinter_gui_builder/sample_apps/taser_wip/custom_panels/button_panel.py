from sarpy.tkinter_gui_builder.panel_templates.basic_button_panel import BasicButtonPanel
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class ButtonPanel(BasicButtonPanel):
    def __init__(self, parent):
        BasicButtonPanel.__init__(self, parent)
        self.fname_select = basic_widgets.Button
        self.update_basic_image = basic_widgets.Button
        self.update_rect_image = basic_widgets.Button

        self.init_w_vertical_layout(["fname_select",
                                     ("update_basic_image", "update image"),
                                     ("update_rect_image", "update rect selection")])
