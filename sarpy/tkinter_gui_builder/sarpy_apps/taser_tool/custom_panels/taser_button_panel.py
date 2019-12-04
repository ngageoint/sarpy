from sarpy.tkinter_gui_builder.panel_templates.basic_button_panel import BasicButtonPanel
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class TaserButtonPanel(BasicButtonPanel):
    def __init__(self, parent):
        BasicButtonPanel.__init__(self, parent)
        self.fname_select = basic_widgets.Button
        self.update_rect_image = basic_widgets.Button
        self.remap_dropdown = basic_widgets.Combobox

        self.init_w_vertical_layout(["fname_select",
                                     ("update_rect_image", "update rect selection"),
                                     ("remap_dropdown", "remap dropdown"),])

        self.remap_dropdown.update_combobox_values(["density",
                                                    "brighter",
                                                    "darker",
                                                    "high contrast",
                                                    "linear",
                                                    "log",
                                                    "pedf",
                                                    "nrl"])
