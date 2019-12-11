from sarpy.tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class TaserButtonPanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
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
        self.set_label_text("taser buttons")
