from sarpy.tkinter_gui_builder.panel_templates.basic_button_panel import BasicButtonPanel
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class ButtonPanel(BasicButtonPanel):
    def __init__(self, parent):
        BasicButtonPanel.__init__(self, parent)
        self.fname_select = basic_widgets.Button
        self.update_basic_image = basic_widgets.Button
        self.update_rect_image = basic_widgets.Button

        self.remap_density = basic_widgets.Button
        self.remap_brighter = basic_widgets.Button
        self.remap_darker = basic_widgets.Button
        self.remap_highcontrast = basic_widgets.Button
        self.remap_linear = basic_widgets.Button
        self.remap_log = basic_widgets.Button
        self.remap_pedf = basic_widgets.Button
        self.remap_nrl = basic_widgets.Button

        self.init_w_vertical_layout(["fname_select",
                                     ("update_basic_image", "update image"),
                                     ("update_rect_image", "update rect selection"),
                                     ("remap_density", "remap density"),
                                     ("remap_brighter", "remap brighter"),
                                     ("remap_darker", "remap darker"),
                                     ("remap_highcontrast", "remap high contrast"),
                                     ("remap_linear", "remap linear"),
                                     ("remap_log", "remap log"),
                                     ("remap_pedf", "remap pedf"),
                                     ("remap_nrl", "remap nrl"),])
