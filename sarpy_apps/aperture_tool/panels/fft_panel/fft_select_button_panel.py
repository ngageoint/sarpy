from tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from tkinter_gui_builder.widgets import basic_widgets


class FFTSelectButtonPanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.inv_fft = basic_widgets.Button

        widget_list = ["inv_fft"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("fft select")
