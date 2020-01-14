from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class FFTSelectButtonPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.inv_fft = basic_widgets.Button

        widget_list = ["inv_fft"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("fft select")
