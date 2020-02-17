from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class OrthoButtonPanel(AbstractWidgetPanel):
    fname_select = basic_widgets.Button
    pan = basic_widgets.Button
    display_ortho = basic_widgets.Button

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        self.init_w_vertical_layout(["fname_select",
                                     "pan",
                                     "display_ortho"])

        self.set_label_text("ortho buttons")
