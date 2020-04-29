from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class Toolbar(AbstractWidgetPanel):
    zoom_in = basic_widgets.Button
    zoom_out = basic_widgets.Button
    pan = basic_widgets.Button
    select_aoi = basic_widgets.Button
    submit_aoi = basic_widgets.Button

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.init_w_horizontal_layout(["zoom_in", "zoom_out", "pan", "select_aoi", "submit_aoi"])
