from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.panel_templates.file_selector.file_selector import FileSelector
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.zoom_in = basic_widgets.Button
        self.zoom_out = basic_widgets.Button
        self.pan = basic_widgets.Button
        self.select = basic_widgets.Button

        widget_list = ["zoom_in", "zoom_out", "pan", "select"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("controls")


class ContextInfoPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.decimation_label = basic_widgets.Label
        self.decimation_val = basic_widgets.Entry

        widget_list = ["decimation_label", "decimation_val"]

        self.init_w_box_layout(widget_list, n_columns=2, column_widths=[20, 10])
        self.set_label_text("info panel")

        self.decimation_label.config(text="decimation")
        self.decimation_val.config(state='disabled')


class ContextMasterDash(AbstractWidgetPanel):
    buttons = ButtonPanel
    file_selector = FileSelector
    info_panel = ContextInfoPanel

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        widget_list = ["file_selector", "buttons", "info_panel"]
        self.init_w_basic_widget_list(widget_list, 2, [1, 2])
        self.set_label_text("wake tool controls")

