from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets
from tkinter_gui_builder.panel_templates.file_selector.file_selector import FileSelector


class DirectionPanel(AbstractWidgetPanel):
    slow_time = basic_widgets.Label          # type: basic_widgets.Label
    fast_time = basic_widgets.Label          # type: basic_widgets.Label

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.init_w_horizontal_layout(["slow_time", "fast_time"])
        self.pack()


class Animation(AbstractWidgetPanel):
    direction = DirectionPanel         # type: DirectionPanel

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        widgets_list = ["direction"]

        self.init_w_basic_widget_list(widgets_list, n_rows=1, n_widgets_per_row_list=[1])
        self.pack()
