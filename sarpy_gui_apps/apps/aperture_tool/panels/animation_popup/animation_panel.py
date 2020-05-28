import tkinter
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class DirectionPanel(AbstractWidgetPanel):
    slow_time = basic_widgets.RadioButton          # type: basic_widgets.RadioButton
    fast_time = basic_widgets.RadioButton          # type: basic_widgets.RadioButton

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.init_w_horizontal_layout(["slow_time", "fast_time"])
        self.selected_value = tkinter.IntVar()
        self.selected_value.set(1)

        self.slow_time.config(variable=self.selected_value, value=1)
        self.fast_time.config(variable=self.selected_value, value=2)
        self.pack()


class AnimationSettingsPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)


class Animation(AbstractWidgetPanel):
    direction = DirectionPanel         # type: DirectionPanel

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        widgets_list = ["direction"]

        self.init_w_basic_widget_list(widgets_list, n_rows=1, n_widgets_per_row_list=[1])
        self.pack()
