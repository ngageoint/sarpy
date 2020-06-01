import tkinter
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class DirectionPanel(AbstractWidgetPanel):
    slow_time = basic_widgets.RadioButton          # type: basic_widgets.RadioButton
    fast_time = basic_widgets.RadioButton          # type: basic_widgets.RadioButton
    reverse = basic_widgets.CheckButton             # type: basic_widgets.CheckButton

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.init_w_horizontal_layout(["slow_time", "fast_time", "reverse"])
        self.selected_value = tkinter.IntVar()
        self.selected_value.set(1)

        self.slow_time.config(variable=self.selected_value, value=1)
        self.fast_time.config(variable=self.selected_value, value=2)
        self.pack()

    def is_slow_time(self):
        if self.selected_value.get() == 1:
            return True
        else:
            return False


class AnimationSettingsPanel(AbstractWidgetPanel):

    number_of_frames = basic_widgets.Entry
    aperture_fraction = basic_widgets.Entry
    frame_rate = basic_widgets.Entry
    cycle_continuously = basic_widgets.CheckButton
    step_forward = basic_widgets.Button
    step_back = basic_widgets.Button
    play = basic_widgets.Button
    stop = basic_widgets.Button

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        self.init_w_box_layout(["Number of Frames:", "number_of_frames", "", "",
                                "Aperture Fraction:", "aperture_fraction", "", "",
                                "Frame Rate:", "frame_rate", "fps", "",
                                "step_back", "step_forward", "play", "stop",
                                "cycle_continuously",], n_columns=4, column_widths=[20, 10, 3, 3])

        self.number_of_frames.set_text("7")
        self.aperture_fraction.set_text("0.25")
        self.frame_rate.set_text("5")


class AnimationPanel(AbstractWidgetPanel):
    direction = DirectionPanel         # type: DirectionPanel
    animation_settings = AnimationSettingsPanel     # type: AnimationSettingsPanel

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        self.parent = parent
        widgets_list = ["direction", "animation_settings"]

        self.init_w_vertical_layout(widgets_list)
        self.pack()
        self.parent.protocol("WM_DELETE_WINDOW", self.close_window)

