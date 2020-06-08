import tkinter
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class DirectionPanel(AbstractWidgetPanel):
    slow_time = basic_widgets.RadioButton          # type: basic_widgets.RadioButton
    fast_time = basic_widgets.RadioButton          # type: basic_widgets.RadioButton
    resolution_mode = basic_widgets.RadioButton     # type: basic_widgets.RadioButton
    full_range_bandwidth = basic_widgets.RadioButton    # type: basic_widgets.RadioButton
    full_az_bandwidth = basic_widgets.RadioButton       # type: basic_widgets.RadioButton
    reverse = basic_widgets.CheckButton             # type: basic_widgets.CheckButton

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.init_w_horizontal_layout(["slow_time", "fast_time", "resolution_mode", "full_range_bandwidth", "full_az_bandwidth", "reverse"])
        self.selected_value = tkinter.IntVar()
        self.selected_value.set(1)

        self.slow_time.config(variable=self.selected_value, value=1)
        self.fast_time.config(variable=self.selected_value, value=2)
        self.resolution_mode.config(variable=self.selected_value, value=3)
        self.full_range_bandwidth.config(variable=self.selected_value, value=4)
        self.full_az_bandwidth.config(variable=self.selected_value, value=5)
        self.pack()

    def is_slow_time(self):
        if self.selected_value.get() == 1:
            return True
        else:
            return False

    def is_fast_time(self):
        if self.selected_value.get() == 2:
            return True
        else:
            return False

    def is_resolution_mode(self):
        if self.selected_value.get() == 3:
            return True
        else:
            return False

    def is_full_range_bandwidth(self):
        if self.selected_value.get() == 4:
            return True
        else:
            return False

    def is_full_az_bandwidth(self):
        if self.selected_value.get() == 5:
            return True
        else:
            return False


class FastSlowSettingsPanel(AbstractWidgetPanel):
    aperture_fraction = basic_widgets.Entry

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        self.init_w_box_layout(["Aperture Fraction:", "aperture_fraction"], n_columns=2, column_widths=[20, 10])
        self.aperture_fraction.set_text("0.25")


class ResolutionSettingsPanel(AbstractWidgetPanel):
    min_res = basic_widgets.Entry
    max_res = basic_widgets.Entry
    step_size = basic_widgets.Entry

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        self.init_w_box_layout(["Min Res", "min_res",
                                "Max Res", "max_res",
                                "Step Size", "step_size"],
                               n_columns=2, column_widths=[20, 10])

        self.min_res.set_text("10")
        self.max_res.set_text("100")


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
                                "Frame Rate:", "frame_rate", "fps", "",
                                "step_back", "step_forward", "play", "stop",
                                "cycle_continuously",], n_columns=4, column_widths=[20, 10, 3, 3])

        self.number_of_frames.set_text("7")
        self.frame_rate.set_text("5")


class AnimationPanel(AbstractWidgetPanel):
    direction = DirectionPanel         # type: DirectionPanel
    animation_settings = AnimationSettingsPanel     # type: AnimationSettingsPanel
    fast_slow_settings = FastSlowSettingsPanel      # type: FastSlowSettingsPanel
    resolution_settings = ResolutionSettingsPanel       # type: ResolutionSettingsPanel

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        self.parent = parent
        widgets_list = ["direction", "animation_settings", "fast_slow_settings", "resolution_settings"]

        self.init_w_vertical_layout(widgets_list)
        self.pack()
        self.parent.protocol("WM_DELETE_WINDOW", self.close_window)

