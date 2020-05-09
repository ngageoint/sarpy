from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class PhaseHistoryPanel(AbstractWidgetPanel):

    start_percent_cross = basic_widgets.Entry
    stop_percent_cross = basic_widgets.Entry
    fraction_cross = basic_widgets.Entry
    resolution_cross = basic_widgets.Entry
    sample_spacing_cross = basic_widgets.Entry
    ground_resolution_cross = basic_widgets.Entry

    start_percent_range = basic_widgets.Entry
    stop_percent_range = basic_widgets.Entry
    fraction_range = basic_widgets.Entry
    resolution_range = basic_widgets.Entry
    sample_spacing_range = basic_widgets.Entry
    ground_resolution_range = basic_widgets.Entry

    resolution_cross_units = basic_widgets.Label
    sample_spacing_cross_units = basic_widgets.Label
    ground_resolution_cross_units = basic_widgets.Label

    resolution_range_units = basic_widgets.Label
    sample_spacing_range_units = basic_widgets.Label
    ground_resolution_range_units = basic_widgets.Label


    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.config(borderwidth=2)

        widget_list = ["", "\033[1mCross-Range\033[0m", "", "Range", "",
                       "Start %", "start_percent_cross", "", "start_percent_range", "",
                       "Stop %", "stop_percent_cross", "", "stop_percent_range", "",
                       "Fraction", "fraction_cross", "", "fraction_range", "",
                       "Resolution", "resolution_cross", "resolution_cross_units", "resolution_range", "resolution_range_units",
                       "Sample Spacing", "sample_spacing_cross", "sample_spacing_cross_units", "sample_spacing_range", "sample_spacing_range_units",
                       "Ground Resolution", "ground_resolution_cross", "ground_resolution_cross_units", "ground_resolution_range", "ground_resolution_range_units"]
        self.init_w_box_layout(widget_list, 5, column_widths=20)

        self.resolution_cross_units.set_text("Units")
        self.sample_spacing_cross_units.set_text("Units")
        self.ground_resolution_cross_units.set_text("Units")

        self.resolution_range_units.set_text("Units")
        self.sample_spacing_range_units.set_text("Units")
        self.ground_resolution_range_units.set_text("Units")

        
