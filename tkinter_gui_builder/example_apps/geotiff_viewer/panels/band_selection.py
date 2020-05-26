from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class BandSelection(AbstractWidgetPanel):

    red_selection = basic_widgets.Combobox      # type: basic_widgets.Combobox
    green_selection = basic_widgets.Combobox    # type: basic_widgets.Combobox
    blue_selection = basic_widgets.Combobox     # type: basic_widgets.Combobox
    alpha_selection = basic_widgets.Combobox    # type: basic_widgets.Combobox

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        widget_list = ["red", "red_selection",
                       "green", "green_selection",
                       "blue", "blue_selection",
                       "alpha", "alpha_selection"]
        self.init_w_box_layout(widget_list, n_columns=2, column_widths=10)
