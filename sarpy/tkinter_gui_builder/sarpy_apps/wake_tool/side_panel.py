from sarpy.tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from sarpy.tkinter_gui_builder.sarpy_apps.wake_tool.button_panel import ButtonPanel
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class SidePanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.buttons = ButtonPanel          # type: ButtonPanel
        self.combobox = basic_widgets.Combobox

        widget_list = ["combobox", "buttons"]

        self.init_w_vertical_layout(widget_list)

        self.set_label_text("wake tool controls")
