from sarpy.tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from sarpy.tkinter_gui_builder.sarpy_apps.wake_tool.button_panel import ButtonPanel
from sarpy.tkinter_gui_builder.panel_templates.file_selector import FileSelector
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class SidePanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.buttons = ButtonPanel          # type: ButtonPanel
        self.file_selector = FileSelector

        widget_list = ["file_selector", "buttons"]

        self.init_w_horizontal_layout(widget_list)

        self.set_label_text("wake tool controls")
