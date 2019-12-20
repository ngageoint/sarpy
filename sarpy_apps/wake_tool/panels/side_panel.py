from tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from sarpy_apps.wake_tool.panels.button_panel import ButtonPanel
from tkinter_gui_builder.panel_templates.file_selector import FileSelector


class SidePanel(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        self.buttons = ButtonPanel          # type: ButtonPanel
        self.file_selector = FileSelector

        widget_list = ["file_selector", "buttons"]

        self.init_w_horizontal_layout(widget_list)

        self.set_label_text("wake tool controls")
