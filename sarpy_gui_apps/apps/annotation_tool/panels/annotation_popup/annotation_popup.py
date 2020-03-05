from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets
from sarpy.annotation.annotate import LabelSchema
import tkinter


class AnnotationPopup(AbstractWidgetPanel):
    parent_types = basic_widgets.Label      # type: basic_widgets.Label
    thing_type = basic_widgets.Combobox  # type: basic_widgets.Combobox

    def __init__(self,
                 parent,
                 label_schema,          # type: LabelSchema
                 ):
        master_frame = tkinter.Frame(parent)
        AbstractWidgetPanel.__init__(self, master_frame)
        widget_list = ["parent_types", "thing_type"]
        self.init_w_vertical_layout(widget_list)
        self.set_label_text("annotate")

        # set up base types for initial dropdown menu

        master_frame.pack()
        self.pack()

