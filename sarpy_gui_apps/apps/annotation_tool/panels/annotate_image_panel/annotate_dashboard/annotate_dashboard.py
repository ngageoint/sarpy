from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class ButtonPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.pan = basic_widgets.Button                         # type: basic_widgets.Button
        self.draw_polygon = basic_widgets.Button                # type: basic_widgets.Button
        self.select_existing_shape = basic_widgets.Combobox     # type: basic_widgets.Combobox
        self.select_closest_shape = basic_widgets.Button        # type: basic_widgets.Button
        self.popup = basic_widgets.Button                       # type: basic_widgets.Button
        self.save_annotations = basic_widgets.Button            # type: basic_widgets.Button

        widget_list = ["pan", "draw_polygon", "select_existing_shape", "select_closest_shape", "popup", "save_annotations"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("annotate controls")


class AnnotateInfoPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.decimation_label = basic_widgets.Label
        self.annotate_decimation_val = basic_widgets.Entry

        widget_list = ["decimation_label", "annotate_decimation_val"]

        self.init_w_box_layout(widget_list, n_columns=2, column_widths=[20, 10])

        self.decimation_label.config(text="decimation")
        self.annotate_decimation_val.config(state='disabled')


class AnnotateDash(AbstractWidgetPanel):
    controls = ButtonPanel
    info_panel = AnnotateInfoPanel

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        widget_list = ["controls", "info_panel"]
        self.init_w_basic_widget_list(widget_list, 2, [1, 2])
