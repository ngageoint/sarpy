from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets
from tkinter_gui_builder.panel_templates.file_selector.file_selector import FileSelector


class ChipSizePanel(AbstractWidgetPanel):
    nx_label = basic_widgets.Label          # type: basic_widgets.Label
    nx = basic_widgets.Entry                # type: basic_widgets.Entry
    ny_label = basic_widgets.Label          # type: basic_widgets.Label
    ny = basic_widgets.Entry                # type: basic_widgets.Entry

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.init_w_box_layout(["nx_label", "nx",
                                "ny_label", "ny"],
                               n_columns=2)
        self.pack()


class LoadImage(AbstractWidgetPanel):
    file_selector = FileSelector            # type: FileSelector
    chip_size_panel = ChipSizePanel         # type: ChipSizePanel

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        widgets_list = ["file_selector", "chip_size_panel"]

        self.init_w_basic_widget_list(widgets_list, n_rows=2, n_widgets_per_row_list=[1, 1])

        self.file_selector.set_fname_filters([("NITF files", ".nitf .NITF .ntf .NTF")])
        self.pack()
