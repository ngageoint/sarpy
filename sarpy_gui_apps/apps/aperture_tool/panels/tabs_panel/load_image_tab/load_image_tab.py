from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets
from tkinter_gui_builder.panel_templates.file_selector.file_selector import FileSelector
import tkinter


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
        self.nx.config(state="disabled")
        self.ny.config(state="disabled")

        self.nx_label.set_text("nx: ")
        self.ny_label.set_text("ny: ")
        self.pack()


# TODO: remote this is we are sure we will never want it.
# class SelectionFilter(AbstractWidgetPanel):
#     none = basic_widgets.RadioButton
#     filter_gaussian = basic_widgets.RadioButton
#     one_over_x_to_the_fourth = basic_widgets.RadioButton
#     filter_hamming = basic_widgets.RadioButton
#     cosine_on_ped = basic_widgets.RadioButton
#
#     def __init__(self, parent):
#         AbstractWidgetPanel.__init__(self, parent)
#         self.selected_value = tkinter.IntVar()
#
#         self.set_radio_buttons()
#
#     def set_radio_buttons(self):
#         self.none = basic_widgets.RadioButton(self, text="None", variable=self.selected_value, value=1)
#         self.filter_gaussian = basic_widgets.RadioButton(self, text="FilterGaussian", variable=self.selected_value, value=2)
#         self.one_over_x_to_the_fourth = basic_widgets.RadioButton(self, text="1/x^4", variable=self.selected_value, value=3)
#         self.filter_hamming = basic_widgets.RadioButton(self, text="filterHamming", variable=self.selected_value, value=4)
#         self.cosine_on_ped = basic_widgets.RadioButton(self, text="Cosine on Ped", variable=self.selected_value, value=5)
#
#         self.none.pack()
#         self.filter_gaussian.pack()
#         self.one_over_x_to_the_fourth.pack()
#         self.filter_hamming.pack()
#         self.cosine_on_ped.pack()
#
#         self.selected_value.set(1)


class LoadImage(AbstractWidgetPanel):
    file_selector = FileSelector            # type: FileSelector
    chip_size_panel = ChipSizePanel         # type: ChipSizePanel
    # selection_filter = SelectionFilter      # type: SelectionFilter

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        widgets_list = ["file_selector", "chip_size_panel"]

        self.init_w_basic_widget_list(widgets_list, n_rows=2, n_widgets_per_row_list=[1, 2])
        #self.selection_filter.set_radio_buttons()

        self.file_selector.set_fname_filters([("NITF files", ".nitf .NITF .ntf .NTF")])
        self.pack()
