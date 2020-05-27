import numpy as np
from typing import Union
import tkinter

NO_TEXT_UPDATE_WIDGETS = ['ttk::scale']


class AbstractWidgetPanel(tkinter.LabelFrame):
    def __init__(self, parent):
        self.parent = parent
        tkinter.LabelFrame.__init__(self, parent)
        self.config(borderwidth=2)
        self._widget_list = None     # type: list
        self.rows = None           # type: tkinter.Frame

    def close_window(self):
        self.parent.withdraw()

    def init_w_horizontal_layout(self,
                                 basic_widget_list,         # type: list
                                 ):
        self.init_w_basic_widget_list(basic_widget_list,
                                      n_rows=1,
                                      n_widgets_per_row_list=[len(basic_widget_list)])

    def init_w_vertical_layout(self,
                               basic_widget_list,           # type: list
                               ):
        self.init_w_basic_widget_list(basic_widget_list,
                                      n_rows=len(basic_widget_list),
                                      n_widgets_per_row_list=list(np.ones(len(basic_widget_list))))

    def init_w_rows(self, basic_widgets_nested_list):
        flattened_list = []
        widgets_per_row = []
        for widget_list in basic_widgets_nested_list:
            widgets_per_row.append(len(widget_list))
            for widget in widget_list:
                flattened_list.append(widget)
        self.init_w_basic_widget_list(flattened_list, len(basic_widgets_nested_list), widgets_per_row)

    def init_w_box_layout(self,
                          basic_widget_list,  # type: list
                          n_columns,  # type: int
                          column_widths=None,  # type: Union[int, list]
                          row_heights=None,  # type: Union[int, list]
                          ):
        n_total_widgets = len(basic_widget_list)
        n_rows = int(np.ceil(n_total_widgets/n_columns))
        n_widgets_per_row = []
        n_widgets_left = n_total_widgets
        for i in range(n_rows):
            n_widgets = n_widgets_left/n_columns
            if n_widgets >= 1:
                n_widgets_per_row.append(n_columns)
            else:
                n_widgets_per_row.append(n_widgets_left)
            n_widgets_left -= n_columns
        self.init_w_basic_widget_list(basic_widget_list, n_rows, n_widgets_per_row)
        for i, widget in enumerate(self._widget_list):
            column_num = np.mod(i, n_columns)
            row_num = int(i/n_columns)
            if column_widths is not None and isinstance(column_widths, type(1)):
                getattr(self, widget).config(width=column_widths)
            elif column_widths is not None and isinstance(column_widths, type([])):
                col_width = column_widths[column_num]
                getattr(self, widget).config(width=col_width)
            if row_heights is not None and isinstance(row_heights, type(1)):
                getattr(self, widget).config(height=row_heights)
            elif row_heights is not None and isinstance(row_heights, type([])):
                row_height = row_heights[row_num]
                getattr(self, widget).config(height=row_height)

    def init_w_basic_widget_list(self,
                                 basic_widget_list,         # type: [str]
                                 n_rows,                    # type: int
                                 n_widgets_per_row_list,    # type: [int]
                                 ):
        """
        This is a convenience method to initialize a basic widget panel.  To use this first make a subclass
        This should also be the master method to initialize a panel.  Other convenience methods can be made
        to perform the button/widget location initialization, but all of those methods should perform their
        ordering then reference this method to actually perform the initialization.
        :param basic_widget_list:
        :param n_rows:
        :param n_widgets_per_row_list:
        :return:
        """
        self.rows = [tkinter.Frame(self) for i in range(n_rows)]
        for row in self.rows:
            row.config(borderwidth=2)
            row.pack()

        # find transition points
        transitions = np.cumsum(n_widgets_per_row_list)
        self._widget_list = []
        row_num = 0
        for i, widget in enumerate(basic_widget_list):
            if i in transitions:
                row_num += 1
            old_widget = None
            if not hasattr(self, widget):
                old_widget = widget
                widget = widget + "_" + str(i)
                setattr(self, widget, tkinter.Label(self.rows[row_num]))
            else:
                setattr(self, widget, getattr(self, widget)(self.rows[row_num]))
            getattr(self, widget).pack(side="left", padx=5, pady=5)
            if getattr(self, widget).widgetName in NO_TEXT_UPDATE_WIDGETS:
                pass
            else:
                getattr(self, widget).config(text=widget.replace("_", " "))
            if old_widget is not None:
                getattr(self, widget).config(text=widget.replace("_", " ")[0:-2])
            self._widget_list.append(widget)

    def set_text_formatting(self, formatting_list):
        pass

    def set_spacing_between_buttons(self, spacing_npix_x=0, spacing_npix_y=None):
        if spacing_npix_y is None:
            spacing_npix_y = spacing_npix_x
        for widget in self._widget_list:
            getattr(self, widget).pack(side="left", padx=spacing_npix_x, pady=spacing_npix_y)

    def set_label_text(self,
                       label,               # type: str
                       ):
        self.config(text=label)

    def unpress_all_buttons(self):
        for i, widget_and_name in enumerate(self._widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            if getattr(self, widget).widgetName == "button":
                getattr(self, widget).config(relief="raised")

    def press_all_buttons(self):
        for i, widget_and_name in enumerate(self._widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            if getattr(self, widget).widgetName == "button":
                getattr(self, widget).config(relief="sunken")

    def activate_all_buttons(self):
        for i, widget_and_name in enumerate(self._widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(state="normal")

    def disable_all_buttons(self):
        for i, widget_and_name in enumerate(self._widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(state="disabled")

    def set_active_button(self,
                          button,
                          ):
        self.unpress_all_buttons()
        self.activate_all_buttons()
        button.config(state="disabled")
        button.config(relief="sunken")
