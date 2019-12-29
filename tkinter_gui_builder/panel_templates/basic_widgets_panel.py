import abc
from six import add_metaclass
import tkinter as tk
import numpy as np


@add_metaclass(abc.ABCMeta)
class BasicWidgetsPanel(tk.LabelFrame):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent)
        self.config(borderwidth=2)
        self.widget_list = None     # type: list
        self.rows = None           # type: tk.Frame

    def init_w_xy_positions_dict(self, positions_dict):
        # TODO: just find the right order then call init_w_basic_widget_list
        stop = 1

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

    def init_w_box_layout(self,
                          basic_widget_list,                # type: list
                          n_columns,                        # type: int
                          widget_width=None,                # type: int
                          widget_height=None,               # type: int
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
        for i, widget_and_name in enumerate(basic_widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(anchor='w')
            if widget_width is not None:
                getattr(self, widget).config(width=widget_width)
            if widget_height is not None:
                getattr(self, widget).config(height=widget_height)

    def init_w_basic_widget_list(self,
                                 basic_widget_list,         # type: list
                                 n_rows,                    # type: int
                                 n_widgets_per_row_list,    # type: list
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
        self.rows = [tk.Frame(self) for i in range(n_rows)]
        for row in self.rows:
            row.config(borderwidth=2)
            row.pack()

        # find transition points
        transitions = np.cumsum(n_widgets_per_row_list)
        self.widget_list = []
        row_num = 0
        for i, widget_and_name in enumerate(basic_widget_list):
            if i in transitions:
                row_num += 1
            widget = widget_and_name
            name = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
                name = widget_and_name[1]
            setattr(self, widget, getattr(self, widget)(self.rows[row_num]))
            getattr(self, widget).pack(side="left", padx=5, pady=5)
            getattr(self, widget).config(text=name)
            self.widget_list.append(widget)

    def set_spacing_between_buttons(self, spacing_npix_x=0, spacing_npix_y=None):
        if spacing_npix_y is None:
            spacing_npix_y = spacing_npix_x
        for widget in self.widget_list:
            getattr(self, widget).pack(side="left", padx=spacing_npix_x, pady=spacing_npix_y)

    def set_label_text(self,
                       label,               # type: str
                       ):
        self.config(text=label)

    def unpress_all_buttons(self):
        for i, widget_and_name in enumerate(self.widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(relief="raised")

    def press_all_buttons(self):
        for i, widget_and_name in enumerate(self.widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(relief="sunken")

    def activate_all_buttons(self):
        for i, widget_and_name in enumerate(self.widget_list):
            widget = widget_and_name
            if type(("", "")) == type(widget_and_name):
                widget = widget_and_name[0]
            getattr(self, widget).config(state="active")
