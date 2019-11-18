import abc
from six import add_metaclass
import tkinter as tk
import numpy as np


@add_metaclass(abc.ABCMeta)
class BasicButtonPanel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.config(highlightbackground="black")
        self.config(highlightthickness=1)
        self.config(borderwidth=5)
        self.rows = None           # type: tk.Frame

    def init_w_xy_positions_dict(self, positions_dict):
        # just find the right order then call init_w_basic_widget_list
        stop = 1

    def init_w_basic_widget_list(self,
                                 basic_widget_list,         # type: list
                                 n_rows,                    # type: int
                                 n_widgets_per_row_list,    # type: list
                                 ):
        """
        This is a convenience method to initialize a basic widget panel.  To use this first make a subclass
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
        row_num = 0
        for i, widget in enumerate(basic_widget_list):
            if i in transitions:
                row_num += 1
            setattr(self, widget, getattr(self, widget)(self.rows[row_num]))
            getattr(self, widget).pack(side="left", padx=5, pady=5)
            getattr(self, widget).config(bd=2)



