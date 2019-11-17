import abc
from six import add_metaclass
import tkinter as tk
import sarpy.tkinter_gui_builder.widget_utils.basic_widgets as basic_widgets


@add_metaclass(abc.ABCMeta)
class BasicButtonPanel2(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.config(highlightbackground="black")
        self.config(highlightthickness=1)
        self.config(borderwidth=5)
        self.rows = None           # type: tk.Frame

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
        self.rows = [tk.Frame for i in range(n_rows)]
        for row in self.rows:
            row.config(borderwidth=2)
            row.pack()

        for widget in basic_widget_list:
            widget = widget(self.rows[0])
            widget.pack(side="left")
