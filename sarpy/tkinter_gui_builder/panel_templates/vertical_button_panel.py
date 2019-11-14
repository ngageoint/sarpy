import tkinter as tk
from collections import OrderedDict


class VerticalButtonPanel(tk.Frame):
    def __init__(self,
                 parent,                # type: tk.Frame
                 widget_dict,           # type: OrderedDict
                 ):
        tk.Frame.__init__(self, parent)
        self.button_array = []
        for
        self.button_1 = tk.Button(self, text="button_1")
        self.button_1.pack()
        self.button_2 = tk.Button(self, text="button_2")
        self.button_2.pack()
