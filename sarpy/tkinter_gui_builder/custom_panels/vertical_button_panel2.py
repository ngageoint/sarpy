import tkinter as tk
import sarpy.tkinter_gui_builder.widget_utils.basic_widgets as basic_widgets


class VerticalButtonPanel2(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.button_1 = basic_widgets.Button(self)
        self.button_1.config(text="decrease")
        self.button_1.pack()
        self.button_2 = basic_widgets.Button(self)
        self.button_2.config(text="increase")
        self.button_2.pack()

    def on_button_click(self, callback):
        self.button_1.bind("<Button-1>", callback)

