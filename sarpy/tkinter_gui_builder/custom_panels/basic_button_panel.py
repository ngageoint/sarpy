import tkinter as tk
import sarpy.tkinter_gui_builder.widget_utils.basic_widgets as basic_widgets


class BasicButtonPanel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.config(highlightbackground="black")
        self.config(highlightthickness=1)
        self.config(borderwidth=5)

        self.column_frame_1 = tk.Frame(self)
        self.column_frame_2 = tk.Frame(self)
        self.column_frame_3 = tk.Frame(self)

        self.column_frame_1.config(borderwidth=2)
        self.column_frame_2.config(borderwidth=2)
        self.column_frame_2.config(borderwidth=2)

        self.button_1 = basic_widgets.Button(self.column_frame_1)
        self.button_2 = basic_widgets.Button(self.column_frame_1)
        self.button_3 = basic_widgets.Button(self.column_frame_2)
        self.button_4 = basic_widgets.Button(self.column_frame_2)
        self.button_5 = basic_widgets.Button(self.column_frame_2)
        self.button_6 = basic_widgets.Button(self.column_frame_3)
        self.button_7 = basic_widgets.Button(self.column_frame_3)

        self.button_1.config(text="button 1")
        self.button_2.config(text="button 2")
        self.button_3.config(text="button 3")
        self.button_4.config(text="button 4")
        self.button_5.config(text="button 5")
        self.button_6.config(text="button 6")
        self.button_7.config(text="button 7")

        self.button_1.pack(side="left")
        self.button_2.pack(side="left")
        self.button_3.pack(side="left")
        self.button_4.pack(side="left")
        self.button_5.pack(side="left")
        self.button_6.pack(side="left")
        self.button_7.pack(side="left")

        self.column_frame_1.pack()
        self.column_frame_2.pack()
        self.column_frame_3.pack()
