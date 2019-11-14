import tkinter as tk


class VerticalButtonPanel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.button_1 = tk.Button(self, text="button_1")
        self.button_1.pack()
        self.button_2 = tk.Button(self, text="button_2")
        self.button_2.pack()
