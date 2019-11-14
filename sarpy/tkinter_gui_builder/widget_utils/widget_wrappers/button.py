import tkinter as tk


class Button(tk.Button):
    def __init__(self, parent):
        self.parent = parent
        tk.Button.__init__(self, parent)

    # def ignore(self, event):
    #     print("ignore")
    #     return "break"

    # events
    def on_left_mouse_click(self, event):
        # self.parent.bind("<Button-1>", self.ignore)
        self.bind("<Button-1>", event)

    # callbacks
    def callback_set_text(self, text):
        self.config(text=text)
