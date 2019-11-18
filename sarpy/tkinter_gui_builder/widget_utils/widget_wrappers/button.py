import tkinter as tk
import functools

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

    def on_left_mouse_click_with_args(self, event, args):
        # self.bind("<Button-1>", event)
        # self.bind("<Button-1>", functools.partial(self._partial_callback_w_param, param=args))
        self.bind("<Button-1>", lambda event, arg=args: self._partial_callback_w_param(event, arg))

    # callbacks
    def callback_set_text(self, text):
        self.config(text=text)

    def _partial_callback_w_param(self, event, param):
        print(param)