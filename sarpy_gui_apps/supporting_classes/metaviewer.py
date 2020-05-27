import tkinter as tk
from tkinter import ttk
from collections import OrderedDict
from sarpy.io.complex.sicd import SICDType


class Metaviewer(ttk.Treeview):
    def __init__(self, master,      # type: tk.Tk
                 ):
        super().__init__(master)
        self.parent = master
        self.parent.geometry("800x600")
        self.pack(expand=True, fill='both')
        self.parent.protocol("WM_DELETE_WINDOW", self.close_window)

    def close_window(self):
        self.parent.withdraw()

    def add_node(self, k, v):
        for key, val in v.items():
            new_key = k + "_" + key
            if isinstance(val, OrderedDict):
                self.insert(k, 1, new_key, text=key)
                self.add_node(new_key, val)
            else:
                self.insert(k, 1, new_key, text=key + ": " + str(val))

    def create_w_sicd(self,
                      sicd_meta,            # type: SICDType
                      ):
        for k, v in sicd_meta.to_dict().items():
            self.insert("", 1, k, text=k)
            self.add_node(k, v)
