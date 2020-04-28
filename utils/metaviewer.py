import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import sarpy.io.complex as sarpy_complex
from collections import OrderedDict

# TODO: Object orient-ify this, turn it into a widget, and move it somewhere outside of utils
root = tk.Tk()
root.geometry("900x900")
tree = ttk.Treeview(root)
ttk.Style().configure('Treeview', rowheight=30)


def add_node(k, v):
    for key, val in v.items():
        new_key = k + "_" + key
        if isinstance(val, OrderedDict):
            tree.insert(k, 1, new_key, text=key)
            add_node(new_key, val)
        else:
            tree.insert(k, 1, new_key, text=key + ": " + str(val))


sicd_fname = askopenfilename(initialdir=os.path.expanduser("~"))
reader_object = sarpy_complex.open(sicd_fname)
sicd_meta_dict = reader_object.sicd_meta.to_dict()


for k, v in sicd_meta_dict.items():
    tree.insert("", 1, k, text=k)
    add_node(k, v)


tree.pack(expand=True, fill='both')
root.mainloop()
