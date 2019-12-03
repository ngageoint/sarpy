import tkinter as tk
from tkinter import ttk
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets

app = tk.Tk()
app.geometry('200x100')

labelTop = tk.Label(app,
                    text="Choose your favourite month")
labelTop.grid(column=0, row=0)


comboExample = basic_widgets.Combobox(app,
                                      values=[
                                        "January",
                                        "February",
                                        "March",
                                        "April"])
print(dict(comboExample))
comboExample.grid(column=0, row=1)
comboExample.current(1)


def callback_print_selection(event):
    print(comboExample.get())

comboExample.update_combobox_values(['May', 'June', 'July', 'August'])

comboExample.on_selection(callback_print_selection)

print(comboExample.current(), comboExample.get())

app.mainloop()
