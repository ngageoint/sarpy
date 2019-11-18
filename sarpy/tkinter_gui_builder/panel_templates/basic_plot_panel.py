import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class BasicPlotPanel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.config(highlightbackground="black")
        self.config(highlightthickness=1)
        self.config(borderwidth=5)
        self.rows = None           # type: tk.Frame

        fig = Figure()
        ax = fig.add_subplot(111)
        self.line, = ax.plot(range(10))

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=1)

    def callback_decrease(self, event):
        self.decrease()

    def decrease(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y - 0.2 * x)
        self.canvas.draw()

    def callback_increase(self, event):
        self.increase()

    def increase(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y + 0.2 * x)
        self.canvas.draw()
