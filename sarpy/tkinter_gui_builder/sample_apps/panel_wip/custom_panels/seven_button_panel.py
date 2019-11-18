from sarpy.tkinter_gui_builder.panel_templates.basic_button_panel2 import BasicButtonPanel2
from sarpy.tkinter_gui_builder.widget_utils import basic_widgets


class SevenButtonPanel(BasicButtonPanel2):
    def __init__(self, parent):
        BasicButtonPanel2.__init__(self, parent)
        self.button1 = basic_widgets.Button
        self.button2 = basic_widgets.Button
        self.button3 = basic_widgets.Button
        self.button4 = basic_widgets.Button
        self.button5 = basic_widgets.Button
        self.button6 = basic_widgets.Button
        self.button7 = basic_widgets.Button

        self.init_w_basic_widget_list(['button1',
                                       'button2',
                                       'button3',
                                       'button4',
                                       'button5',
                                       'button6',
                                       'button7',
                                       ],
                                      3,
                                      [2, 3, 2])

        # self.init_w_xy_positions_dict({"button1": (0, 0),
        #                                "button2": (1, 0),
        #                                "button3": (0, 1),
        #                                "button4": (1, 1),
        #                                "button5": (2, 1),
        #                                "button6": (0, 2),
        #                                "button7": (1, 2)
        #                                })

        self.button1.config(text="button 1")
        self.button2.config(text="button 2")
        self.button3.config(text="button 3")
        self.button4.config(text="button 4")
        self.button5.config(text="button 5")
        self.button6.config(text="button 6")
        self.button7.config(text="button 7")

