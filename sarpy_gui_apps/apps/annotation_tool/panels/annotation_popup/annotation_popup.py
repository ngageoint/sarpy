from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets
from sarpy.annotation.annotate import LabelSchema
import tkinter


class AnnotationPopup(AbstractWidgetPanel):
    parent_types = basic_widgets.Label      # type: basic_widgets.Label
    thing_type = basic_widgets.Combobox  # type: basic_widgets.Combobox

    def __init__(self,
                 parent,
                 label_schema,          # type: LabelSchema
                 ):

        self.label_schema = label_schema

        master_frame = tkinter.Frame(parent)
        AbstractWidgetPanel.__init__(self, master_frame)
        widget_list = ["parent_types", "thing_type"]
        self.init_w_vertical_layout(widget_list)
        self.set_label_text("annotate")

        # set up base types for initial dropdown menu
        base_type_ids = []
        for id, parents in label_schema.parent_types.items():
            main_parent = parents[-1]
            base_type_ids.append(main_parent)
        base_type_ids = set(base_type_ids)
        base_labels = []
        for id in base_type_ids:
            base_labels.append(label_schema.labels[id])
        self.thing_type.update_combobox_values(base_labels)

        self.parent_types.set_text("parents: ")

        master_frame.pack()
        self.pack()

        # set up callbacks
        self.thing_type.on_selection(self.callback_update_selection)

    def callback_update_selection(self, event):
        selection_id = [key for key, val in self.label_schema.labels.items() if val == self.thing_type.get()][0]
        children_ids = []
        for key, val in self.label_schema.parent_types.items():
            if selection_id in val:
                child_index = val.index(selection_id) - 1
                if child_index >= 0:
                    children_ids.append(val[child_index])
        if children_ids != []:
            children_ids = set(children_ids)
            child_labels = []
            current_parent_text = self.parent_types.get_text()
            new_parent_text = current_parent_text + " -> " + self.label_schema.labels[selection_id]
            self.parent_types.set_text(new_parent_text)
            for id in children_ids:
                child_labels.append(self.label_schema.labels[id])
            self.thing_type.update_combobox_values(child_labels)
            stop = 1



