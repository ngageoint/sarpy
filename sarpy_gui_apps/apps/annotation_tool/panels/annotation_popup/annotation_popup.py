from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets
from sarpy_gui_apps.apps.annotation_tool.main_app_variables import AppVariables
from sarpy.annotation.annotate import Annotation
import tkinter


class AppVariables:
    parent_types_main_text = ""


class AnnotationPopup(AbstractWidgetPanel):
    parent_types = basic_widgets.Label      # type: basic_widgets.Label
    thing_type = basic_widgets.Combobox  # type: basic_widgets.Combobox
    reset = basic_widgets.Button        # type: basic_widgets.Button
    submit = basic_widgets.Button       # type: basic_widgets.Button
    comment = basic_widgets.Entry       # type: basic_widgets.Entry

    def __init__(self,
                 parent,
                 main_app_variables,    # type: AppVariables
                 ):
        self.label_schema = main_app_variables.label_schema
        self.main_app_variables = main_app_variables
        self.variables = AppVariables

        master_frame = tkinter.Frame(parent)
        AbstractWidgetPanel.__init__(self, master_frame)
        widget_list = ["parent_types", "thing_type", "reset", "submit", "comment"]
        self.init_w_basic_widget_list(widget_list, 3, [1, 1, 3])
        self.set_label_text("annotate")

        # set up base types for initial dropdown menu
        self.setup_main_parent_selections()

        master_frame.pack()
        self.pack()

        self.parent_types.set_text(self.variables.parent_types_main_text)

        # set up callbacks
        self.thing_type.on_selection(self.callback_update_selection)
        self.reset.on_left_mouse_click(self.callback_reset)
        self.submit.on_left_mouse_click(self.callback_submit)

    def callback_update_selection(self, event):
        selection_id = [key for key, val in self.label_schema.labels.items() if val == self.thing_type.get()][0]
        children_ids = []
        for key, val in self.label_schema.parent_types.items():
            if selection_id in val:
                child_index = val.index(selection_id) - 1
                if child_index >= 0:
                    children_ids.append(val[child_index])
        children_ids = set(children_ids)
        child_labels = []
        current_parent_text = self.parent_types.get_text()
        if current_parent_text == self.variables.parent_types_main_text:
            new_parent_text = self.label_schema.labels[selection_id]
        else:
            new_parent_text = current_parent_text + " -> " + self.label_schema.labels[selection_id]
        if children_ids:
            for id in children_ids:
                child_labels.append(self.label_schema.labels[id])
            self.thing_type.update_combobox_values(child_labels)
            self.parent_types.set_text(new_parent_text)
        else:
            self.thing_type.configure(state="disabled")

    def callback_reset(self, event):
        self.thing_type.configure(state="normal")
        self.setup_main_parent_selections()

    def callback_submit(self, event):
        comment_text = self.comment.get()
        thing_type = self.thing_type.get()
        stop = 1

    def setup_main_parent_selections(self):
        base_type_ids = []
        for type_id, parents in self.label_schema.parent_types.items():
            main_parent = parents[-1]
            base_type_ids.append(main_parent)
        base_type_ids = set(base_type_ids)
        base_labels = []
        for type_id in base_type_ids:
            base_labels.append(self.label_schema.labels[type_id])
        self.thing_type.update_combobox_values(base_labels)
        self.parent_types.set_text("")
