from sarpy.annotation.annotate import FileAnnotationCollection
from sarpy.annotation.schema_processing import LabelSchema
from sarpy.annotation.annotate import Annotation
from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas import ImageCanvas
import os


class AppVariables:
    def __init__(self):
        self.image_fname = None     # type: str

        # set up label schema stuff
        self.label_schema = LabelSchema
        self.file_annotation_collection = FileAnnotationCollection
        self.file_annotation_fname = None                   # type: str
        self.canvas_geom_ids_to_annotations_id_dict = {}
        self.annotate_canvas = ImageCanvas                  # type: ImageCanvas
        self.context_canvas = ImageCanvas                   # type: ImageCanvas
        self.new_annotation = False                         # type: bool
