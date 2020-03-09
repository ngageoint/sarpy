from sarpy.annotation.annotate import FileAnnotationCollection
from sarpy.annotation.schema_processing import LabelSchema
from sarpy.annotation.annotate import Annotation


class AppVariables:
    def __init__(self):
        self.image_fname = None     # type: str
        self.shapes_in_selector = []

        # set up label schema stuff
        self.label_schema = LabelSchema("0.0",
                                        {"1": "space",
                                         "2": "earth",
                                         "3": "land",
                                         "4": "water",
                                         "5": "ocean",
                                         "6": "lake",
                                         "7": "river",
                                         "8": "forest",
                                         "9": "grassland",
                                         "10": "pine",
                                         "11": "redwood"},
                                        subtypes={"2": ["3", "4"],
                                                  "3": ["8", "9"],
                                                  "4": ["5", "6", "7"],
                                                  "8": ["10", "11"]},
                                        confidence_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        permitted_geometries=["Polygon"]
                                        )
        self.file_annotation_collection = FileAnnotationCollection
        self.temp_annotation_submission = Annotation        # type: Annotation
        self.canvas_geom_ids_to_annotations_id_dict = {}
