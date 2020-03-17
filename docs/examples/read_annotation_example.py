from sarpy.annotation.annotate import FileAnnotationCollection
import os

image_fname = os.path.expanduser('~/Data/sarpy_data/nitf/sicd_example_1_PFA_RE32F_IM32F_HH.nitf')
json_path = os.path.expanduser("~/tmp3.json")


annotation_collection = FileAnnotationCollection.from_file(json_path)

for feature in annotation_collection.annotations.features:
    coords = feature.geometry.get_coordinate_list()
    stop = 1