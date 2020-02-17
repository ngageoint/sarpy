import os
from tkinter_gui_builder.utils.geometry_utils.kml_util import KmlUtil

output_fname = os.path.expanduser("~/Downloads/test_kml.kml")

kml_util = KmlUtil()


def mockup():
    point = (94.75, 31.56)
    linestring = [(94, 31), (95, 30), (96, 35)]
    polygon = [(92, 30), (93, 30), (93, 31), (92, 30)]
    kml_util.add_point("point1", point)
    kml_util.add_linestring("linestring1", linestring)
    kml_util.add_polygon("polygon1", polygon)

    kml_str = kml_util.write_to_string()
    print(kml_str)
    kml_util.write_to_file(output_fname)


if __name__ == '__main__':
    mockup()
