import os
from tkinter_gui_builder.utils.geometry_utils.kml_util import KmlUtil

output_fname = os.path.expanduser("~/Downloads/test_kml.kml")

kml_util = KmlUtil()


def mockup():
    point = (94.75, 31.56)
    linestring = [(94, 31), (95, 30), (96, 35)]
    kml_util.add_point("point1", point)
    kml_util.add_linestring("linestring1", linestring)

    print(kml_util.write_to_python_terminal())
    kml_util.write_to_file(output_fname)
