import os
from tkinter_gui_builder.utils.geometry_utils.kml_util import KmlUtil

output_fname = os.path.expanduser("~/Downloads/test_kml.kml")

kml_util = KmlUtil()
kml_util.mockup(output_fname)
