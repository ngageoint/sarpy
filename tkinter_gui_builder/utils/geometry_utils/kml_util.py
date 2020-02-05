from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree import ElementTree
from xml.dom import minidom
from tkinter_gui_builder.utils.geometry_utils.kml_constants import KMLConstants


class KmlUtil:
    def __init__(self):
        self.xml_top_element = Element('kml', {'xmlns': "http://www.opengis.net/kml/2.2"})
        self.xml_document = SubElement(self.xml_top_element, "Document")
        self.indent = "  "

    def write_to_file(self, output_fname):
        """Return a pretty-printed XML string for the Element.
        """
        all_text = ElementTree.tostring(self.xml_top_element, encoding='UTF-8', method='xml')
        file_handle = open(output_fname, mode='wb')
        file_handle.write(all_text)
        file_handle.close()

    def write_to_string(self):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(self.xml_top_element, 'UTF-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", )

    def add_point(self, name, xy_coord, description=None):
        points = [xy_coord]
        self._add_shape_by_name_and_list_of_coords(name, KMLConstants.POINT, points, description)

    def add_linestring(self,
                       name,  # type: str
                       xy_coords,  # type: [(float, float, float)]
                       description=None,  # type: str
                       ):
        self._add_shape_by_name_and_list_of_coords(name, KMLConstants.LINESTRING, xy_coords, description)

    def add_polygon(self, name, xy_coords, description=None):
        placemark = SubElement(self.xml_document, KMLConstants.PLACEMARK)
        placemark_name = SubElement(placemark, KMLConstants.NAME)
        placemark_name.text = name
        placemark_polygon = SubElement(placemark, KMLConstants.POLYGON)
        outer_boundary_is = SubElement(placemark_polygon, KMLConstants.OUTERBOUNDARYIS)
        linear_ring = SubElement(outer_boundary_is, KMLConstants.LINEARRING)
        placemark_coords = SubElement(linear_ring, KMLConstants.COORDINATES)
        linestring_text = "\n"
        for coord in xy_coords:
            linestring_text = linestring_text + self.indent * 7 + (str(coord)[1:-1])
            linestring_text = linestring_text + "\n"
        linestring_text = linestring_text + self.indent * 6
        placemark_coords.text = linestring_text
        if description:
            placemark_description = SubElement(placemark, KMLConstants.DESCRIPTION)
            placemark_description.text = description

    def _add_shape_by_name_and_list_of_coords(self,
                                              name,  # type: str
                                              shape_type,       # type: str
                                              xy_coords,  # type: [(float, float, float)]
                                              description=None,  # type: str
                                              ):
        placemark = SubElement(self.xml_document, KMLConstants.PLACEMARK)
        placemark_name = SubElement(placemark, KMLConstants.NAME)
        placemark_name.text = name
        placemark_shape = SubElement(placemark, shape_type)
        placemark_coords = SubElement(placemark_shape, KMLConstants.COORDINATES)
        linestring_text = "\n"
        for coord in xy_coords:
            linestring_text = linestring_text + self.indent * 5 + (str(coord)[1:-1])
            linestring_text = linestring_text + "\n"
        linestring_text = linestring_text + self.indent * 4
        placemark_coords.text = linestring_text
        if description:
            placemark_description = SubElement(placemark, KMLConstants.DESCRIPTION)
            placemark_description.text = description