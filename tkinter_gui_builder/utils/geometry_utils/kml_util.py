from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree import ElementTree
from xml.dom import minidom


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

    def write_to_python_terminal(self):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(self.xml_top_element, 'UTF-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", )

    def add_point(self,
                  name,  # type: str
                  xy_coord,  # type: (float, float, float)
                  description=None,  # type: str
                  ):
        placemark = SubElement(self.xml_document, 'Placemark')
        placemark_name = SubElement(placemark, "name")
        placemark_name.text = name
        placemark_point = SubElement(placemark, "Point")
        placemark_coords = SubElement(placemark_point, "coordinates")
        placemark_coords.text = str(xy_coord)[1:-1]
        if description:
            placemark_description = SubElement(placemark, "description")
            placemark_description.text = description

    def add_linestring(self,
                  name,  # type: str
                  xy_coords,  # type: [(float, float, float)]
                  description=None,  # type: str
                  ):
        placemark = SubElement(self.xml_document, 'Placemark')
        placemark_name = SubElement(placemark, "name")
        placemark_name.text = name
        placemark_linestring = SubElement(placemark, "LineString")
        placemark_coords = SubElement(placemark_linestring, "coordinates")
        linestring_text = "\n"
        for coord in xy_coords:
            linestring_text = linestring_text + self.indent*4 + (str(coord)[1:-1])
            linestring_text = linestring_text + "\n"
        linestring_text = linestring_text + self.indent*3
        placemark_coords.text = linestring_text
        if description:
            placemark_description = SubElement(placemark, "description")
            placemark_description.text = description
