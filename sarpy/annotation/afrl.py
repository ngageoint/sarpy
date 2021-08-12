"""
Basic usage for the AFRL labeling structure files
"""

__classification__ = "UNCLASSIFIED"
__authors__ = "Thomas McCullough"

# TODO: it's not yet clear what should be here

if __name__ == '__main__':
    import os
    from sarpy.io.xml.base import parse_xml_from_string
    from sarpy.annotation.afrl_elements.Research import ResearchType

    the_root = r'C:\Users\jr80407\Downloads\AFRL_information'
    fname = 'sicd_example_1_PFA_RE32F_IM32F_VV-research.xml'
    the_file = os.path.join(the_root, fname)
    with open(the_file, 'r') as fi:
        xml_string = fi.read()

    root_node, xml_ns = parse_xml_from_string(xml_string)

    research = ResearchType.from_node(root_node, xml_ns)
    print(research.DetailImageInfo.Width_3dB)

    fname_out = 'new_{}'.format(fname)
    with open(os.path.join(the_root, fname_out), 'wb') as fi:
        fi.write(research.to_xml_bytes())
