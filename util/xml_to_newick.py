import xml.etree.ElementTree as ET

def xml_to_newick(xml_string):
    def newick_recursive(element, attribute="name"):
        if len(element) == 0:  # Leaf node
            name = element.text
            attribute = element.attrib.get(attribute, "")
            return f"{name}:{attribute}"
        else:
            children_newick = ",".join(newick_recursive(child) for child in element)
            attribute = element.attrib.get(attribute, "")
            return f"({children_newick}){attribute}"

    root = ET.fromstring(xml_string)
    return newick_recursive(root) + ";"
