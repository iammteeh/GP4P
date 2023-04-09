import xml.etree.ElementTree as ET

def xml_to_newick(xml_string):
    def newick_recursive(element):
        if element.tag == "configurationOption":
            name = element.find("name").text.strip()
            if len(element) == 1:  # Leaf node
                return f"{name}"
            else:
                children_newick = ",".join(newick_recursive(child) for child in element.findall("./"))
                return f"({children_newick}){name}"
        return ""

    root = ET.fromstring(xml_string)
    newick_tree = newick_recursive(root) + ";"
    return newick_tree
