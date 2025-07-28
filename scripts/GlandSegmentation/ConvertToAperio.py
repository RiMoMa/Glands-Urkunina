import os
import xml.etree.ElementTree as ET

def convert_aperio_to_asap(aperio_xml_path, asap_xml_path, mpp=0.25):
    tree = ET.parse(aperio_xml_path)
    root = tree.getroot()

    asap_root = ET.Element("Annotations")

    for i, annotation in enumerate(root.findall(".//Annotation")):
        name = annotation.attrib.get("Name", f"Annotation {i}")
        color = annotation.attrib.get("Color", "255,0,0")

        asap_anno = ET.SubElement(asap_root, "Annotation", {
            "Name": name,
            "Type": "Polygon",
            "PartOfGroup": name,
            "Color": color
        })

        coords_element = ET.SubElement(asap_anno, "Coordinates")
        for j, coord in enumerate(annotation.findall(".//Coordinate")):
            x = float(coord.attrib["X"]) * mpp
            y = float(coord.attrib["Y"]) * mpp
            ET.SubElement(coords_element, "Coordinate", {
                "Order": str(j),
                "X": str(x),
                "Y": str(y)
            })

    ET.ElementTree(asap_root).write(asap_xml_path, encoding="utf-8", xml_declaration=True)

def batch_convert(directory, mpp=0.25):
    for filename in os.listdir(directory):
        if filename.endswith(".xml") and not filename.endswith("_asap.xml"):
            aperio_path = os.path.join(directory, filename)
            asap_filename = filename.replace(".xml", "_asap.xml")
            asap_path = os.path.join(directory, asap_filename)
            print(f"Convirtiendo: {filename} -> {asap_filename}")
            convert_aperio_to_asap(aperio_path, asap_path, mpp)

# Uso
if __name__ == "__main__":
    carpeta = "/home/ricardo/Glands-Urkunina/processed_data_gland_type/combined"
    batch_convert(carpeta, mpp=0.25)
