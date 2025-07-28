import xml.etree.ElementTree as ET
import os
def parse_color(color_str):
    if "," in color_str:
        try:
            r, g, b = map(int, color_str.split(","))
            return f"#{r:02X}{g:02X}{b:02X}"
        except:
            return "#00FF00"  # fallback
    else:
        try:
            return "#" + f"{int(color_str):06X}"
        except:
            return "#00FF00"  # fallback

def convert_fake_asap_to_true_asap(input_path, output_path, mpp=0.25):
    tree = ET.parse(input_path)
    root = tree.getroot()

    new_root = ET.Element("ASAP_Annotations")
    annotations_tag = ET.SubElement(new_root, "Annotations")

    for i, annotation in enumerate(root.findall(".//Annotation")):
        name = annotation.attrib.get("Name", f"Annotation {i}")
        color = annotation.attrib.get("LineColor", "65280")
        color = annotation.attrib.get("LineColor", "65280")
        hex_color = parse_color(color)

        true_anno = ET.SubElement(annotations_tag, "Annotation", {
            "Name": name,
            "Type": "Polygon",
            "PartOfGroup": "None",
            "Color": hex_color
        })

        coords = ET.SubElement(true_anno, "Coordinates")
        region = annotation.find(".//Region")
        if region is None:
            continue

        vertices = region.find("Vertices")
        if vertices is None:
            continue

        for j, vtx in enumerate(vertices.findall("Vertex")):
            x = float(vtx.attrib["X"]) * mpp
            y = float(vtx.attrib["Y"]) * mpp
            ET.SubElement(coords, "Coordinate", {
                "Order": str(j),
                "X": str(x),
                "Y": str(y)
            })

    ET.ElementTree(new_root).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"✔ Guardado como ASAP válido: {output_path}")

# Uso en lote
def batch_convert(directory, mpp=1):
    for fname in os.listdir(directory):
        if fname.endswith(".xml"):
            input_path = os.path.join(directory, fname)
            output_path = os.path.join(directory, fname.replace(".xml", "_asap.xml"))
            convert_fake_asap_to_true_asap(input_path, output_path, mpp)


if __name__ == "__main__":
    carpeta = "/data/ricardo/dataMetaplasiaCompletaIncompleta/AperioAnotacionesTipoGlandulas/Jesus/"
    batch_convert(carpeta, mpp=1)
