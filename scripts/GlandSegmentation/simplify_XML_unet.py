import os
import json
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.validation import make_valid

def simplify_polygon(vertices, tolerance=5.0):
    """Simplifica un polígono dado una tolerancia."""
    if len(vertices) < 4:
        return vertices  # No se puede simplificar un polígono con menos de 4 puntos
    poly = Polygon(vertices)
    simplified_poly = poly.simplify(tolerance, preserve_topology=True)
    if not simplified_poly.is_valid:
        simplified_poly = make_valid(simplified_poly)
    if isinstance(simplified_poly, Polygon) and len(simplified_poly.exterior.coords) >= 4:
        return list(simplified_poly.exterior.coords)
    else:
        return vertices
def reduce_points_in_xml(xml_path, output_path, tolerance=5.0):
    """Simplifica todos los polígonos del XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for annotation in root.findall(".//Annotation"):
        coordinates_elem = annotation.find("Coordinates")
        vertices = [
            (float(coord.get("X")), float(coord.get("Y")))
            for coord in coordinates_elem.findall("Coordinate")
        ]
        simplified_vertices = simplify_polygon(vertices, tolerance)
        for coord in coordinates_elem.findall("Coordinate"):
            coordinates_elem.remove(coord)
        for order, (x, y) in enumerate(simplified_vertices):
            ET.SubElement(coordinates_elem, "Coordinate", {"Order": str(order), "X": str(x), "Y": str(y)})

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"📦 XML simplificado guardado en: {output_path}")

def main():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    detections_path_xml = config["processed_path"]
    suffix_in = config.get("suffix_final_unet", "_completed_Unet.xml")
    suffix_out = config.get("suffix_simplified", "_simplified_Unet.xml")

    detections = [f for f in os.listdir(detections_path_xml) if f.endswith(suffix_in)]

    for detection_file in detections:
        case_name = detection_file.replace(suffix_in, "")
        xml_path = os.path.join(detections_path_xml, detection_file)
        output_path = os.path.join(detections_path_xml, f"{case_name}{suffix_out}")

        print(f"🧪 Simplificando XML: {detection_file}")
        reduce_points_in_xml(xml_path, output_path, tolerance=10.0)

if __name__ == "__main__":
    main()
