import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import cv2
import numpy as np
from openslide import OpenSlide

def clean_binary_mask_with_annotations(svs_path, xml_path, output_xml_path, min_area=64*64, kernel_size=7):
    """
    Limpia una máscara binaria generada a partir de anotaciones XML y elimina regiones pequeñas.
    Compatible con anotaciones ASAP generadas por el pipeline UNet.

    Args:
        svs_path (str): Ruta al archivo SVS.
        xml_path (str): Ruta al archivo XML con anotaciones.
        output_xml_path (str): Ruta para guardar el nuevo archivo XML limpio.
        min_area (int): Área mínima para conservar una región (en píxeles).
    """
    slide = OpenSlide(svs_path)
    wsi_width, wsi_height = slide.level_dimensions[0]
    binary_mask = np.zeros((wsi_height, wsi_width), dtype=np.uint8)

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"❌ Error al parsear {xml_path}")
        return

    root = tree.getroot()

    for annotation in root.iter('Annotation'):
        coords = [
            (int(float(coord.attrib['X'])), int(float(coord.attrib['Y'])))
            for coord in annotation.iter('Coordinate')
        ]
        if len(coords) < 3:
            continue
        coords = [(x, y) for x, y in coords if 0 <= x < wsi_width and 0 <= y < wsi_height]
        if len(coords) < 3:
            continue
        try:
            cv2.fillPoly(binary_mask, [np.array(coords, dtype=np.int32)], 1)
        except Exception as e:
            print(f"Error al dibujar una anotación: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_root = ET.Element("ASAP_Annotations")
    annotations_elem = ET.SubElement(new_root, "Annotations")

    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue
        poly = Polygon([(pt[0][0], pt[0][1]) for pt in contour])
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue

        # Manejo seguro de la geometría resultante
        if isinstance(poly, (Polygon, MultiPolygon)):
            polygons_to_process = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        elif poly.geom_type == 'GeometryCollection':
            polygons_to_process = [geom for geom in poly.geoms if isinstance(geom, Polygon)]
            if not polygons_to_process:
                print("⚠️ GeometryCollection sin polígonos válidos. Saltando.")
                continue
        else:
            print(f"⚠️ Tipo de geometría inesperado: {poly.geom_type}. Saltando.")
            continue

        for sub_poly in polygons_to_process:
            if sub_poly.area > min_area:
                ann = ET.SubElement(annotations_elem, "Annotation", {
                    "Name": f"Gland_{i + 1}",
                    "Type": "Polygon",
                    "PartOfGroup": "None",
                    "Color": "#F4FA58"
                })
                coords_elem = ET.SubElement(ann, "Coordinates")
                for j, (x, y) in enumerate(sub_poly.exterior.coords):
                    ET.SubElement(coords_elem, "Coordinate", {
                        "Order": str(j),
                        "X": str(x),
                        "Y": str(y)
                    })

    ET.SubElement(new_root, "AnnotationGroups")
    tree_output = ET.ElementTree(new_root)
    tree_output.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"[✓] Guardado: {output_xml_path}")



def main():
    import json
    with open("config.json") as f:
        config = json.load(f)

    dataset_path = os.path.join(config["input_path"], config["svs_subfolder"])
    processed_path = config["processed_path"]
    output_suffix = "_corrected_Unet.xml"

    detections = [
        f for f in os.listdir(processed_path)
        if f.endswith("_annotations.xml") and not f.endswith(output_suffix)
    ]

    for xml_file in detections:
        case_name = xml_file.replace("_annotations.xml", "")
        svs_path = os.path.join(dataset_path, f"{case_name}.svs")
        xml_path = os.path.join(processed_path, xml_file)
        output_path = os.path.join(processed_path, f"{case_name}{output_suffix}")

        if not os.path.exists(svs_path):
            print(f"❌ SVS no encontrado: {svs_path}")
            continue
        if os.path.exists(output_path):
            print(f"✔️ Ya existe: {output_path}")
            continue

        print(f"🧽 Limpiando {case_name}...")
        clean_binary_mask_with_annotations(svs_path, xml_path, output_path)

if __name__ == "__main__":
    main()
