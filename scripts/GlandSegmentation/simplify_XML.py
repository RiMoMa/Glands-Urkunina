"simplify_XML.py"
import os
import json
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.validation import make_valid

# Simplificar polígono

def simplify_polygon(vertices, tolerance=5.0):
    """
    Simplifica un polígono usando la tolerancia especificada.

    Args:
        vertices (list of tuples): Coordenadas del polígono.
        tolerance (float): Nivel de simplificación (mayor valor, más simplificación).

    Returns:
        list of tuples: Coordenadas simplificadas del polígono.
    """
    poly = Polygon(vertices)
    simplified_poly = poly.simplify(tolerance, preserve_topology=True)

    # Asegurarse de que el polígono simplificado sea válido
    if not simplified_poly.is_valid:
        simplified_poly = make_valid(simplified_poly)

    # Convertir a lista de coordenadas si el resultado es un polígono
    if isinstance(simplified_poly, Polygon):
        return list(simplified_poly.exterior.coords)
    return []

# Reducir puntos en XML

def reduce_points_in_xml(xml_path, output_path, tolerance=5.0):
    """
    Simplifica los polígonos en un archivo XML para reducir el número de puntos.

    Args:
        xml_path (str): Ruta al archivo XML de entrada.
        output_path (str): Ruta al archivo XML de salida.
        tolerance (float): Tolerancia para simplificar los polígonos.

    Returns:
        None
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for annotation in root.findall(".//Annotation"):
        coordinates_elem = annotation.find("Coordinates")
        vertices = [
            (float(coord.get("X")), float(coord.get("Y")))
            for coord in coordinates_elem.findall("Coordinate")
        ]

        # Simplificar los puntos del polígono
        simplified_vertices = simplify_polygon(vertices, tolerance)

        # Limpiar las coordenadas originales y añadir las simplificadas
        for coord in coordinates_elem.findall("Coordinate"):
            coordinates_elem.remove(coord)

        for order, (x, y) in enumerate(simplified_vertices):
            ET.SubElement(coordinates_elem, "Coordinate", {"Order": str(order), "X": str(x), "Y": str(y)})

    # Guardar el nuevo archivo XML
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Archivo simplificado guardado en: {output_path}")

# Pipeline principal

def main():
    # Leer configuración desde el archivo JSON
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Configuración dinámica basada en el archivo JSON
    detections_path_xml = config["processed_path"]

    # Listar archivos XML con sufijo "_completed.xml"
    detections = [f for f in os.listdir(detections_path_xml) if f.endswith(config["suffix_final"])]

    for detection_file in detections:
        case_name = os.path.splitext(detection_file)[0]
        print(f"Procesando simplificación de puntos en: {case_name}")

        xml_path = os.path.join(detections_path_xml, detection_file)
        simplified_output_path = os.path.join(detections_path_xml, f"{case_name}{config['suffix_final']}")

        # Simplificar el XML generado
        reduce_points_in_xml(xml_path, simplified_output_path, tolerance=10.0)

if __name__ == "__main__":
    main()
