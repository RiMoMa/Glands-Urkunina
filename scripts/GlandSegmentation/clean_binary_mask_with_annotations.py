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

    Args:
        svs_path (str): Ruta al archivo SVS.
        xml_path (str): Ruta al archivo XML con anotaciones.
        output_xml_path (str): Ruta para guardar el nuevo archivo XML limpio.
        min_area (int): Área mínima para conservar una región (en píxeles).

    Returns:
        None
    """
    # Cargar la imagen SVS
    slide = OpenSlide(svs_path)

    # Extraer las dimensiones del nivel 0
    wsi_width, wsi_height = slide.level_dimensions[0]

    # Crear una máscara binaria de la resolución original
    binary_mask = np.zeros((wsi_height, wsi_width), dtype=np.uint8)

    # Cargar el XML y procesar las anotaciones
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Dibujar las anotaciones en la máscara binaria
    for annotation in root.iter('Annotation'):
        points = [
            (int(float(coord.get('X'))), int(float(coord.get('Y'))))
            for coord in annotation.iter('Coordinate')
        ]

        # Verificar si los puntos son válidos
        if len(points) < 3:
            print("Advertencia: Anotación con menos de 3 puntos, ignorando.")
            continue

        # Filtrar puntos fuera de rango
        points = [(x, y) for x, y in points if 0 <= x < wsi_width and 0 <= y < wsi_height]
        if len(points) < 3:
            print("Advertencia: Todos los puntos de esta anotación están fuera del rango, ignorando.")
            continue

        try:
            cv2.fillPoly(binary_mask, [np.array(points, dtype=np.int32)], 1)
        except Exception as e:
            print(f"Error al procesar una anotación: {e}")
            continue

    # Dilatar la máscara para suavizar bordes
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Encontrar contornos en la máscara dilatada
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear un nuevo XML compatible con ASAP
    output_xml = ET.Element("ASAP_Annotations")
    annotations_elem = ET.SubElement(output_xml, "Annotations")

    # Procesar cada contorno y agregarlo al nuevo XML
    for i, contour in enumerate(contours):
        # Filtrar contornos con menos de 3 puntos únicos
        if len(contour) < 3:
            continue

        # Crear un polígono con Shapely
        poly = Polygon([(point[0][0], point[0][1]) for point in contour])

        # Validar y corregir el polígono si es necesario
        if not poly.is_valid:
            poly = make_valid(poly)

        # Manejar casos donde make_valid genera una GeometryCollection
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            polygons_to_process = list(poly.geoms)
        elif isinstance(poly, Polygon):
            polygons_to_process = [poly]
        else:
            print(f"Tipo de geometría inesperada: {type(poly)}")
            continue

        # Procesar cada polígono resultante
        for sub_poly in polygons_to_process:
            if sub_poly.area > min_area:  # Filtrar regiones pequeñas
                annotation = ET.SubElement(annotations_elem, "Annotation", {
                    "Name": f"Gland_{i+1}",
                    "Type": "Polygon",
                    "PartOfGroup": "None",
                    "Color": "#F4FA58"
                })
                coordinates_elem = ET.SubElement(annotation, "Coordinates")
                for j, (x, y) in enumerate(sub_poly.exterior.coords):
                    ET.SubElement(coordinates_elem, "Coordinate", {"Order": str(j), "X": str(x), "Y": str(y)})

    # Guardar el nuevo XML
    tree_output = ET.ElementTree(output_xml)
    tree_output.write(output_xml_path)

    print(f"Nuevo XML guardado en: {output_xml_path}.")


def main():
    import json
        # Leer configuración
    with open("/media/ricardo/Datos/Project_GastricMorphometry/config.json") as config_file:
        config = json.load(config_file)

    dataset_path = os.path.join(config["input_path"], config["svs_subfolder"])
    detections_path_xml = os.path.join(config["processed_path"])
    output_path = config["processed_path"]


#    dataset_path = "/media/ricardo/Datos/Data/Reviewed/"
#    detections_path_xml = "/media/ricardo/Datos/DataProcesadas2/"

    # Listar archivos XML de detecciones y sus correspondientes SVS
    #detections = [f for f in os.listdir(detections_path_xml) if f.endswith(".xml")]

    # Lista de sufijos a excluir
    excluded_suffixes = ["_completed.xml", "_corrected.xml"]

    # Filtrar solo los archivos XML que no tienen los sufijos excluidos
    detections = [
        f for f in os.listdir(detections_path_xml)
        if f.endswith(".xml") and not any(f.endswith(suffix) for suffix in excluded_suffixes)
    ]

    for detection_file in detections:
        case_name = os.path.splitext(detection_file)[0]  # Obtener el nombre base del caso (sin extensión)
        svs_path = os.path.join(dataset_path, f"{case_name}.svs")
        xml_path = os.path.join(detections_path_xml, detection_file)
        output_xml_path = os.path.join(detections_path_xml, f"{case_name}_corrected.xml")
        print("cleaning :", svs_path)


        if os.path.exists(output_xml_path):
            print(f"File {output_xml_path} already exists. Skipping {case_name}...")
            continue

        if os.path.exists(svs_path) and os.path.exists(xml_path):
            print(f"Processing {case_name}...")
            clean_binary_mask_with_annotations(svs_path, xml_path, output_xml_path)
        else:
            print(f"Missing data for {case_name}, skipping.")

if __name__ == "__main__":
    main()
