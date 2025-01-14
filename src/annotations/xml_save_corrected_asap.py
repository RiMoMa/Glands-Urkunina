import xml.etree.ElementTree as ET

def save_to_asap_xml_with_normal_annotations(original_xml_path, anomalies, masks, wsi_coords, output_xml_path):
    """
    Guarda las nuevas máscaras segmentadas y las anotaciones originales no marcadas como anomalías en un único archivo XML.

    Parameters:
        original_xml_path (str): Ruta al XML original.
        anomalies (list): Lista de anomalías procesadas.
        masks (list): Máscaras generadas por SAM.
        wsi_coords (list): Coordenadas WSI de las máscaras.
        output_xml_path (str): Ruta para guardar el nuevo XML combinado.
    """
    # Cargar el XML original
    original_tree = ET.parse(original_xml_path)
    original_root = original_tree.getroot()

    # Crear el nuevo árbol XML
    new_tree = ET.Element("ASAP_Annotations")
    annotations_elem = ET.SubElement(new_tree, "Annotations")

    # Incluir anotaciones no marcadas como anomalías
    anomaly_ids = {anomaly["id"] for anomaly in anomalies}  # IDs de anomalías
    for annotation in original_root.iter('Annotation'):
        annotation_id = annotation.attrib.get("Name", "Unknown")
        if annotation_id not in anomaly_ids:
            # Copiar anotación original
            annotations_elem.append(annotation)

    # Incluir máscaras segmentadas de anomalías
    for i, (anomaly, mask_coords) in enumerate(zip(anomalies, wsi_coords)):
        annotation = ET.SubElement(annotations_elem, "Annotation", {
            "Name": f"Segmented_{anomaly['id']}",
            "Type": "Polygon",
            "PartOfGroup": "None",
            "Color": "#58FA58"
        })
        coordinates_elem = ET.SubElement(annotation, "Coordinates")
        for j, (x, y) in enumerate(mask_coords):
            ET.SubElement(coordinates_elem, "Coordinate", {"Order": str(j), "X": str(x), "Y": str(y)})

    # Guardar el nuevo archivo XML
    tree_output = ET.ElementTree(new_tree)
    tree_output.write(output_xml_path)