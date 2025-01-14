
def generate_xml_annotations(gland_annotations, output_path):
    import xml.etree.ElementTree as ET

    """
    Crea un archivo XML con las anotaciones de cada glándula segmentada compatible con ASAP.

    Parameters:
        gland_annotations (list of dict): Lista de anotaciones, cada una con coordenadas y clase.
        output_path (str): Ruta del archivo XML de salida.
    """
    # Crear la estructura XML inicial
    asap_annotations = ET.Element("ASAP_Annotations")
    annotations = ET.SubElement(asap_annotations, "Annotations")

    for i, gland in enumerate(gland_annotations):
        annotation = ET.SubElement(
            annotations, "Annotation",
            Name=str(i+1),
            Type="Polygon",
            PartOfGroup="None",
            Color="#F4FA58"  # Ajusta el color según necesites
        )

        # Añadir las coordenadas de la glándula
        coordinates = ET.SubElement(annotation, "Coordinates")
        for order, (x, y) in enumerate(gland['coords']):
            ET.SubElement(
                coordinates, "Coordinate",
                Order=str(order),
                X=str(x),
                Y=str(y)
            )

    # Añadir el elemento AnnotationGroups vacío
    ET.SubElement(asap_annotations, "AnnotationGroups")

    # Guardar el XML en el archivo de salida
    tree = ET.ElementTree(asap_annotations)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"XML guardado en: {output_path}")



