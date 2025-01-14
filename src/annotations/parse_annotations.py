import numpy as np
def calculate_segment_properties(coords):
    """Calcula las propiedades de los segmentos entre puntos consecutivos."""
    vectors = np.diff(coords, axis=0, append=[coords[0]])
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / np.pi
    return lengths, angles

def detect_straight_edges(coords, length_threshold=160, angle_tolerance=3):
    """Detecta segmentos rectos en las coordenadas."""
    lengths, angles = calculate_segment_properties(coords)
    for length in lengths:
        if length > length_threshold:
            return True
    return False

def parse_annotations(xml_path):
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon
    """Carga las anotaciones de un archivo XML y detecta anomalías."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    anomalies = []
    for annotation in root.iter('Annotation'):
        coords = [(float(coord.get('X')), float(coord.get('Y'))) for coord in annotation.iter('Coordinate')]
        poly = Polygon(coords)
        if detect_straight_edges(coords, length_threshold=140):
            anomalies.append({
                "polygon": poly,
                "id": annotation.attrib.get("Name", "Unknown")
            })
    return anomalies