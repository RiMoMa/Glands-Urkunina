import os
from glob import glob
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.strtree import STRtree

def cargar_anotaciones(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for ann in root.findall(".//Annotation"):
        coords = ann.find(".//Coordinates")
        puntos = []
        for c in coords.findall("Coordinate"):
            x, y = float(c.attrib["X"].replace(",", ".")), float(c.attrib["Y"].replace(",", "."))
            puntos.append((x, y))
        if len(puntos) >= 3:
            poly = Polygon(puntos)
            if poly.is_valid and poly.area > 0:
                annotations.append((ann, poly))
    return tree, annotations

def filtrar_superpuestos(annotations):
    polys = [p for _, p in annotations]
    tree = STRtree(polys)
    keep = set()
    eliminados = set()

    for i, (_, poly_i) in enumerate(annotations):
        if i in eliminados:
            continue
        intersecantes = [j for j, (_, poly_j) in enumerate(annotations)
                         if j != i and j not in eliminados and poly_i.intersects(poly_j)]
        todos = intersecantes + [i]
        max_idx = max(todos, key=lambda k: annotations[k][1].area)
        keep.add(max_idx)
        eliminados.update(set(todos) - {max_idx})
    return keep

def escribir_nuevo_xml(tree, annotations, keep_set, output_path):
    root = tree.getroot()
    all_annotations = root.find("Annotations")
    for i, ann in enumerate(all_annotations.findall("Annotation")):
        if i not in keep_set:
            all_annotations.remove(ann)
    tree.write(output_path)

# === Procesamiento por carpeta ===
input_dir = "/home/ricardo/Glands-Urkunina/processed_data_gland_type/Glandulas_Tipo_xml/Andres"
output_dir = os.path.join(input_dir, "sin_superposiciones")
os.makedirs(output_dir, exist_ok=True)

xml_paths = glob(os.path.join(input_dir, "*.xml"))

for xml_file in xml_paths:
    try:
        tree, annotations = cargar_anotaciones(xml_file)
        keep_set = filtrar_superpuestos(annotations)
        output_file = os.path.join(output_dir, os.path.basename(xml_file))
        escribir_nuevo_xml(tree, annotations, keep_set, output_file)
        print(f"✅ Procesado: {os.path.basename(xml_file)}")
    except Exception as e:
        print(f"❌ Error en {xml_file}: {e}")
