import os
import pandas as pd
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
from pathlib import Path

# Rutas base
folder_regions_xml = '/data/ricardo/dataMetaplasiaCompletaIncompleta/AperioAnotacionesTipoGlandulas/AlgortimoUnion/Andres/'
folder_detected_glands = '/home/ricardo/Glands-Urkunina/processed_data_gland_type/combined/'
output_folder = './glands_labeled_xml/'
os.makedirs(output_folder, exist_ok=True)
def cargar_poligonos_para_edicion(xml_path):
    """Carga polígonos (solo los elementos <Annotation>) para edición de etiquetas"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    anotaciones = []
    for ann in root.findall(".//Annotation"):
        coords_tag = ann.find("Coordinates")
        if coords_tag is None:
            continue
        coords = [
            (float(c.attrib['X']), float(c.attrib['Y']))
            for c in coords_tag.findall("Coordinate")
        ]
        if len(coords) >= 3:
            poly = Polygon(coords)
            anotaciones.append((poly, ann))  # regresamos la etiqueta original
    return anotaciones, root

def cargar_poligonos_asap(xml_path):
    """Carga polígonos y etiquetas desde archivo XML tipo ASAP."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    anotaciones = []

    for ann in root.findall(".//Annotation"):
        etiqueta = ann.attrib.get("Name", "")
        for region in ann.findall(".//Region"):
            vertices = region.find("Vertices")
            coords = [(float(v.attrib['X']), float(v.attrib['Y'])) for v in vertices.findall("Vertex")]
            if len(coords) >= 3:
                poligono = Polygon(coords)
                anotaciones.append((poligono, etiqueta, region))
    return anotaciones, root
def cargar_poligonos_generico(xml_path):
    """Carga polígonos de un XML (ASAP clásico o con <Coordinates>)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    anotaciones = []

    for ann in root.findall(".//Annotation"):
        etiqueta = ann.attrib.get("Name", "")
        coords_tag = ann.find("Coordinates")  # nuevo formato
        if coords_tag is not None:
            coords = [
                (float(c.attrib['X']), float(c.attrib['Y']))
                for c in coords_tag.findall("Coordinate")
            ]
        else:
            region = ann.find(".//Region")
            if region is None:
                continue
            coords_tag = region.find("Vertices")
            coords = [
                (float(v.attrib['X']), float(v.attrib['Y']))
                for v in coords_tag.findall("Vertex")
            ]
        if len(coords) >= 3:
            poly = Polygon(coords)
            anotaciones.append((poly, etiqueta, ann))
    return anotaciones, root

def crear_nuevo_xml(etiquetas_glandulas):
    """Crea un archivo ASAP XML con las glándulas y sus etiquetas."""
    new_root = ET.Element("ASAP_Annotations")
    annotations_tag = ET.SubElement(new_root, "Annotations")

    for idx, (poly, etiqueta) in enumerate(etiquetas_glandulas):
        annotation = ET.SubElement(annotations_tag, "Annotation", {
            "Type": "0", "LineColor": "16711680", "Name": etiqueta  # rojo
        })
        regions = ET.SubElement(annotation, "Regions")
        region = ET.SubElement(regions, "Region", {
            "Type": "0", "Id": str(idx), "Text": etiqueta
        })
        vertices_tag = ET.SubElement(region, "Vertices")
        for x, y in poly.exterior.coords:
            ET.SubElement(vertices_tag, "Vertex", {"X": str(x), "Y": str(y)})

    ET.SubElement(new_root, "Regions")
    return ET.ElementTree(new_root)

# === MAIN ===

# CSV de correspondencia
df = pd.read_csv("/data/ricardo/dataMetaplasiaCompletaIncompleta/correspondencia.csv", header=None, names=["svs", "idx"])

for _, row in df.iterrows():
    base_name = Path(row["svs"]).stem
    gland_file = os.path.join(folder_detected_glands, f"{base_name}_combined.xml")
    region_file = os.path.join(folder_regions_xml, f"{row['idx']}_asap.xml")

    if not os.path.exists(gland_file) or not os.path.exists(region_file):
        print(f"[!] Saltando {base_name} (faltan archivos)")
        continue

    # Cargar glándulas y regiones del patólogo
    # Cargar glándulas y regiones del patólogo
    tree = ET.parse(gland_file)
    root = tree.getroot()
    annotations_tag = root.find("Annotations")

    # Construir lista directamente desde annotations_tag
    gland_polygons = []
    for ann in annotations_tag.findall("Annotation"):
        coords_tag = ann.find("Coordinates")
        if coords_tag is not None:
            coords = [(float(c.attrib['X']), float(c.attrib['Y'])) for c in coords_tag.findall("Coordinate")]
            if len(coords) >= 3:
                gland_polygons.append((Polygon(coords), ann))

    region_polygons, _ = cargar_poligonos_generico(region_file)  # el del patólogo sí puede tener <Vertices>
    # gland_tree = ET.parse(gland_file)
    # gland_root = gland_tree.getroot()
    # annotations_tag = gland_root.find("Annotations")


    for gland_poly, ann_tag in gland_polygons:
        etiqueta = "no_intersection"
        for region_poly, etiqueta_patologo, _ in region_polygons:
            if gland_poly.intersects(region_poly):
                etiqueta = etiqueta_patologo
                break
        if etiqueta != "no_intersection":
            ann_tag.attrib["Name"] = etiqueta
        else:
            annotations_tag.remove(ann_tag)
            # ✅ solo cambia el nombre

    # Guardar el XML original con nombres de glándulas actualizados


    # Crear y guardar nuevo XML etiquetado
    salida_path = os.path.join(output_folder, f"{base_name}_labeled.xml")
    tree.write(salida_path)


    print(f"[✓] Guardado: {salida_path}")
