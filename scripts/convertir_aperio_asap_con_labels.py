import xml.etree.ElementTree as ET
import os

# Clasificación por palabras clave
completa_keys = ["intestino delgado", "completa"]
incompleta_keys = ["colon", "colónico", "intestino grueso", "incompleta"]
excluir_keys = ["no veo", "no le veo", "dificil", "difícil", "olgim", "no metaplasia", "no tener en cuenta", "inflamación"]

def clasificar_texto(texto):
    t = texto.lower()
    if any(k in t for k in excluir_keys):
        return None
    if any(k in t for k in completa_keys):
        return "metaplasia_completa"
    if any(k in t for k in incompleta_keys):
        return "metaplasia_incompleta"
    return None

def parse_color(color_str):
    if "," in color_str:
        try:
            r, g, b = map(int, color_str.split(","))
            return f"#{r:02X}{g:02X}{b:02X}"
        except:
            return "#00FF00"
    else:
        try:
            return "#" + f"{int(color_str):06X}"
        except:
            return "#00FF00"

def procesar_y_convertir(input_path, output_path, mpp=1.0):
    tree = ET.parse(input_path)
    root = tree.getroot()

    new_root = ET.Element("ASAP_Annotations")
    annotations_tag = ET.SubElement(new_root, "Annotations")

    contador = {
        "metaplasia_completa": 0,
        "metaplasia_incompleta": 0
    }

    for i, annotation in enumerate(root.findall(".//Annotation")):
        region = annotation.find(".//Region")
        if region is None:
            continue

        texto = region.attrib.get("Text", "").strip()
        clase = clasificar_texto(texto)
        if not clase:
            continue  # Ignorar si no clasifica

        index = contador[clase]
        contador[clase] += 1

        name = f"{clase}_{index}"
        color = annotation.attrib.get("LineColor", "65280")
        hex_color = parse_color(color)

        true_anno = ET.SubElement(annotations_tag, "Annotation", {
            "Name": name,
            "Type": "Polygon",
            "PartOfGroup": "None",
            "Color": hex_color
        })

        coords = ET.SubElement(true_anno, "Coordinates")
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
    print(f"✔ Archivo convertido y clasificado: {output_path}")

# Procesamiento por carpeta
def batch_procesar_convertir(directorio, mpp=1.0):
    for archivo in os.listdir(directorio):
        if archivo.endswith(".xml"):
            input_path = os.path.join(directorio, archivo)
            output_path = os.path.join(directorio, archivo.replace(".xml", "_asap.xml"))
            procesar_y_convertir(input_path, output_path, mpp)

# --- Ejecución ---
if __name__ == "__main__":
    carpeta = "/data/ricardo/dataMetaplasiaCompletaIncompleta/AperioAnotacionesTipoGlandulas/AlgortimoUnion/JoseMiguel/"  # 🔁 CAMBIAR AQUÍ
    batch_procesar_convertir(carpeta, mpp=1.0)
