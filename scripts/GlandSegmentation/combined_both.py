import os
import json
import xml.etree.ElementTree as ET
import re
def load_annotations(xml_path, color, start_index=1):
    """Carga anotaciones desde un archivo XML de ASAP, asigna color y renumera."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    for i, ann in enumerate(root.findall(".//Annotation")):
        ann.attrib["Name"] = f"Gland_{start_index + i}"
        ann.attrib["Color"] = color
        annotations.append(ann)

    return annotations, start_index + len(annotations)

def merge_xmls(unet_xml_path, sam_xml_path, output_path):
    """Combina anotaciones de SAM y UNet en un solo archivo XML ASAP."""
    root = ET.Element("ASAP_Annotations")
    annotations_elem = ET.SubElement(root, "Annotations")
    ET.SubElement(root, "AnnotationGroups")  # estructura mínima ASAP

    # Cargar y combinar UNet (verde)
    unet_annotations, next_index = load_annotations(unet_xml_path, "#00FF00", start_index=1)
    for ann in unet_annotations:
        annotations_elem.append(ann)

    # Cargar y combinar SAM (rojo)
    sam_annotations, _ = load_annotations(sam_xml_path, "#FF0000", start_index=next_index)
    for ann in sam_annotations:
        annotations_elem.append(ann)

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ Combinado guardado: {output_path}")

def main():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    input_folder = config.get("processed_path", "processed_data")
    output_folder = config.get("combined_path", os.path.join(input_folder, "combined"))
    os.makedirs(output_folder, exist_ok=True)

    suffix_unet = config.get("suffix_simplified", "_simplified_Unet.xml")
    suffix_sam = config.get("suffix_sam", "_corrected_final_corrected_final.xml")

    # Buscar archivos UNet
    unet_files = [f for f in os.listdir(input_folder) if f.endswith(suffix_unet)]

    for unet_file in unet_files:
        # Quitar la extensión
        case_name = os.path.splitext(unet_file)[0]
        # Eliminar el sufijo "_corrected"
        case_name = re.sub(r'_simplified_Unet$', '', case_name)

        case_id =  case_name
        unet_path = os.path.join(input_folder, unet_file)
        sam_file = f"{case_id}{suffix_sam}"
        sam_path = os.path.join(input_folder, sam_file)
        output_path = os.path.join(output_folder, f"{case_id}_combined.xml")

        if not os.path.exists(sam_path):
            print(f"⚠️ SAM no encontrado para {case_id}, saltando.")
            continue

        print(f"🔀 Combinando {case_id}...")
        merge_xmls(unet_path, sam_path, output_path)

if __name__ == "__main__":
    main()
