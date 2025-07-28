import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from openslide import OpenSlide
from shapely.geometry import Polygon
from skimage.measure import find_contours

import torch
import segmentation_models_pytorch as smp

from src.annotations.parse_annotations import parse_annotations
from src.annotations.xml_save_corrected_asap import save_to_asap_xml_with_normal_annotations
from src.utils.utils import min_max_norm


def extract_patch_from_anomaly(anomaly, slide, patch_size=512):
    """Extrae un parche centrado en la anotación desde el WSI."""
    #coords = anomaly["coords"]
    coords = list(anomaly["polygon"].exterior.coords)

    poly = Polygon(coords)
    cx, cy = map(int, poly.centroid.coords[0])

    half = patch_size // 2
    left = max(cx - half, 0)
    top = max(cy - half, 0)

    patch = slide.read_region((left, top), 0, (patch_size, patch_size)).convert("RGB")
    patch_np = np.array(patch)

    return patch_np, (left, top)


def apply_unet_on_patch(patch_rgb, model, device, threshold=0.5):
    """Aplica UNet a un parche y retorna los contornos en coordenadas absolutas."""
    h, w, _ = patch_rgb.shape
    img = min_max_norm(patch_rgb.astype(np.float32))
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    mask = (pred > threshold).astype(np.uint8)
    contours = find_contours(mask, level=0.5)

    # Escalar coordenadas a espacio WSI
    rescaled_contours = [
        [(int(p[1] / mask.shape[1] * w), int(p[0] / mask.shape[0] * h)) for p in contour]
        for contour in contours
    ]
    return rescaled_contours


def main():
    # Leer configuración
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_path = os.path.join(config["input_path"], config["svs_subfolder"])
    detections_path_xml = config["processed_path"]
    device = config.get("device", "cuda")
    weight_path = config["unet_weight_path"]
    patch_size = config.get("patch_size", 512)

    # Cargar modelo UNet
    print("🔁 Cargando modelo UNet...")
    model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device).eval()

    detections = [f for f in os.listdir(detections_path_xml) if f.endswith("_corrected_Unet.xml")]

    for xml_file in detections:
        case_name = xml_file.replace("_corrected_Unet.xml", "")
        slide_path = os.path.join(dataset_path, f"{case_name}.svs")
        xml_path = os.path.join(detections_path_xml, xml_file)
        output_xml_path = os.path.join(detections_path_xml, f"{case_name}_completed_Unet.xml")

        if not os.path.exists(slide_path) or not os.path.exists(xml_path):
            print(f"⚠️ Archivos faltantes para {case_name}, saltando.")
            continue

        if os.path.exists(output_xml_path):
            print(f"✔️ Ya existe: {output_xml_path}")
            continue

        print(f"🧠 Refinando con UNet: {case_name}...")
        anomalies = parse_annotations(xml_path)
        slide = OpenSlide(slide_path)

        refined_coords = []
        refined_masks = []

        for anomaly in anomalies:
            #print(f"🔍 Anotación con error en {case_name}: {anomaly}")

            patch, (left, top) = extract_patch_from_anomaly(anomaly, slide, patch_size=patch_size)
            contours = apply_unet_on_patch(patch, model, device)

            for contour in contours:
                # Convertir contorno local a coordenadas absolutas
                global_contour = [(x + left, y + top) for x, y in contour]
                refined_coords.append(global_contour)
                refined_masks.append(None)  # No usamos máscaras explícitas, solo coordenadas

        # Crear nuevo archivo XML con los contornos refinados
        save_to_asap_xml_with_normal_annotations(
            xml_path,
            anomalies,
            refined_masks,
            refined_coords,
            output_xml_path
        )


if __name__ == "__main__":
    main()
