import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import numpy as np
from openslide import OpenSlide
from segment_anything import sam_model_registry, SamPredictor
import cv2
from src.utils.utils_SAM import apply_sam_on_patch
from src.utils.utils_SAM import extract_patch_from_anomaly
from src.annotations.xml_save_corrected_asap import save_to_asap_xml_with_normal_annotations
from src.annotations.parse_annotations import parse_annotations
# Configuración de SAM
# sam_checkpoint = "/media/ricardo/Datos/SegmentacionGlandulasFinal/SAM/segment-anything/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)

import json

def main():

    # Leer configuración
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    dataset_path = os.path.join(config["input_path"], config["svs_subfolder"])
    detections_path_xml = config["processed_path"]
    output_path = config["processed_path"]
    sam_checkpoint = config["sam_checkpoint"]
    model_type = config["model_type"]
    device = config["device"]

    # Configurar SAM dinámicamente
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    #dataset_path = "/media/ricardo/Datos/Data/Reviewed/"
    #detections_path_xml = "/media/ricardo/Datos/DataProcesadas2/"

    detections = [f for f in os.listdir(detections_path_xml) if f.endswith("corrected.xml")]

    for detection_file in detections:
        case_name = os.path.splitext(detection_file)[0]
        case_name = case_name[0:case_name.index("_")]
        print(case_name)
        slide_path = os.path.join(dataset_path, f"{case_name}.svs")
        xml_path = os.path.join(detections_path_xml, detection_file)
        output_xml_path_completed = os.path.join(detections_path_xml, f"{case_name}_completed.xml")

        if os.path.exists(slide_path) and os.path.exists(xml_path):
            if os.path.exists(output_xml_path_completed):
                print('File exists: ',output_xml_path_completed)
                continue
            print(f"Processing SAM refined in {case_name}...")
            anomalies = parse_annotations(xml_path)
            slide = OpenSlide(slide_path)
            wsi_coords_list = []
            masks = []

            for anomaly in anomalies:
                patch, (left, top, _, _) = extract_patch_from_anomaly(anomaly, slide)
                mask, wsi_coords = apply_sam_on_patch(patch, anomaly, predictor, left, top)
                # masks.append(mask)
                wsi_coords_list.append(wsi_coords)

            save_to_asap_xml_with_normal_annotations(
                xml_path,  # XML original
                anomalies,  # Anomalías detectadas
                masks,  # Máscaras de anomalías segmentadas
                wsi_coords_list,  # Coordenadas de máscaras
                output_xml_path_completed  # Ruta de salida
            )
        else:
            print(f"Missing data for {case_name}, skipping.")


if __name__ == "__main__":
    main()