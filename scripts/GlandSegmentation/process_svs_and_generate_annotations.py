import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import openslide
import pandas as pd
import re

import subprocess

import src.preprocessing.functionPatchesExtraction as functionPatchesExtraction
from src.preprocessing.normalization import normalize_dataset_with_matlab
import torch

import os
import glob
import subprocess
import src.utils.utils as utils
import psutil
import PIL.Image
import math

def generate_WSI(config, path_svs, case_name,output_im,output_folder):
    from skimage.measure import label, regionprops,find_contours
    from src.annotations.xml_generator import generate_xml_annotations
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    from src.utils.utils_SAM import create_binary_mask,process_batch_with_sam
    """
    Genera una imagen WSI con anotaciones de glándulas basada en los parámetros del pipeline.

    Parameters:
        config (dict): Configuración del pipeline, cargada desde `config.json`.
        path_svs (str): Ruta al archivo SVS de entrada.
        case_name (str): Nombre del caso a procesar.
    """
    # Leer configuraciones
    patchsize = config.get("patch_size", 512 * 3)
    batch_size = config.get("batch_size", 100)
    high_resol = config.get("high_resol", False)
    img_normalize = config.get("img_normalize", True)
    input_dataset = config["processed_path"]

    # Cargar SVS
    slide = openslide.OpenSlide(path_svs)
    svs_size = slide.level_dimensions[0]
    base_magnification = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    target_magnification = 40.0
    scale_factor = base_magnification / target_magnification if base_magnification > target_magnification else 1.0
    adjusted_patchsize = int(patchsize * scale_factor)

    # Validar memoria RAM disponible
    float32_image_size = patchsize * patchsize * 4 * 3
    available_ram = psutil.virtual_memory().available
    max_images_in_ram = available_ram // float32_image_size

    if max_images_in_ram < batch_size:
        print("Advertencia: La RAM disponible es limitada. Reduzca el tamaño del lote o la resolución.")

    # Generar coordenadas de extracción
    num_rows = svs_size[1] // adjusted_patchsize
    num_columns = svs_size[0] // adjusted_patchsize
    extraction_coords = {
        (i, j): (i * adjusted_patchsize, j * adjusted_patchsize)
        for i in range(num_columns) for j in range(num_rows)
    }

    num_batches = math.ceil(len(extraction_coords) / batch_size)



    # Configurar SAM
    sam_checkpoint = config["sam_checkpoint"]
    model_type = config["model_type"]
    device = config.get("device", "cuda:0")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Crear dataloader
    import src.preprocessing.dataloaderSVSGlands as dataloader
    val_dl = dataloader.DataLoader(
        data=extraction_coords,
        svs_path=path_svs,
        ImgNormalize=img_normalize,
        CaseName=case_name,
        batch_size=batch_size,
        patch_size=512,
        num_threads_in_multithreaded=1,
        seed_for_shuffle=5243,
        return_incomplete=True,
        shuffle=False,
        infinite=False,
        inputDataset=input_dataset,
    )

    gland_annotations = []
    num_batches = math.ceil(len(extraction_coords) / batch_size)

    # Procesar por lotes
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Procesando lotes"):
            val_batch = next(val_dl)
            imgs = val_batch['data']
            if len(imgs) == 0:
                continue

            total_preds = process_batch_with_sam(imgs, mask_generator=mask_generator)
            tile_coords = val_batch['tileCoords']
            tileInfo = val_batch['tileInfo']

            if not os.path.exists(output_im):
                data = np.zeros((patchsize, patchsize, 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')

            # Procesar cada predicción
            for total_pred, tInfo, img, (x, y) in zip(total_preds, tileInfo, imgs, tile_coords):
                # res = PIL.Image.fromarray(total_pred.astype(np.uint8)*255).resize(patchsize, Image.NEAREST)
                binary_mask = total_pred

                res = PIL.Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(
                    (adjusted_patchsize, adjusted_patchsize), Image.BILINEAR)

                imgRgb = np.transpose(img, (1, 2, 0)).astype(int)
                imgRgb = PIL.Image.fromarray((imgRgb).astype(np.uint8)).resize((adjusted_patchsize, adjusted_patchsize),
                                                                               Image.LANCZOS)

                ### mejor poner aca lo del xml

                canvasRgb = utils.overlay_boundary(np.array(imgRgb), np.array(res) * 1)

                imgRgb = PIL.Image.fromarray((canvasRgb * 255).astype(np.uint8)).resize(
                    (adjusted_patchsize, adjusted_patchsize),
                    Image.LANCZOS)
                # imgRgb.save('img_' + str(0) + '_' + str(0) + '.bmp')
                # ## kakadu
                # subprocess.call(
                #     ['/home/ricardo/kakadu/bin/kdu_compress', '-i', 'img_' + str(0) + '_' + str(0) + '.bmp', '-o', outputIm,
                #      'Creversible=yes', 'Clevels=4', 'Stiles={' + str(patchsize) + ',' + str(patchsize) + '}', 'Clayers=10',
                #      'Cprecincts={512,512},{256,256},{128,128},{64,64},{32,32}', 'Corder=LRCP',
                #      'ORGgen_plt=yes', 'ORGtparts=R', 'Cuse_sop=yes', 'Cuse_precincts=yes', '-frag',
                #      str(tInfo[1]) + ',' + str(tInfo[0]) + ',' + str(1) + ',' + str(1), 'Sdims={' + str(svs_size[1]) + ',' + str(svs_size[0]) + '}', 'Scomponents=3'],
                # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                binary_mask = (binary_mask > 0).astype(int)
                factor_mask = int(adjusted_patchsize / np.shape(total_pred)[0])
                contours = find_contours(binary_mask, level=0.5)
                for contour in contours:
                    # Escalar las coordenadas del contorno y ajustar a la posición en la imagen global
                    coords = [(x + int(point[1] * factor_mask), y + int(point[0] * factor_mask)) for point in
                              contour]

                    # Añadir la anotación de la glándula con los puntos del contorno
                    gland_annotations.append({
                        "coords": coords,
                        "class": 'gland'
                    })

                # Generar el XML con las anotaciones
            output_xml_path = os.path.join(output_folder, f"{case_name}.xml")
            generate_xml_annotations(gland_annotations, output_xml_path)

    # Generar archivo XML
    output_xml_path = os.path.join(input_dataset, f"{case_name}.xml")
    generate_xml_annotations(gland_annotations, output_xml_path)
    print(f"Anotaciones guardadas en: {output_xml_path}")

# path_svs = 'U001.svs'
# patchsize=(512*4,512*4)
# batch_size=50
# HighResol = False
# ImgNormalize= False
# outputIm='prueba2048NormalizedMetaplasiasolving.jpc'
# MetaplasiaPredsCase = MetaplasiaPreds.loc['U001']
# generate_WSI(patchsize, batch_size, path_svs, outputIm=outputIm, HighResol=HighResol, ImgNormalize=ImgNormalize,
#
#
#                      MetaplasiaPredsCase=MetaplasiaPredsCase,CaseName='U001')





def process_all_images_with_pipeline(config):
    """
    Procesa todos los archivos SVS en la carpeta especificada según el archivo config.json.

    Parameters:
        config (dict): Configuración cargada desde config.json.
    """
    # Obtener parámetros de configuración
    input_folder = os.path.join(config["input_path"], config["svs_subfolder"])
    output_folder = config["processed_path"]
    patchsize = config.get("patch_size", 512 * 3)
    batch_size = config.get("batch_size", 100)
    high_resol = config.get("high_resol", False)
    img_normalize = config.get("img_normalize", True)

    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Listar archivos SVS
    svs_files = [f for f in os.listdir(input_folder) if f.endswith(".svs")]
    if not svs_files:
        print(f"No se encontraron archivos SVS en {input_folder}.")
        return

    # Procesar cada archivo SVS
    for svs_file in svs_files:
        print(f"Procesando: {svs_file}")
        output_im = svs_file.replace('.svs', '.jpc')
        slide_path = os.path.join(input_folder, svs_file)
        output_xml_path = os.path.join(output_folder, f"{os.path.splitext(svs_file)[0]}.xml")

        # Verificar si el archivo XML ya existe
        if os.path.exists(output_xml_path):
            print(f"Archivo XML ya existe para {svs_file}, saltando.")
            continue
        case_name = os.path.basename(svs_file).replace('.svs', '')

        functionPatchesExtraction.generate_Patches(patchsize, batch_size, slide_path, outputIm=output_im, HighResol=high_resol,
                         ImgNormalize=img_normalize, CaseName=case_name,OutputPath=output_folder)

        normalize_dataset_with_matlab(case_name, output_folder)


        # Generar WSI y anotaciones
        generate_WSI(config, slide_path, case_name, output_im, output_folder)
        print(f"Procesamiento completado para {svs_file}.")

def main():
    # Cargar configuración
    with open("/media/ricardo/Datos/Project_GastricMorphometry/config.json", "r") as config_file:
        config = json.load(config_file)

    # Procesar imágenes con la configuración cargada
    process_all_images_with_pipeline(config)

if __name__ == "__main__":
    main()
