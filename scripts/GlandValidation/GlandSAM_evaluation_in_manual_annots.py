#1. Sacar las Mascaras Con SAM
from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
import cv2
import os
from src.utils.utils_SAM import create_binary_mask
import numpy as np
import csv
from scipy.io import loadmat
import numpy as np
import cv2
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy.io import loadmat
import os
import glob


def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

def dice_coefficient(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    return 2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))

# Función para extraer características por glándula
def extract_gland_features(image, mask):
    # Etiquetar glándulas en la máscara
    labeled_mask, num_glands = label(mask)
    properties = regionprops(labeled_mask)

    # Inicializar lista de características
    gland_features = []

    for prop in properties:
        gland = {
            "label": prop.label,
            "area": prop.area,
            "perimeter": prop.perimeter,
            "circularity": (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0,
            "centroid": prop.centroid
        }

        # Extraer ROI de la imagen original usando el bounding box
        minr, minc, maxr, maxc = prop.bbox
        gland_image = image[minr:maxr, minc:maxc]

        # Asegurarse de que el ROI no esté vacío
        if gland_image.size == 0:
            continue

        # Calcular matriz de coocurrencia (GLCM) para texturas
        glcm = graycomatrix(
            gland_image,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        gland["contrast"] = graycoprops(glcm, "contrast").mean()
        gland["homogeneity"] = graycoprops(glcm, "homogeneity").mean()
        gland["entropy"] = -np.sum(glcm * np.log2(glcm + 1e-10))

        gland_features.append(gland)

    return gland_features



import glob
import os
import cv2
import json

# Definir directorios
# Cargar configuración desde config.json
with open('scripts/GlandValidation/config_validation.json', 'r') as config_file:
    config = json.load(config_file)

manual_mask_dir = config["manual_mask_dir"]
image_dir = config["image_dir"]
output_dir = config["output_dir"]


# Cargar modelo SAM
sam_checkpoint = config["sam_checkpoint"]
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamAutomaticMaskGenerator(sam)




os.makedirs(output_dir, exist_ok=True)
masks_paths = glob.glob(f"{manual_mask_dir}/**/*.*", recursive=True)

# Buscar todas las imágenes recursivamente
image_paths = glob.glob(f"{image_dir}/**/*.*", recursive=True)

for img_path in image_paths:
    # Verificar si el archivo es una imagen válida
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        print(f"Procesando: {img_path}")
        relative_path = os.path.relpath(os.path.dirname(img_path), image_dir)  # Subcarpeta relativa
        save_dir = os.path.join(output_dir, relative_path)
        os.makedirs(save_dir, exist_ok=True)  # Crear subcarpeta en el output
        mask_output_path = os.path.join(save_dir, f"{os.path.basename(img_path)[:-4]}_mask.png")
        if os.path.exists(mask_output_path):
            print(f"Ya existe: {mask_output_path}, omitiendo...")
            continue

        # Leer la imagen
        image = cv2.imread(img_path)

        # Aquí va el procesamiento específico (predicción y generación de máscaras)
        masks = predictor.generate(image)
#        mask = create_binary_mask(masks, min_area_threshold=1000, max_area_threshold=12144, min_circularity_threshold=0.26)
        mask = create_binary_mask(masks, min_area_threshold=20000, max_area_threshold=2214400,
                                  min_circularity_threshold=0.46) # Guardar la máscara con un nombre basado en la ruta original

        cv2.imwrite(mask_output_path, mask.astype("uint8") * 255)


#2. Abrir las mascaras manuales



results = []

for filename in masks_paths:


    manual_mask = loadmat(os.path.join(filename))
    manual_mask = manual_mask["MaskL"]
    roi_mask = manual_mask > 0

    manual_mask = manual_mask* (manual_mask > 1)
    basename = os.path.basename(filename)
    # Eliminar la extensión
    name_no_ext = os.path.splitext(basename)[0]  # '227_078_21_he_6'

    # Extraer la parte sin el índice final
    base_id = "_".join(name_no_ext.split('_')[:-1])  # '227_078_21_he'

    # Determinar si es metaplasia o control
    if 'metaplasia' in filename:
        condition = 'metaplasia'
    elif 'control' in filename:
        condition = 'control'
    else:
        condition = 'unknown'
    image_path = os.path.join(image_dir,condition+"_isbi", base_id ,name_no_ext+ ".png")  # Asume extensión .png
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    output_name = os.path.join(output_dir,condition+'_isbi',base_id,name_no_ext + '_mask.png')

    auto_mask = cv2.imread(output_name)

    # Intersección entre máscaras
    roi_mask = roi_mask.astype('uint8')  # Convertir roi_mask si no es uint8
    auto_mask_channel = (auto_mask[:, :, 0] > 0).astype('uint8')  # Convertir auto_mask a uint8

    # Aplicar bitwise_and
    roi_automatic = cv2.bitwise_and(roi_mask, auto_mask_channel)

    iou = iou_score(manual_mask, roi_automatic)

    dice = dice_coefficient(manual_mask, roi_automatic)

    manual_features = extract_gland_features(image, manual_mask)
    auto_features = extract_gland_features(image, auto_mask)

    # Guardar resultados
    results.append({
        "filename": filename,
        "IoU": iou,
        "Dice": dice,
        "manual_features": manual_features,
        "auto_features": auto_features
    })

    # Guardar en CSV
with open("metrics_results.csv", "w", newline="") as csvfile:
    fieldnames = ["filename", "IoU", "Dice"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)



# Mostrar resultados por glándula
for result in results:
    print(f"Resultados para: {result['filename']}")
    print(f"IoU: {result['IoU']:.3f}, Dice: {result['Dice']:.3f}")
    for m, a in zip(result["manual_features"], result["auto_features"]):
        print(f"Glándula {m['label']}:")
        print(f" - Área Manual: {m['area']}, Área Automática: {a['area']}")
        print(f" - Circularidad Manual: {m['circularity']:.3f}, Circularidad Automática: {a['circularity']:.3f}")
        print(f" - Contraste Manual: {m['contrast']:.3f}, Contraste Automático: {a['contrast']:.3f}")
