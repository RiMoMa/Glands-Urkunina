import numpy as np
from skimage.measure import perimeter
import cv2
def create_binary_mask(anns, min_area_threshold=400, max_area_threshold=5000, min_circularity_threshold=0.8):
    if len(anns) == 0:
        return

    # Ordenar las anotaciones por área
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Crear una máscara binaria con el tamaño de la segmentación
    binary_mask = np.zeros((sorted_anns[0]['segmentation'].shape[0],
                            sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)

    # Llenar la máscara binaria solo con objetos que cumplan con el área y circularidad
    for ann in sorted_anns:
        area = ann['area']

        # Calcular perímetro usando skimage.measure.perimeter
        m = ann['segmentation']
        obj_perimeter = perimeter(m)

        # Calcular circularidad
        circularity = (4 * np.pi * area) / (obj_perimeter ** 2) if obj_perimeter > 0 else 0

        # Aplicar los filtros de área y circularidad
        if min_area_threshold <= area <= max_area_threshold and circularity >= min_circularity_threshold:
            binary_mask[m] = 1

    return binary_mask



# Uso de las funciones
#labels, canvas = process_glands(MetaplasiaPredsCase, res, x, y)
#save_npy(labels, 'labels.npy')
#save_npy(canvas, 'canvas.npy')
#overlay(canvas, imgRgb)
def process_batch_with_sam(batch,mask_generator):
    batch_results = []
    for img in batch:
        # Convertir tensor a formato numpy HWC compatible con SAM
        img_np = np.transpose(img,(1, 2, 0))  # Convierte el tensor PyTorch a NumPy (HWC)
        masks = mask_generator.generate(img_np.astype(np.uint8))
        masks = create_binary_mask(masks, min_area_threshold=1000, max_area_threshold=12144, min_circularity_threshold=0.26)
        batch_results.append(masks)
    return batch_results

def clean_binary_mask(binary_mask, min_area=100, max_area_ratio=0.6):
    """
    Limpia una máscara binaria eliminando ruido y detecciones demasiado grandes.

    Args:
        binary_mask (np.ndarray): Imagen binaria (valores 0 y 255).
        min_area (int): Área mínima para conservar una región.
        max_area_ratio (float): Proporción máxima de área respecto al tamaño total de la imagen.

    Returns:
        np.ndarray: Máscara binaria limpia.
    """
    # Asegurarse de que sea binaria
    #_, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # Encuentra contornos
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear máscara vacía para rellenar los contornos válidos
    cleaned_mask = np.zeros_like(binary_mask)

    # Área total de la imagen
    image_area = binary_mask.shape[0] * binary_mask.shape[1]

    # Filtrar contornos por área
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area_ratio * image_area:
            cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_mask

from src.utils.select_random_points_polygon import select_random_points_in_polygon
def apply_sam_on_patch(patch, anomaly, predictor, left, top, score_threshold=0.9):
    """Usa SAM para segmentar la región del parche alrededor de la anomalía."""
    input_point = np.array(select_random_points_in_polygon(anomaly["polygon"], left=left, top=top))
    input_label = np.array([1] * len(input_point))

    predictor.set_image(np.array(patch))
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    best_mask = masks[0]
    best_score = scores[0]

    # Asegurarse de que la máscara es binaria
    binary_mask = (best_mask > 0).astype(np.uint8)
    binary_mask = clean_binary_mask(binary_mask, min_area=40941, max_area_ratio=0.30)
    # Encontrar los contornos de la máscara
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convertir las coordenadas de los contornos al espacio original del WSI
    contour_coords = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            contour_coords.append((int(x + left), int(y + top)))

    return binary_mask, contour_coords



def extract_patch_from_anomaly(anomaly, slide, margin=600):
    """Extrae un parche centrado en la anomalía desde el WSI."""
    poly = anomaly["polygon"]
    min_x, min_y, max_x, max_y = poly.bounds
    left = max(int(min_x) - margin, 0)
    top = max(int(min_y) - margin, 0)
    width = int(max_x - min_x) + 2 * margin
    height = int(max_y - min_y) + 2 * margin
    patch = slide.read_region((left, top), 0, (width, height)).convert("RGB")
    return patch, (left, top, width, height)
