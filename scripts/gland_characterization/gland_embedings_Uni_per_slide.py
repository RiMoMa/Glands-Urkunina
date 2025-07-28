import os
import openslide
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.affinity import translate
import numpy as np
import cv2
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- Configuración ---
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)  # Login con Hugging Face

# Definir parámetros de UNI2-h
timm_kwargs = {
    'img_size': 224,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667 * 2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True
}

# Cargar modelo UNI2-h
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
def parse_gland_polygons(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    glands = []
    for ann in root.findall(".//Annotation"):
        name = ann.attrib.get("Name", "")
        coords_tag = ann.find("Coordinates")
        if coords_tag is None: continue
        coords = [
            (float(c.attrib["X"].replace(",", ".")), float(c.attrib["Y"].replace(",", ".")))
            for c in coords_tag.findall("Coordinate")
        ]

        if len(coords) >= 3:
            glands.append((name, Polygon(coords)))
    return glands

def extract_patch(slide, polygon, level=0,PATCH_SIZE=224):
    minx, miny, maxx, maxy = polygon.bounds
    minx, miny = int(minx), int(miny)
    width, height = int(maxx - minx), int(maxy - miny)

    img = slide.read_region((minx, miny), level, (width, height)).convert("RGB")
    np_img = np.array(img)

    # Generar máscara en coordenadas locales
    local_poly = translate(polygon, xoff=-minx, yoff=-miny)
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(local_poly.exterior.coords, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # --- Blur extremo (fondo) ---
    blurred = cv2.GaussianBlur(np_img, (51, 51), sigmaX=50)

    # --- Combinación: fondo borroso, glándula original ---
    masked_rgb = np.zeros_like(np_img)
    for c in range(3):
        masked_rgb[..., c] = np.where(mask == 255, np_img[..., c], blurred[..., c])

    # --- Redimensionar ---
    resized = cv2.resize(masked_rgb, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)
    ## para mascara unicamente sin modificar el fondo
    # # Aplicar máscara
    # masked_img = cv2.bitwise_and(np_img, np_img, mask=mask)
    #
    # # Redimensionar a 224x224
    # resized = cv2.resize(masked_img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    # return Image.fromarray(resized)

def embed_patch(patches, batch_size=32):
    embeddings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #### un solo parche
    batch_patches = patches
    tensor_batch = torch.stack(
        [transform(batch_patches)]
    ).to(device)
    with torch.inference_mode():
        features = model(tensor_batch).cpu().numpy()

    # ## para multiples parches
    # for size, patch_list in patches.items():
    #     all_features = []
    #     n = len(patch_list)
    #     for i in range(0, n, batch_size):
    #         batch_patches = patch_list[i:i + batch_size]
    #         tensor_batch = torch.stack(
    #             [transform(Image.fromarray((patch * 255).astype(np.uint8))) for patch in batch_patches]
    #         ).to(device)
    #
    #         with torch.inference_mode():
    #             features = model(tensor_batch).cpu().numpy()
    #             all_features.append(features)
    #
    #     embeddings[size] = np.concatenate(all_features, axis=0)

    #return embeddings
    return features[0]
# --- Main ---
def procesar_wsi(xml_path, wsi_path, out_csv="embeddings.csv"):
    slide = openslide.OpenSlide(str(wsi_path))
    glands = parse_gland_polygons(xml_path)

    rows = []
    for name, poly in tqdm(glands):
        try:
            img_patch = extract_patch(slide, poly)
            emb = embed_patch(img_patch)
            rows.append({"gland_name": name, "embedding": emb})
        except Exception as e:
            print(f"[!] Error con {name}: {e}")

    df = pd.DataFrame(rows)
    df[["e"+str(i) for i in range(len(df.embedding.iloc[0]))]] = pd.DataFrame(df["embedding"].tolist())
    df.drop(columns="embedding").to_csv(out_csv, index=False)
    print(f"[✓] Guardado: {out_csv}")

# --- Ejecutar si se llama directamente ---
if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--xml", required=True, help="Archivo XML con glándulas")
    # parser.add_argument("--wsi", required=True, help="Archivo WSI (.svs, .tiff...)")
    # parser.add_argument("--out", default="embeddings.csv", help="Archivo de salida CSV")
    # args = parser.parse_args()
    #
    # procesar_wsi(args.xml, args.wsi, args.out)
    xml = '/home/ricardo/Glands-Urkunina/processed_data_gland_type/Glandulas_Tipo_xml/Andres/001_459_2021_he.xml'
    wsi = '/data/ricardo/dataMetaplasiaCompletaIncompleta/Slides/HE/001_459_2021_he.svs'
    out ='borra_embeddings.csv'
    procesar_wsi(xml, wsi, out)
