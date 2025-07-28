from src.preprocessing.functionPatchesExtraction import generate_Patches
from src.preprocessing.normalization import normalize_dataset_with_matlab
from src.segmentation.unet_segmenter import generate_WSI_unet
import os, json

def process_all_images_with_pipeline(config):
    input_folder = os.path.join(config["input_path"], config["svs_subfolder"])
    output_folder = config["processed_path"]
    patchsize = config.get("patch_size", 512 * 3)
    batch_size = config.get("batch_size", 100)
    img_normalize = config.get("img_normalize", True)
    high_resol = config.get("high_resol", False)

    os.makedirs(output_folder, exist_ok=True)
    svs_files = [f for f in os.listdir(input_folder) if f.endswith(".svs")]

    for svs_file in svs_files:
        case_name = svs_file.replace(".svs", "")
        slide_path = os.path.join(input_folder, svs_file)
        output_im = svs_file.replace('.svs', '.jpc')
        output_xml_path = os.path.join(output_folder, f"{case_name}_annotations.xml")

        if os.path.exists(output_xml_path):
            print(f"{output_xml_path} ya existe. Saltando.")
            continue
        # generate_Patches(patchsize, batch_size, slide_path, outputIm=output_im,
        #                                            HighResol=high_resol,
        #                                            ImgNormalize=img_normalize, CaseName=case_name,
        #                                            OutputPath=output_folder)



        # if img_normalize:
        #     normalize_dataset_with_matlab(case_name, output_folder)

        generate_WSI_unet(config, slide_path, case_name, output_im, output_folder)

def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    process_all_images_with_pipeline(config)

if __name__ == "__main__":
    main()
