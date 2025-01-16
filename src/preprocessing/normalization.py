import os
import glob
import subprocess

def normalize_dataset_with_matlab(case_name, output_folder):
    """
    Normaliza las imágenes del dataset utilizando un script de MATLAB.

    Parameters:
        case_name (str): Nombre del caso a procesar.
        output_folder (str): Carpeta donde se encuentran las imágenes del caso.
    """
    # Definir las rutas de entrada y salida
    input_path = os.path.join(output_folder, case_name, '*.png')
    output_path = os.path.join(output_folder, 'Dataset_Normalized', case_name)

    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Obtener los archivos en ambas carpetas
    input_files = glob.glob(input_path)
    output_files = glob.glob(os.path.join(output_path, "*.png"))

    # Obtener los nombres base de los archivos
    input_files = [os.path.basename(file) for file in input_files]
    output_files = [os.path.basename(file) for file in output_files]

    # Verificar si los archivos coinciden
    if set(input_files) == set(output_files):
        print(f"Los archivos coinciden para el caso {case_name}.")
        return

    print(f"Normalizando imágenes para el caso {case_name}...")

    # Construir el comando para MATLAB
    command = (
        f"addpath(genpath('src/preprocessing/color_normalization_matlab/')); "
        f"NormalizacionColor_registro('{input_path}', '{output_path}'); exit;"
    )

    # Llamar a MATLAB con subprocess.call
    try:
        subprocess.call(
            [
                'matlab',
                '-nodisplay',
                '-nosplash',
                '-r',
                command
            ]
        )
        print(f"Normalización completada para el caso {case_name}.")
    except Exception as e:
        print(f"Error al ejecutar MATLAB para {case_name}: {e}")


