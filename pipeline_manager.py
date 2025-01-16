import os
import json
import subprocess
from datetime import datetime

# Configuración inicial
def create_config():
    """Crea un archivo config.json si no existe."""
    config = {
        "input_path": "input_data",
        "processed_path": "processed_data",
        "final_output": "final_output",
        "svs_subfolder": "images",
        "xml_subfolder": "annotations",
        "suffix_corrected": "_corrected.xml",
        "suffix_completed": "_completed.xml",
        "suffix_final": "_corrected_final.xml",
        "log_path": "logs",
    }

    if not os.path.exists("scripts/GlandSegmentation/config.json"):
        with open("scripts/GlandSegmentation/config.json", "w") as config_file:
            json.dump(config, config_file, indent=4)
        print("Archivo config.json creado.")
    else:
        print("Archivo config.json ya existe.")

# Crear estructura de carpetas
def create_folder_structure(config):
    """Crea la estructura de carpetas basada en config.json."""
    required_folders = [
        config["input_path"],
        os.path.join(config["input_path"], config["svs_subfolder"]),
        os.path.join(config["input_path"], config["xml_subfolder"]),
        config["processed_path"],
        config["final_output"],
        config["log_path"]
    ]
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)
    print("Estructura de carpetas creada.")

# Registro de logs
def log_message(message, log_path):
    """Registra mensajes en un archivo de log."""
    log_file = os.path.join(log_path, f"pipeline_log_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(log_file, "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Ejecución de scripts
def run_script(script_name, config, log_path):
    """Ejecuta un script y registra el resultado."""
    log_message(f"Iniciando {script_name}...", log_path)
    try:
        subprocess.run(["python", f"scripts/GlandSegmentation/{script_name}"], check=True)
        log_message(f"{script_name} ejecutado correctamente.", log_path)
    except subprocess.CalledProcessError as e:
        log_message(f"Error ejecutando {script_name}: {e}", log_path)

# Verificación de dependencias
def verify_files(config):
    """Verifica que las entradas necesarias estén disponibles."""
    svs_path = os.path.join(config["input_path"], config["svs_subfolder"])
    xml_path = os.path.join(config["input_path"], config["xml_subfolder"])

    svs_files = [f for f in os.listdir(svs_path) if f.endswith(".svs")]
    xml_files = [f for f in os.listdir(xml_path) if f.endswith(".xml")]

    if not svs_files:
        print(f"No se encontraron archivos SVS en {svs_path}.")
        return False
    if not xml_files:
        print(f"No se encontraron archivos XML en {xml_path}.")
        return False
    print(f"{len(svs_files)} archivos SVS y {len(xml_files)} archivos XML encontrados.")
    return True

# Pipeline principal
def main():
    # Configuración inicial
    create_config()
    with open("config.json") as config_file:
        config = json.load(config_file)

    create_folder_structure(config)

    # Verificar dependencias
    if not verify_files(config):
        print("Archivos requeridos faltantes. Por favor, verifica las entradas.")
        return

    # Ejecutar pipeline
    scripts = [
        "process_svs_and_generate_annotations.py",
        "clean_binary_mask_with_annotations.py",
        "refined_SAM_segmentation.py",
        "final_clean.py",
        "simplify_XML.py"
    ]
    for script in scripts:
        run_script(script, config, config["log_path"])

    log_message("Pipeline completado.", config["log_path"])

if __name__ == "__main__":
    main()
