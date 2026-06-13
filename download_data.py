import os
import urllib.request

def download_dataset():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    data_dir = "data"
    output_path = os.path.join(data_dir, "results.csv")
    
    # Crear directorio si no existe
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directorio '{data_dir}' creado.")
        
    print(f"Descargando dataset desde:\n{url}\n")
    try:
        # Descargar el archivo
        urllib.request.urlretrieve(url, output_path)
        print(f"¡Descarga exitosa! Archivo guardado en: {output_path}")
        # Mostrar tamaño del archivo
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Tamaño del archivo: {file_size:.2f} MB")
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")

if __name__ == "__main__":
    download_dataset()
