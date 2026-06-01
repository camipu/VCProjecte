import os
import shutil

def ordenar_letras():
    # Conseguir la lista de archivos en el directorio actual
    archivos = os.listdir('.')
    
    # Extensiones válidas de imágenes que queremos mover
    extensiones_validas = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
    
    print("Iniciando la reorganización del dataset...")
    cont_movidos = 0

    for archivo in archivos:
        # Ignorar carpetas y archivos que no sean imágenes (como Thumbs.db o este propio script)
        if not os.path.isfile(archivo) or not archivo.lower().endswith(extensiones_validas):
            continue
            
        # El nombre del archivo sigue el patrón: "X_1.tif". El primer caracter antes del '_' es la clase.
        # Dividimos el nombre por el primer guion bajo
        partes = archivo.split('_')
        if len(partes) < 2:
            print(f" -> Saltando archivo sin patrón esperado: {archivo}")
            continue
            
        nombre_carpeta = partes[0]  # Esto será '0', '4', 'D', 'Y', etc.
        
        # Crear la carpeta para el carácter si no existe todavía
        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)
            
        # Mover el archivo dentro de su respectiva carpeta
        destino = os.path.join(nombre_carpeta, archivo)
        shutil.move(archivo, destino)
        cont_movidos += 1

    print(f"\n¡Proceso completado! Se han movido {cont_movidos} archivos a sus carpetas correspondientes.")

if __name__ == "__main__":
    ordenar_letras()