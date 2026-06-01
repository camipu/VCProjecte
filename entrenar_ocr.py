import cv2
import numpy as np
import os

# Configuración de rutas según tu estructura actual
DATASET_LETRAS = "LetrasGeneradas"  
MODELO_SALIDA = "modelo_knn.xml"
MAPEO_SALIDA = "mapeo_clases.txt"
IMG_WIDTH, IMG_HEIGHT = 20, 30    # Tamaño homogéneo estándar para el OCR

def cargar_y_entrenar():
    if not os.path.exists(DATASET_LETRAS):
        print(f"Error: No se encuentra la carpeta {DATASET_LETRAS}")
        return

    X_train = []
    y_train = []
    
    clases = sorted([d for d in os.listdir(DATASET_LETRAS) if os.path.isdir(os.path.join(DATASET_LETRAS, d))])
    mapeo_clases = {clase: i for i, clase in enumerate(clases)}
    
    with open(MAPEO_SALIDA, "w") as f:
        for clase, idx in mapeo_clases.items():
            f.write(f"{clase}:{idx}\n")

    print(f"Leyendo y adaptando caracteres desde '{DATASET_LETRAS}'...")
    for clase in clases:
        carpeta_clase = os.path.join(DATASET_LETRAS, clase)
        imagenes = [f for f in os.listdir(carpeta_clase) if f.lower().endswith(('.tif', '.png', '.jpg'))]
        
        idx_etiqueta = mapeo_clases[clase]
        
        for nom_img in imagenes:
            img_path = os.path.join(carpeta_clase, nom_img)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 1. Redimensionar al tamaño del pipeline
            img_res = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            # 2. ¡EL CAMBIO CLAVE!: Forzar binarización INVERTIDA (THRESH_BINARY_INV)
            # Esto hace que las letras del dataset pasen a ser BLANCAS sobre fondo NEGRO,
            # calcando el comportamiento de 'thresh' en segment_cc.py
            _, img_bin = cv2.threshold(img_res, 127, 255, cv2.THRESH_BINARY_INV)
            
            # 3. Convertir a vector plano
            descriptor = img_bin.flatten()
            
            X_train.append(descriptor)
            y_train.append(idx_etiqueta)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    print(f"Dataset cargado y emparejado. Total de muestras: {len(X_train)}")
    
    # 4. Entrenar el clasificador KNN
    print("Re-entrenando clasificador KNN de OpenCV...")
    knn = cv2.ml.KNearest_create()
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    
    # Guardar modelo mejorado
    knn.save(MODELO_SALIDA)
    print(f"¡Éxito! Modelo adaptado guardado en '{MODELO_SALIDA}'.")

if __name__ == "__main__":
    cargar_y_entrenar()