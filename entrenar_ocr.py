import cv2
import numpy as np
import os

DATASET_LETRAS = "LetrasGeneradas"  
MODELO_SALIDA = "modelo_knn.xml"
MAPEO_SALIDA = "mapeo_clases.txt"
IMG_WIDTH, IMG_HEIGHT = 20, 30    

def extraer_caracteristicas_avanzadas(img_bin, aspect_ratio=1.0):
    """
    Extrae un descriptor robusto combinando la imagen con características 
    geométricas que diferencian letras conflictivas como la 'A' y el '4'.
    """
    # 1. Base del descriptor: los píxeles planos
    pixeles_planos = img_bin.flatten() / 255.0  # Normalizado entre 0 y 1
    
    # 2. Proyecciones horizontales y verticales (Suma de píxeles por filas y columnas)
    # Esto ayuda a detectar dónde se concentra la masa (el '4' suele tener más peso abajo a la derecha)
    proj_horiz = np.sum(img_bin, axis=1) / (255.0 * IMG_WIDTH)
    proj_vert = np.sum(img_bin, axis=0) / (255.0 * IMG_HEIGHT)
    
    # 3. Dividir la imagen en 4 cuadrantes y calcular la densidad de cada uno
    h_mitad, w_mitad = IMG_HEIGHT // 2, IMG_WIDTH // 2
    c1 = np.sum(img_bin[0:h_mitad, 0:w_mitad])
    c2 = np.sum(img_bin[0:h_mitad, w_mitad:IMG_WIDTH])
    c3 = np.sum(img_bin[h_mitad:IMG_HEIGHT, 0:w_mitad])
    c4 = np.sum(img_bin[h_mitad:IMG_HEIGHT, w_mitad:IMG_WIDTH])
    total_pixeles = np.sum(img_bin) if np.sum(img_bin) > 0 else 1.0
    densidades = np.array([c1, c2, c3, c4]) / total_pixeles
    
    # 4. Concatenar todo en un único vector de características súper robusto
    descriptor = np.concatenate([pixeles_planos, proj_horiz, proj_vert, densidades, [aspect_ratio]])
    return descriptor.astype(np.float32)

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

    print(f"Leyendo y extrayendo descriptores estructurales desde '{DATASET_LETRAS}'...")
    cont_antiguas = 0
    cont_nuevas = 0

    for clase in clases:
        carpeta_clase = os.path.join(DATASET_LETRAS, clase)
        imagenes = [f for f in os.listdir(carpeta_clase) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        
        idx_etiqueta = mapeo_clases[clase]
        
        for nom_img in imagenes:
            img_path = os.path.join(carpeta_clase, nom_img)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Calcular relación de aspecto antes de redimensionar (Ancho / Alto)
            h_orig, w_orig = img.shape[:2]
            aspect_ratio = float(w_orig) / float(h_orig) if h_orig > 0 else 1.0
            
            # Redimensionar y binarizar
            img_res = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            _, img_bin = cv2.threshold(img_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Corrección inteligente de fondo
            borde_superior = img_bin[0, :]
            if np.sum(borde_superior == 255) > (IMG_WIDTH / 2):
                img_bin = cv2.bitwise_not(img_bin)
                cont_antiguas += 1
            else:
                cont_nuevas += 1
            
            # EXTRAER DESCRIPTOR COMPUESTO
            descriptor = extraer_caracteristicas_avanzadas(img_bin, aspect_ratio)
            
            X_train.append(descriptor)
            y_train.append(idx_etiqueta)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    print(f"\n[INFO] Dataset estructurado:")
    print(f"       - Total de muestras cargadas: {len(X_train)}")
    print(f"       - Tamaño del nuevo descriptor por letra: {X_train.shape[1]} características")
    
    print("\nEntrenando clasificador KNN con descriptores geométricos...")
    knn = cv2.ml.KNearest_create()
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    
    knn.save(MODELO_SALIDA)
    print(f"¡Éxito! Modelo avanzado guardado en '{MODELO_SALIDA}'.")

if __name__ == "__main__":
    cargar_y_entrenar()