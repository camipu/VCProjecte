import cv2
import numpy as np
import os
from ocr_utils import IMG_WIDTH, IMG_HEIGHT, extraer_caracteristicas

DATASET_LETRAS = "LetrasGeneradas"
MODELO_SALIDA  = "modelo_knn.xml"
MAPEO_SALIDA   = "mapeo_clases.txt"


def augmentar_imagen(img_bin):
    """
    Genera 10 variants d'una imatge binària 20×30.
    Simula variacions de captura: rotació, gruix de traç i petits desplaçaments.
    """
    cx, cy   = IMG_WIDTH / 2, IMG_HEIGHT / 2
    variants = [img_bin]

    # Rotació lleu (±3° i ±6°)
    for angle in [-6, -3, 3, 6]:
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(img_bin, M, (IMG_WIDTH, IMG_HEIGHT),
                             flags=cv2.INTER_NEAREST, borderValue=0)
        variants.append(rot)

    # Erosió (traç més prim) i dilatació (traç més gruixut)
    k = np.ones((2, 2), np.uint8)
    variants.append(cv2.erode(img_bin,  k, iterations=1))
    variants.append(cv2.dilate(img_bin, k, iterations=1))

    # Translació lleu (±2 px horitz i vert)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M       = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img_bin, M, (IMG_WIDTH, IMG_HEIGHT), borderValue=0)
        variants.append(shifted)

    return variants  # 11 variants per imatge original


def cargar_y_entrenar():
    if not os.path.exists(DATASET_LETRAS):
        print(f"Error: No s'ha trobat '{DATASET_LETRAS}'")
        return

    clases = sorted(d for d in os.listdir(DATASET_LETRAS)
                    if os.path.isdir(os.path.join(DATASET_LETRAS, d)))
    mapeo  = {c: i for i, c in enumerate(clases)}

    with open(MAPEO_SALIDA, "w") as f:
        for c, i in mapeo.items():
            f.write(f"{c}:{i}\n")

    print(f"Classes: {len(clases)}  →  {' '.join(clases)}")

    X_train, y_train = [], []
    per_classe = {}

    print(f"Llegint '{DATASET_LETRAS}' i aplicant augmentació (×11)...\n")

    for clase in clases:
        carpeta = os.path.join(DATASET_LETRAS, clase)
        imgs    = [f for f in os.listdir(carpeta)
                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        idx     = mapeo[clase]
        count   = 0

        for nom in imgs:
            img = cv2.imread(os.path.join(carpeta, nom), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h_orig, w_orig = img.shape[:2]
            aspect_ratio   = float(w_orig) / float(h_orig) if h_orig > 0 else 1.0

            img_res = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            _, img_bin = cv2.threshold(img_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Correcció de fons: assegurar caràcter blanc sobre fons negre
            if np.sum(img_bin[0, :] == 255) > IMG_WIDTH / 2:
                img_bin = cv2.bitwise_not(img_bin)

            for variant in augmentar_imagen(img_bin):
                X_train.append(extraer_caracteristicas(variant, aspect_ratio))
                y_train.append(idx)
                count += 1

        per_classe[clase] = count

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    min_c = min(per_classe.values())
    max_c = max(per_classe.values())
    print(f"Dataset augmentat : {len(X_train)} mostres totals")
    print(f"Descriptor        : {X_train.shape[1]} característiques per mostra")
    print(f"Mostres per classe: min={min_c}  max={max_c}\n")

    print("Entrenant KNN...")
    knn = cv2.ml.KNearest_create()
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    knn.save(MODELO_SALIDA)
    print(f"Model guardat a '{MODELO_SALIDA}' ✓")
    print(f"Mapeo  guardat a '{MAPEO_SALIDA}' ✓")


if __name__ == "__main__":
    cargar_y_entrenar()
