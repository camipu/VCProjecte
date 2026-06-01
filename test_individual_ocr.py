import cv2
import numpy as np
import os

# Configuración idéntica al entrenamiento
IMG_WIDTH, IMG_HEIGHT = 20, 30
MODELO_PATH = "modelo_knn.xml"
MAPEO_PATH = "mapeo_clases.txt"
IMAGEN_TEST = "ProvaOCR.jpg"  # Asegúrate de que el nombre coincida


def cargar_ocr():
    if not os.path.exists(MODELO_PATH) or not os.path.exists(MAPEO_PATH):
        raise FileNotFoundError(
            "No se encuentra el modelo. Ejecuta primero entrenar_ocr.py"
        )

    knn = cv2.ml.KNearest_load(MODELO_PATH)
    mapeo = {}
    with open(MAPEO_PATH, "r") as f:
        for line in f:
            if ":" in line:
                clase, idx = line.strip().split(":")
                mapeo[int(idx)] = clase
    return knn, mapeo


def test_imagen_completa():
    # 1. Cargar el modelo
    try:
        knn_model, mapeo_dic = cargar_ocr()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 2. Leer la imagen de prueba en escala de grises
    if not os.path.exists(IMAGEN_TEST):
        print(f"[ERROR] No se encuentra el archivo de imagen: {IMAGEN_TEST}")
        return

    roi = cv2.imread(IMAGEN_TEST, cv2.IMREAD_GRAYSCALE)

    # 3. Binarización (Fondo blanco -> Negro, Letras negras -> Blancas)
    _, thresh = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Limpieza básica por si hay ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. Detectar caracteres usando componentes conectadas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    components = []
    # Filtro de área más bajo (min_area=20) por si los caracteres de la plantilla son pequeños
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 20:
            components.append((x, y, w, h))

    # Ordenar de izquierda a derecha (y opcionalmente por filas si están en cascada)
    # De momento, ordenamos de izquierda a derecha por simplicidad
    components.sort(key=lambda c: (c[1] // 20, c[0]))

    # Lienzo para pintar el resultado visual
    out_visual = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    texto_detectado = ""

    print(f"Detectados {len(components)} caracteres en la hoja de prueba.\n")

    # 5. Evaluar cada carácter
    for x, y, w, h in components:
        caracter_recortado = thresh[y : y + h, x : x + w]

        # Aplicar el mismo Padding de 2 píxeles que pusimos en el script principal
        caracter_recortado = cv2.copyMakeBorder(
            caracter_recortado,
            top=2,
            bottom=2,
            left=2,
            right=2,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        # Redimensionar y aplanar
        caracter_res = cv2.resize(caracter_recortado, (IMG_WIDTH, IMG_HEIGHT))
        muestra = caracter_res.flatten().astype(np.float32).reshape(1, -1)

        # Clasificar
        _, resultado, _, _ = knn_model.findNearest(muestra, k=3)
        letra_predicha = mapeo_dic[int(resultado[0][0])]

        texto_detectado += letra_predicha

        # Dibujar rectángulos y texto
        cv2.rectangle(out_visual, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            out_visual,
            letra_predicha,
            (x + 2, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # 6. Guardar y mostrar resultados
    cv2.imwrite("resultado_prova_ocr.jpg", out_visual)
    print(f"Texto reconocido en total:\n{texto_detectado}\n")
    print("¡Visualización guardada en 'resultado_prova_ocr.jpg'!")


if __name__ == "__main__":
    test_imagen_completa()