import cv2
import numpy as np
import os
import argparse

# Configuración fija del tamaño (debe coincidir exactamente con el entrenamiento)
IMG_WIDTH, IMG_HEIGHT = 20, 30


def cargar_ocr(modelo_path="modelo_knn.xml", mapeo_path="mapeo_clases.txt"):
    """
    Carga el modelo KNN guardado y su correspondiente mapa de traducción de forma segura.
    """
    if not os.path.exists(modelo_path) or not os.path.exists(mapeo_path):
        raise FileNotFoundError(
            f"Falta '{modelo_path}' o '{mapeo_path}'. Ejecuta primero entrenar_ocr.py"
        )
        
    knn = cv2.ml.KNearest_load(modelo_path)
    
    mapeo = {}
    with open(mapeo_path, "r") as f:
        for line in f:
            if ":" in line:
                clase, idx = line.strip().split(':')
                mapeo[int(idx)] = clase
    return knn, mapeo


def segmentar_y_reconocer_components(roi, knn_model, mapeo_dic, min_area=50):
    """
    Etiqueta componentes connexes, filtra ruido, ordena de izquierda a derecha
    y clasifica cada dígito inmediatamente después mediante el OCR.
    """
    # 1. Binarización e inversión (Letras blancas, fondo negro)
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Limpieza de imperfecciones o pequeños puntos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Etiquetado por componentes conectadas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    components = []
    for i in range(1, num_labels):  # El 0 se salta porque es el fondo negro
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtros geométricos para certificar que la isla tiene proporciones de letra
        if area >= min_area and h > w * 0.6:
            components.append((x, y, w, h, area))

    # 4. Ordenar rigurosamente de izquierda a derecha (coordenada X) para leer en orden
    components.sort(key=lambda c: c[0])

    # Preparar lienzo de salida en color para pintar los diagnósticos
    out_visual = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    texto_matricula = ""

    # 5. PASAR EL OCR A CADA DÍGITO DETECTADO
    for i, (x, y, w, h, area) in enumerate(components):
        caracter_recortado = thresh[y:y+h, x:x+w]
        
        # Calcular relación de aspecto real de la matrícula antes del resize
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        
        # Padding de seguridad
        caracter_recortado = cv2.copyMakeBorder(
            caracter_recortado, 
            top=2, bottom=2, left=2, right=2, 
            borderType=cv2.BORDER_CONSTANT, 
            value=0
        )
        
        caracter_res = cv2.resize(caracter_recortado, (IMG_WIDTH, IMG_HEIGHT))
        
        # --- AQUÍ USAMOS EL NUEVO DESCRIPTOR ROBUSTO ---
        # Volvemos a calcular los cuadrantes y proyecciones para este carácter concreto
        pixeles_planos = caracter_res.flatten() / 255.0
        proj_horiz = np.sum(caracter_res, axis=1) / (255.0 * IMG_WIDTH)
        proj_vert = np.sum(caracter_res, axis=0) / (255.0 * IMG_HEIGHT)
        
        h_mitad, w_mitad = IMG_HEIGHT // 2, IMG_WIDTH // 2
        c1 = np.sum(caracter_res[0:h_mitad, 0:w_mitad])
        c2 = np.sum(caracter_res[0:h_mitad, w_mitad:IMG_WIDTH])
        c3 = np.sum(caracter_res[h_mitad:IMG_HEIGHT, 0:w_mitad])
        c4 = np.sum(caracter_res[h_mitad:IMG_HEIGHT, w_mitad:IMG_WIDTH])
        total_p = np.sum(caracter_res) if np.sum(caracter_res) > 0 else 1.0
        densidades = np.array([c1, c2, c3, c4]) / total_p
        
        # Generar vector muestra idéntico al entrenamiento
        muestra = np.concatenate([pixeles_planos, proj_horiz, proj_vert, densidades, [aspect_ratio]])
        muestra = muestra.astype(np.float32).reshape(1, -1)
        
        # Predicción KNN buscando los 3 vecinos más próximos
        _, resultado, _, _ = knn_model.findNearest(muestra, k=3)
        letra_predicha = mapeo_dic[int(resultado[0][0])]
        
        texto_matricula += letra_predicha

        # Dibujar resultados
        cv2.rectangle(out_visual, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(out_visual, letra_predicha, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return thresh, out_visual, texto_matricula, components


def pipeline_segmentacio_y_ocr(directori_entrada="resultats", directori_sortida="resultats_ocr", min_area=50):
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    # Intentar cargar el clasificador entrenado e invertido
    try:
        knn_model, mapeo_dic = cargar_ocr()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return

    subcarpetes = [d for d in os.listdir(directori_entrada)
                   if os.path.isdir(os.path.join(directori_entrada, d))]

    print(f"Processant {len(subcarpetes)} ROIs amb segmentació i reconeixement OCR...\n")

    for nom in sorted(subcarpetes):
        roi_path = os.path.join(directori_entrada, nom, "07_roi_matricula_retallada.jpg")
        if not os.path.exists(roi_path):
            continue

        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            continue

        carpeta_out = os.path.join(directori_sortida, nom)
        os.makedirs(carpeta_out, exist_ok=True)

        # Llamar a la lógica unificada
        thresh, out_visual, texto, components = segmentar_y_reconocer_components(
            roi, knn_model, mapeo_dic, min_area
        )

        # Guardar las imágenes resultantes de diagnóstico
        cv2.imwrite(os.path.join(carpeta_out, "01_binaria.jpg"), thresh)
        cv2.imwrite(os.path.join(carpeta_out, "02_ocr_output.jpg"), out_visual)

        # -----------------------------------------------------------------
        # ¡NUEVO PASO!: Guardar la matrícula en un fichero de texto (.txt)
        # -----------------------------------------------------------------
        txt_output_path = os.path.join(carpeta_out, "matricula_ocr.txt")
        with open(txt_output_path, "w") as f_txt:
            f_txt.write(f"{texto}\n")

        # Mostrar resultados detallados por consola
        print(f"  [{nom}] Matrícula llegida: '{texto}' -> Guardada en: {txt_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--entrada', default='resultats', 
                        help='Carpeta que contiene los recortes originales de las matrículas')
    parser.add_argument('--sortida', default='resultats_ocr', 
                        help='Carpeta donde se guardarán los resultados leídos')
    parser.add_argument('--min_area', type=int, default=50, 
                        help='Área mínima de píxeles para considerar un carácter válido')
    args = parser.parse_args()
    
    pipeline_segmentacio_y_ocr(args.entrada, args.sortida, args.min_area)