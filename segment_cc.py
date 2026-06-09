import cv2
import numpy as np
import os
import argparse
from ocr_utils import IMG_WIDTH, IMG_HEIGHT, extraer_caracteristicas, segmentar_caracteres


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
    Segmenta i reconeix els caràcters d'una ROI de matrícula.
    Usa ocr_utils per garantir descriptor idèntic al entrenament.
    """
    thresh, components = segmentar_caracteres(roi, min_area)
    out_visual      = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    texto_matricula = ""

    for x, y, w, h, area in components:
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        caracter = thresh[y:y+h, x:x+w]
        caracter = cv2.copyMakeBorder(caracter, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        caracter = cv2.resize(caracter, (IMG_WIDTH, IMG_HEIGHT))

        muestra        = extraer_caracteristicas(caracter, aspect_ratio).reshape(1, -1)
        _, res, _, _   = knn_model.findNearest(muestra, k=3)
        letra          = mapeo_dic[int(res[0][0])]
        texto_matricula += letra

        cv2.rectangle(out_visual, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(out_visual, letra, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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