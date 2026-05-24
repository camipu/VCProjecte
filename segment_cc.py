import cv2
import numpy as np
import os
import argparse


def segmentar_components(roi, min_area=50):
    """
    Etiqueta components connexes i retorna les bounding boxes ordenades per x.
    """
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    components = []
    for i in range(1, num_labels):  # 0 és el fons
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], \
                            stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and h > w * 0.6:
            components.append((x, y, w, h, area))

    # Ordenar d'esquerra a dreta
    components.sort(key=lambda c: c[0])
    return thresh, components


def dibuixar_components(roi, components):
    out = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if len(roi.shape) == 2 else roi.copy()
    for i, (x, y, w, h, area) in enumerate(components):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(out, str(i), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    return out


def pipeline_segmentacio(directori_entrada="resultats", directori_sortida="resultats_cc", min_area=50):
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    subcarpetes = [d for d in os.listdir(directori_entrada)
                   if os.path.isdir(os.path.join(directori_entrada, d))]

    print(f"Processant {len(subcarpetes)} ROIs...\n")

    for nom in sorted(subcarpetes):
        roi_path = os.path.join(directori_entrada, nom, "07_roi_matricula_retallada.jpg")
        if not os.path.exists(roi_path):
            continue

        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            continue

        carpeta_out = os.path.join(directori_sortida, nom)
        os.makedirs(carpeta_out, exist_ok=True)

        thresh, components = segmentar_components(roi, min_area)
        cv2.imwrite(os.path.join(carpeta_out, "01_binaria.jpg"), thresh)

        out = dibuixar_components(thresh, components)
        cv2.imwrite(os.path.join(carpeta_out, "02_components.jpg"), out)

        print(f"  [{nom}] {len(components)} components detectades")
        for i, (x, y, w, h, area) in enumerate(components):
            print(f"         #{i}  x={x} y={y} w={w} h={h} area={area}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--entrada', default='resultats')
    parser.add_argument('--sortida', default='resultats_cc')
    parser.add_argument('--min_area', type=int, default=50)
    args = parser.parse_args()
    pipeline_segmentacio(args.entrada, args.sortida, args.min_area)
