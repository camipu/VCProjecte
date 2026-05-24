import cv2
import numpy as np
import os
import argparse


def alinear(roi):
    """Corregeix la inclinació de la ROI usant minAreaRect sobre el contorn principal."""
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]

    # minAreaRect retorna angles entre -90 i 0; ajustem perquè sigui proper a 0
    if angle < -45:
        angle += 90

    h, w = roi.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def retallar_marges(roi):
    """Elimina els marges blancs/negres al voltant dels caràcters."""
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return roi
    x, y, w, h = cv2.boundingRect(coords)
    return roi[y:y+h, x:x+w]


def trobar_separacions(roi, min_gap=3):
    """
    Perfil de projecció vertical: suma píxels foscos per columna.
    Retorna les coordenades x on hi ha valleys (separació entre caràcters).
    """
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    perfil = np.sum(thresh, axis=0) / 255

    llindar = perfil.max() * 0.1
    en_valley = False
    valleys = []
    inici = 0

    for x, val in enumerate(perfil):
        if val <= llindar and not en_valley:
            en_valley = True
            inici = x
        elif val > llindar and en_valley:
            en_valley = False
            amplada = x - inici
            if amplada >= min_gap:
                valleys.append((inici, x))

    return perfil, valleys


def dibuixar_separacions(roi_color, valleys):
    """Dibuixa línies verticals als gaps detectats."""
    out = cv2.cvtColor(roi_color, cv2.COLOR_GRAY2BGR) if len(roi_color.shape) == 2 else roi_color.copy()
    for x0, x1 in valleys:
        cx = (x0 + x1) // 2
        cv2.line(out, (cx, 0), (cx, out.shape[0]), (0, 0, 255), 1)
    return out


def pipeline_alineament(directori_entrada="resultats", directori_sortida="resultats_alineats"):
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    subcarpetes = [d for d in os.listdir(directori_entrada)
                   if os.path.isdir(os.path.join(directori_entrada, d))]

    print(f"Processant {len(subcarpetes)} ROIs...\n")

    for nom in sorted(subcarpetes):
        roi_path = os.path.join(directori_entrada, nom, "07_roi_matricula_retallada.jpg")
        if not os.path.exists(roi_path):
            print(f"  [{nom}] ROI no trobada, saltem.")
            continue

        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            continue

        carpeta_out = os.path.join(directori_sortida, nom)
        os.makedirs(carpeta_out, exist_ok=True)

        # PAS 1: Alinear
        alineada = alinear(roi)
        cv2.imwrite(os.path.join(carpeta_out, "01_alineada.jpg"), alineada)

        # PAS 2: Retallar marges
        retallada = retallar_marges(alineada)
        cv2.imwrite(os.path.join(carpeta_out, "02_sense_marges.jpg"), retallada)

        # PAS 3: Binaritzar
        _, binaria = cv2.threshold(retallada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(carpeta_out, "03_binaria.jpg"), binaria)

        # PAS 4: Separació de caràcters sobre la imatge binària
        perfil, valleys = trobar_separacions(retallada)
        out = dibuixar_separacions(binaria, valleys)
        cv2.imwrite(os.path.join(carpeta_out, "04_separacions.jpg"), out)

        print(f"  [{nom}] {len(valleys)} separacions detectades: {valleys}")

    print(f"\nResultats desats a '{directori_sortida}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--entrada', default='resultats')
    parser.add_argument('--sortida', default='resultats_alineats')
    args = parser.parse_args()
    pipeline_alineament(args.entrada, args.sortida)
