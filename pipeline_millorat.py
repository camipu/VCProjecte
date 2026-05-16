import cv2
import numpy as np
import os
import sys


# ---------------------------------------------------------------------------
# PREPROCESSAMENT
# ---------------------------------------------------------------------------

def preprocessar(img_bgr, target_w=800):
    h, w = img_bgr.shape[:2]
    scale = target_w / w
    img = cv2.resize(img_bgr, (target_w, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return img


# ---------------------------------------------------------------------------
# GENERACIÓ DE CANDIDATS
# ---------------------------------------------------------------------------

def extreure_candidats(gris):
    """
    Genera regions candidates usant morfologia + gradient Sobel X.
    Kernel (25×3) en lloc de (25×5): menys inflat vertical, bounding boxes
    més ajustats a l'alçada real de la matrícula → millora el IoU.
    """
    filtered = cv2.bilateralFilter(gris, d=11, sigmaColor=17, sigmaSpace=17)

    sobelx = cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3)
    abs_sobel = cv2.convertScaleAbs(sobelx)

    _, thresh = cv2.threshold(abs_sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Kernel (25×3): amplada suficient per unir lletres, alçada mínima
    # per no inflar el bounding box més de 1-2px per damunt/davall del text
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh, kernel_h, iterations=2)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)

    contorns, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contorns, abs_sobel


# ---------------------------------------------------------------------------
# PUNTUACIÓ DE CANDIDATS
# ---------------------------------------------------------------------------

def puntuar_candidat(x, y, w, h, sobel_img, gris_img, img_h, img_w):
    """
    Retorna un score [0, 1] per a un candidat.

    Criteris (ponderats):

    1. Ratio score (40%): Decay gaussià centrat a 4.35.
       Molt més discriminatiu que 1/(1+|diff|): penalitza fortament qualsevol
       candidat amb ratio allunyat del valor EU estàndard (520×112 mm = 4.64,
       però el dataset mostra ~4.35 de mitjana real mesurada).

    2. Densitat de vores Sobel (30%): presència de text = molts gradients verticals.

    3. Brillantor (20%): matrícules EU = fons blanc/groc.
       Les zones de carrosseria fosca, asfalt i sostre puntuen baix.

    4. Àrea relativa (10%): filtre suau de mida.
    """
    area = w * h
    img_area = img_h * img_w

    # --- 1. Ratio score (Gaussià) ---
    ratio = w / h
    ratio_ideal = 4.35
    ratio_score = np.exp(-((ratio - ratio_ideal) ** 2) / (2 * 1.2 ** 2))

    # --- 2. Densitat de vores ---
    roi_sobel = sobel_img[y:y + h, x:x + w]
    edge_density = np.sum(roi_sobel > 50) / (area + 1)
    edge_score = min(edge_density / 0.10, 1.0)

    # --- 3. Brillantor del ROI ---
    roi_gris = gris_img[y:y + h, x:x + w]
    mean_bright = np.mean(roi_gris) / 255.0
    bright_score = min(mean_bright / 0.45, 1.0)

    # --- 4. Àrea relativa ---
    rel_area = area / img_area
    if rel_area < 0.004 or rel_area > 0.15:
        area_score = 0.0
    else:
        area_score = max(0.0, 1.0 - abs(rel_area - 0.03) / 0.06)

    return ratio_score * 0.40 + edge_score * 0.30 + bright_score * 0.20 + area_score * 0.10


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def detectar_matricula(img_bgr):
    """
    Detecta la matrícula en una imatge BGR.
    Retorna (x, y, w, h, score) en coordenades de la imatge original,
    o None si no es troba cap candidat prou bo.
    """
    h_orig, w_orig = img_bgr.shape[:2]

    img_proc = preprocessar(img_bgr)
    h_proc, w_proc = img_proc.shape[:2]

    sx = w_orig / w_proc
    sy = h_orig / h_proc

    gris = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    candidats, sobel_img = extreure_candidats(gris)

    # Marge de vora: exclou regions que toquen els límits de la imatge.
    # Causa principal dels falsos positius (x=0, y=0) de la versió anterior.
    margin_x = max(4, int(0.015 * w_proc))
    margin_y = max(4, int(0.015 * h_proc))

    millor_score = 0.0
    millor = None

    for c in candidats:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue

        # Filtre de vora: cap region que toqui el border de la imatge
        if (x < margin_x or y < margin_y or
                (x + w) > w_proc - margin_x or
                (y + h) > h_proc - margin_y):
            continue

        ratio = w / h
        if not (2.5 < ratio < 7.0):
            continue

        rel_w = w / w_proc
        rel_h = h / h_proc
        if rel_w < 0.08 or rel_w > 0.55:
            continue
        # Màxim reduït al 15%: elimina regions massa altes
        if rel_h < 0.02 or rel_h > 0.15:
            continue

        score = puntuar_candidat(x, y, w, h, sobel_img, gris, h_proc, w_proc)
        if score > millor_score:
            millor_score = score
            millor = (x, y, w, h)

    if millor is None or millor_score < 0.10:
        return None

    x, y, w, h = millor
    return int(x * sx), int(y * sy), int(w * sx), int(h * sy), millor_score


# ---------------------------------------------------------------------------
# PROCESSAT EN LOT
# ---------------------------------------------------------------------------

def processar_directori(directori_entrada="images", directori_sortida="resultats_millorat"):
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    arxius = sorted([f for f in os.listdir(directori_entrada) if f.lower().endswith(extensions)])

    if not arxius:
        print(f"No s'han trobat imatges a '{directori_entrada}'")
        return

    print(f"Processant {len(arxius)} imatges...\n")
    detectades = 0

    for nom in arxius:
        img = cv2.imread(os.path.join(directori_entrada, nom))
        if img is None:
            print(f"  [SKIP] {nom}")
            continue

        resultat = detectar_matricula(img)

        if resultat:
            x, y, w, h, score = resultat
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, f"Score:{score:.2f} R:{w/h:.1f}", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detectades += 1
            print(f"  [OK] {nom}  score={score:.3f}  bbox=({x},{y},{w},{h})  ratio={w/h:.2f}")
        else:
            cv2.putText(img, "SENSE DETECCIO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print(f"  [ ] {nom}  — sense detecció")

        cv2.imwrite(os.path.join(directori_sortida, f"res_{nom}"), img)

    print(f"\nDetectades: {detectades}/{len(arxius)} ({100*detectades/len(arxius):.1f}%)")
    print(f"Resultats a: {directori_sortida}/")


if __name__ == "__main__":
    dir_in  = sys.argv[1] if len(sys.argv) > 1 else "images"
    dir_out = sys.argv[2] if len(sys.argv) > 2 else "resultats_millorat"
    processar_directori(dir_in, dir_out)
