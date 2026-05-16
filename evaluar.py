"""
Avaluació quantitativa de pipelines de detecció de matrícules.

Mètriques:
  - IoU (Intersection over Union): mesurem el solapament entre la
    detecció i el ground truth. El llindar estàndard de TP és IoU ≥ 0.5.
  - Precisió = TP / (TP + FP)
  - Recall    = TP / (TP + FN)
  - F1        = 2 * P * R / (P + R)
  - IoU mitjà (sobre els TP)

Ús:
  python3 evaluar.py                    # compara tots dos pipelines
  python3 evaluar.py --pipeline nou     # només el nou
  python3 evaluar.py --pipeline antic   # només morfo.py
"""

import cv2
import numpy as np
import os
import sys
import argparse


# ---------------------------------------------------------------------------
# LECTURA DE GROUND TRUTH
# ---------------------------------------------------------------------------

def llegir_anotacio(path_txt):
    """
    Format: filename<TAB>x<TAB>y<TAB>w<TAB>h<TAB>text
    Retorna (x, y, w, h) o None si l'arxiu no és vàlid.
    """
    try:
        with open(path_txt) as f:
            parts = f.readline().strip().split('\t')
        if len(parts) >= 5:
            return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    except Exception:
        pass
    return None


def carregar_ground_truth(directori_dataset):
    """
    Llegeix totes les anotacions d'un directori.
    Retorna dict: nom_imatge → (x, y, w, h)
    """
    gt = {}
    for f in os.listdir(directori_dataset):
        if not f.endswith('.txt'):
            continue
        nom_base = f.replace('.txt', '')
        # Busquem la imatge corresponent (pot ser .jpg, .png...)
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            img_path = os.path.join(directori_dataset, nom_base + ext)
            if os.path.exists(img_path):
                bbox = llegir_anotacio(os.path.join(directori_dataset, f))
                if bbox:
                    gt[img_path] = bbox
                break
    return gt


# ---------------------------------------------------------------------------
# MÈTRIQUES
# ---------------------------------------------------------------------------

def iou(box1, box2):
    """Intersection over Union entre dos bboxes en format (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    unio  = w1 * h1 + w2 * h2 - inter
    return inter / unio if unio > 0 else 0.0


# ---------------------------------------------------------------------------
# WRAPPERS DELS PIPELINES
# ---------------------------------------------------------------------------

def detectar_amb_nou_pipeline(img_bgr):
    """Crida pipeline_millorat.py i retorna (x, y, w, h) o None."""
    from pipeline_millorat import detectar_matricula
    res = detectar_matricula(img_bgr)
    if res is None:
        return None
    x, y, w, h, _ = res
    return x, y, w, h


def detectar_amb_morfo(img_bgr):
    """
    Reimplementació inline de morfo.py per poder cridar-la com a funció.
    Retorna (x, y, w, h) en coordenades originals, o None.
    """
    alt_orig, ample_orig = img_bgr.shape[:2]

    img_res = cv2.resize(img_bgr, (ample_orig * 2, alt_orig * 2),
                         interpolation=cv2.INTER_LANCZOS4)
    gris = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat  = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel_th)
    tophat    = cv2.morphologyEx(gris, cv2.MORPH_TOPHAT,   kernel_th)
    processada = cv2.add(gris, tophat)
    processada = cv2.subtract(processada, blackhat)

    sobelx = cv2.Sobel(processada, cv2.CV_16S, 1, 0, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobelx = cv2.GaussianBlur(abs_sobelx, (5, 5), 0)

    _, thresh = cv2.threshold(abs_sobelx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_unir = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    morf = cv2.dilate(thresh, kernel_unir, iterations=2)
    morf = cv2.morphologyEx(morf, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)))

    contorns, _ = cv2.findContours(morf.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidats = []
    for c in contorns:
        x, y, w, h = cv2.boundingRect(c)
        ratio   = w / float(h) if h > 0 else 0
        area    = cv2.contourArea(c)
        solidity = area / float(w * h) if (w * h) > 0 else 0
        if 1.8 < ratio < 7.5 and area > 1500 and solidity > 0.45:
            candidats.append((x, y, w, h, area))

    if not candidats:
        return None

    x2, y2, w2, h2, _ = max(candidats, key=lambda c: c[4])
    # Tornem a escala original (morfo treballava al doble)
    return int(x2 / 2), int(y2 / 2), int(w2 / 2), int(h2 / 2)


# ---------------------------------------------------------------------------
# AVALUACIÓ
# ---------------------------------------------------------------------------

def avaluar_pipeline(nom_pipeline, fn_detectar, ground_truth, llindar_iou=0.5, verbose=False):
    """
    Executa fn_detectar sobre totes les imatges del ground_truth i calcula
    les mètriques de detecció.

    Retorna dict amb: TP, FP, FN, precisio, recall, f1, iou_mig
    """
    tp = fp = fn = 0
    ious = []

    for img_path, gt_bbox in ground_truth.items():
        img = cv2.imread(img_path)
        if img is None:
            fn += 1
            continue

        det = fn_detectar(img)

        if det is None:
            fn += 1
            if verbose:
                print(f"  [FN] {os.path.basename(img_path)}")
        else:
            iou_val = iou(det, gt_bbox)
            ious.append(iou_val)
            if iou_val >= llindar_iou:
                tp += 1
                if verbose:
                    print(f"  [TP] {os.path.basename(img_path)}  IoU={iou_val:.3f}")
            else:
                fp += 1
                if verbose:
                    print(f"  [FP] {os.path.basename(img_path)}  IoU={iou_val:.3f}  "
                          f"det={det}  gt={gt_bbox}")

    total = tp + fp + fn
    precisio = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1       = 2 * precisio * recall / (precisio + recall) if (precisio + recall) > 0 else 0.0
    iou_mig  = float(np.mean(ious)) if ious else 0.0

    print(f"\n{'='*55}")
    print(f"  Pipeline: {nom_pipeline}")
    print(f"  Imatges avaluades: {total}")
    print(f"{'='*55}")
    print(f"  TP (IoU ≥ {llindar_iou}): {tp:3d}   FP: {fp:3d}   FN: {fn:3d}")
    print(f"  Precisió : {precisio:.3f}  ({100*precisio:.1f}%)")
    print(f"  Recall   : {recall:.3f}  ({100*recall:.1f}%)")
    print(f"  F1 Score : {f1:.3f}  ({100*f1:.1f}%)")
    print(f"  IoU mitjà (sobre deteccions): {iou_mig:.3f}")
    print(f"{'='*55}\n")

    return dict(tp=tp, fp=fp, fn=fn, precisio=precisio,
                recall=recall, f1=f1, iou_mig=iou_mig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Avalua pipelines de detecció de matrícules")
    parser.add_argument('--dataset',  default='dataset',
                        help='Directori amb imatges i anotacions (default: dataset)')
    parser.add_argument('--pipeline', choices=['nou', 'antic', 'tots'], default='tots',
                        help='Quin pipeline avaluar (default: tots)')
    parser.add_argument('--verbose',  action='store_true',
                        help='Mostra detall per cada imatge')
    parser.add_argument('--iou',      type=float, default=0.5,
                        help='Llindar IoU per considerar TP (default: 0.5)')
    args = parser.parse_args()

    gt = carregar_ground_truth(args.dataset)
    if not gt:
        print(f"ERROR: No s'han trobat anotacions a '{args.dataset}'")
        sys.exit(1)

    print(f"Ground truth carregat: {len(gt)} imatges anotades")

    resultats = {}

    if args.pipeline in ('nou', 'tots'):
        print("\nAvaluant pipeline MILLORAT...")
        resultats['nou'] = avaluar_pipeline(
            "Pipeline Millorat (CLAHE + Scoring)",
            detectar_amb_nou_pipeline, gt,
            llindar_iou=args.iou, verbose=args.verbose
        )

    if args.pipeline in ('antic', 'tots'):
        print("\nAvaluant pipeline ANTIC (morfo.py)...")
        resultats['antic'] = avaluar_pipeline(
            "Pipeline Antic (morfo.py)",
            detectar_amb_morfo, gt,
            llindar_iou=args.iou, verbose=args.verbose
        )

    # Resum comparatiu
    if len(resultats) == 2:
        nou   = resultats['nou']
        antic = resultats['antic']
        print("COMPARACIÓ DIRECTA")
        print(f"{'Mètrica':<15} {'Antic':>10} {'Nou':>10} {'Millora':>10}")
        print("-" * 47)
        for k, label in [('precisio', 'Precisió'), ('recall', 'Recall'),
                          ('f1', 'F1 Score'), ('iou_mig', 'IoU mitjà')]:
            delta = nou[k] - antic[k]
            signe = "+" if delta >= 0 else ""
            print(f"{label:<15} {antic[k]:>10.3f} {nou[k]:>10.3f} {signe}{delta:>9.3f}")


if __name__ == "__main__":
    main()
