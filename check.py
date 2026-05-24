"""
Comprova els resultats del pipeline contra el ground truth del dataset.
Llegeix els bounding boxes guardats per pipeline_pas_a_pas (bbox.txt)
i calcula accuracy, precisió, recall i F1.

Ús:
  python3 check.py                        # usa dataset/ i resultats/
  python3 check.py --resultats altra_dir  # carpeta de resultats alternativa
  python3 check.py --iou 0.4              # canvia el llindar IoU (defecte: 0.5)
  python3 check.py --verbose              # mostra detall de cada cas
"""

import numpy as np
import os
import sys
import argparse


def iou(det, gt):
    """Intersection over Union entre (x,y,w,h) i (x,y,w,h)."""
    x1, y1, w1, h1 = det
    x2, y2, w2, h2 = gt
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def llegir_gt(txt_path):
    """Llegeix el ground truth d'un fitxer .txt  →  (x, y, w, h) o None."""
    try:
        with open(txt_path) as f:
            parts = f.readline().strip().split('\t')
        if len(parts) >= 5:
            return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    except Exception:
        pass
    return None


def llegir_bbox(bbox_path):
    """Llegeix el bbox detectat pel pipeline  →  (x, y, w, h) o None."""
    try:
        with open(bbox_path) as f:
            parts = f.readline().strip().split()
        if len(parts) == 4:
            return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--resultats', default='resultats',
                        help='Carpeta amb els resultats del pipeline (defecte: resultats)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='Llindar IoU per considerar detecció correcta (defecte 0.5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostra detall de cada imatge')
    args = parser.parse_args()

    tp = fp = fn = 0
    iou_vals = []

    files = sorted(f for f in os.listdir(args.dataset) if f.endswith('.txt'))
    if not files:
        print(f"No s'han trobat fitxers .txt a '{args.dataset}'")
        sys.exit(1)

    print(f"\n{'Imatge':<18} {'GT (x,y,w,h)':<22} {'Det (x,y,w,h)':<22} {'IoU':>6}  {'Resultat'}")
    print('─' * 80)

    for txt_f in files:
        base = txt_f.replace('.txt', '')
        gt = llegir_gt(os.path.join(args.dataset, txt_f))
        if gt is None:
            continue

        bbox_path = os.path.join(args.resultats, base, "bbox.txt")
        det = llegir_bbox(bbox_path)

        iou_v = 0.0
        if det is None:
            fn += 1
            resultat = '✗ FN (sense detecció)'
            det_str = 'None'
        else:
            iou_v = iou(det, gt)
            iou_vals.append(iou_v)
            det_str = f'({det[0]},{det[1]},{det[2]},{det[3]})'
            if iou_v >= args.iou:
                tp += 1
                resultat = '✓ TP'
            else:
                fp += 1
                resultat = f'✗ FP  (IoU={iou_v:.3f})'

        gt_str = f'({gt[0]},{gt[1]},{gt[2]},{gt[3]})'
        print(f"{base:<18} {gt_str:<22} {det_str:<22} {iou_v:>6.3f}  {resultat}")

    # ── Resum ──────────────────────────────────────────────────────────────
    total = tp + fp + fn
    precisio = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1       = 2 * precisio * recall / (precisio + recall) if (precisio + recall) > 0 else 0.0
    iou_mig  = float(np.mean(iou_vals)) if iou_vals else 0.0

    print('\n' + '═' * 60)
    print(f"  Imatges avaluades : {total}")
    print(f"  Llindar IoU       : {args.iou}")
    print('─' * 60)
    print(f"  TP  : {tp:3d}   FP  : {fp:3d}   FN  : {fn:3d}")
    print(f"  Precisió  : {100*precisio:6.1f}%")
    print(f"  Recall    : {100*recall:6.1f}%")
    print(f"  F1 Score  : {100*f1:6.1f}%")
    print(f"  IoU mitjà : {iou_mig:.3f}")
    print('═' * 60)


if __name__ == '__main__':
    main()
