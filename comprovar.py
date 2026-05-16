"""
Comprova els resultats del pipeline contra el ground truth del dataset.
Mostra per cada imatge si la detecció és correcta i calcula l'accuracy global.

Ús:
  python3 comprovar.py              # usa dataset/ i pipeline_millorat
  python3 comprovar.py --iou 0.4   # canvia el llindar IoU (defecte: 0.5)
  python3 comprovar.py --verbose    # mostra detall de cada cas
"""

import cv2
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='Llindar IoU per considerar detecció correcta (defecte 0.5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostra detall de cada imatge')
    args = parser.parse_args()

    from pipeline_millorat import detectar_matricula

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

        # Busca la imatge corresponent
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            p = os.path.join(args.dataset, base + ext)
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            fn += 1
            if args.verbose:
                print(f"  {base}: imatge no trobada")
            continue

        img = cv2.imread(img_path)
        if img is None:
            fn += 1
            continue

        det = detectar_matricula(img)

        if det is None:
            fn += 1
            iou_v = 0.0
            resultat = '✗ FN (sense detecció)'
            det_str = 'None'
        else:
            dx, dy, dw, dh, _ = det
            iou_v = iou((dx, dy, dw, dh), gt)
            iou_vals.append(iou_v)
            det_str = f'({dx},{dy},{dw},{dh})'
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
