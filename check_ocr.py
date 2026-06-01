"""
Avalua els resultats OCR de resultats_ocr/ contra el ground truth de dataset/.

Mètriques reportades:
  - Global accuracy        : matrícules correctes / total imatges del dataset
  - Taxa de detecció       : imatges amb resultat OCR / total
  - OCR accuracy (detectat): correctes / detectades  (subset processat)
  - Bbox accuracy (IoU)    : deteccions bbox amb IoU ≥ llindar
  - OCR sobre bbox bona    : correctes OCR dins les que tenien bbox correcta

Ús:
  python3 check_ocr.py
  python3 check_ocr.py --dataset dataset --ocr resultats_ocr --resultats resultats
  python3 check_ocr.py --iou 0.4 --verbose
  python3 check_ocr.py --parcial        # compta com a correcte si totes les lletres coincideixen
"""

import os
import sys
import argparse


# ─── Lectura de fitxers ──────────────────────────────────────────────────────

def llegir_gt(txt_path):
    """Llegeix ground truth → (x, y, w, h, plate_text) o None."""
    try:
        with open(txt_path) as f:
            parts = f.readline().strip().split('\t')
        # Format: filename  x  y  w  h  plate_text
        if len(parts) >= 6:
            return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), parts[5].strip().upper()
    except Exception:
        pass
    return None


def llegir_ocr(ocr_path):
    """Llegeix el text OCR → string o None."""
    try:
        with open(ocr_path) as f:
            text = f.readline().strip().upper()
        return text if text else None
    except Exception:
        return None


def llegir_bbox(bbox_path):
    """Llegeix bbox detectat → (x, y, w, h) o None."""
    try:
        with open(bbox_path) as f:
            parts = f.readline().strip().split()
        if len(parts) == 4:
            return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        pass
    return None


# ─── Mètriques ───────────────────────────────────────────────────────────────

def iou(det, gt):
    x1, y1, w1, h1 = det
    x2, y2, w2, h2 = gt
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def es_correcte(pred, gt, parcial=False):
    """Compara predicció OCR amb ground truth (exacte o per caràcters)."""
    if pred is None:
        return False
    if parcial:
        # Compta com a correcte si tots els caràcters predits coincideixen (subset)
        return all(c in gt for c in pred) and len(pred) == len(gt)
    return pred == gt


def distancia_edicio(a, b):
    """Distància de Levenshtein entre dues cadenes."""
    if a is None:
        return len(b)
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m]


def accuracy_caracters(pred, gt):
    """Percentatge de caràcters correctes (1 - normalized edit distance)."""
    if not gt:
        return 0.0
    d = distancia_edicio(pred or "", gt)
    return max(0.0, 1.0 - d / len(gt))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   default='dataset')
    parser.add_argument('--ocr',       default='resultats_ocr',
                        help='Carpeta amb els resultats OCR (defecte: resultats_ocr)')
    parser.add_argument('--resultats', default='resultats',
                        help='Carpeta amb els bbox detectats (defecte: resultats)')
    parser.add_argument('--iou',       type=float, default=0.5,
                        help='Llindar IoU per considerar bbox correcta (defecte: 0.5)')
    parser.add_argument('--parcial',   action='store_true',
                        help='Compta matrícula correcta si la distància d\'edició ≤ 1')
    parser.add_argument('--verbose',   action='store_true',
                        help='Mostra detall de cada imatge')
    args = parser.parse_args()

    gt_files = sorted(f for f in os.listdir(args.dataset) if f.endswith('.txt'))
    if not gt_files:
        print(f"No s'han trobat .txt a '{args.dataset}'")
        sys.exit(1)

    # Contadors principals
    total          = 0   # imatges al dataset
    detectades     = 0   # imatges amb resultat OCR (fitxer matricula_ocr.txt present)
    correctes      = 0   # exactament iguals
    bbox_bona      = 0   # bbox detectada amb IoU ≥ llindar
    bbox_i_ocr     = 0   # bbox bona I OCR correcte
    char_acc_sum   = 0.0 # suma d'accuracies per caràcter (només detectades)
    edit_dists     = []  # distàncies d'edició (només detectades)

    if args.verbose:
        print(f"\n{'Imatge':<18} {'GT':<10} {'OCR':<10} {'Coincideix':<11} {'IoU':>6}  Bbox")
        print('─' * 68)

    for txt_f in gt_files:
        base = txt_f.replace('.txt', '')
        gt_data = llegir_gt(os.path.join(args.dataset, txt_f))
        if gt_data is None:
            continue

        gt_x, gt_y, gt_w, gt_h, gt_text = gt_data
        total += 1

        # Llegir OCR
        ocr_path = os.path.join(args.ocr, base, "matricula_ocr.txt")
        pred_text = llegir_ocr(ocr_path)

        # Llegir bbox
        bbox_path = os.path.join(args.resultats, base, "bbox.txt")
        det_bbox  = llegir_bbox(bbox_path)

        # Avaluar bbox
        iou_val = 0.0
        bbox_ok = False
        if det_bbox is not None:
            iou_val = iou(det_bbox, (gt_x, gt_y, gt_w, gt_h))
            bbox_ok = iou_val >= args.iou
            if bbox_ok:
                bbox_bona += 1

        # Avaluar OCR
        if pred_text is not None:
            detectades += 1
            char_acc_sum += accuracy_caracters(pred_text, gt_text)
            edit_dists.append(distancia_edicio(pred_text, gt_text))

        if args.parcial:
            ok = (distancia_edicio(pred_text or "", gt_text) <= 1)
        else:
            ok = es_correcte(pred_text, gt_text)

        if ok:
            correctes += 1
            if bbox_ok:
                bbox_i_ocr += 1

        if args.verbose:
            estat_ocr  = "✓" if ok          else "✗"
            estat_bbox = "✓" if bbox_ok     else ("–" if det_bbox is None else "✗")
            pred_show  = pred_text if pred_text else "—"
            print(f"{base:<18} {gt_text:<10} {pred_show:<10} {estat_ocr:<11} {iou_val:>6.3f}  {estat_bbox}")

    # ── Resum ─────────────────────────────────────────────────────────────────
    no_detectades = total - detectades
    correctes_detectades = sum(
        1 for txt_f in gt_files
        if llegir_ocr(os.path.join(args.ocr, txt_f.replace('.txt', ''), "matricula_ocr.txt")) is not None
        and es_correcte(
            llegir_ocr(os.path.join(args.ocr, txt_f.replace('.txt', ''), "matricula_ocr.txt")),
            (llegir_gt(os.path.join(args.dataset, txt_f)) or (None, None, None, None, ""))[4]
        )
    )

    global_acc  = 100 * correctes  / total       if total       > 0 else 0.0
    det_rate    = 100 * detectades / total        if total       > 0 else 0.0
    ocr_acc     = 100 * correctes_detectades / detectades if detectades > 0 else 0.0
    bbox_rate   = 100 * bbox_bona  / total        if total       > 0 else 0.0
    bbox_ocr    = 100 * bbox_i_ocr / bbox_bona    if bbox_bona   > 0 else 0.0
    char_acc    = 100 * char_acc_sum / detectades if detectades  > 0 else 0.0
    edit_mitja  = sum(edit_dists) / len(edit_dists) if edit_dists else 0.0

    mode = "distància edició ≤ 1" if args.parcial else "coincidència exacta"

    print('\n' + '═' * 60)
    print(f"  AVALUACIÓ OCR  [{mode}]")
    print('═' * 60)
    print(f"  Imatges al dataset         : {total}")
    print(f"  Imatges processades (OCR)  : {detectades}  ({det_rate:.1f}%)")
    print(f"  Imatges sense OCR          : {no_detectades}")
    print('─' * 60)
    print(f"  ── Accuracy global ──────────────────────────────────")
    print(f"     Matrícules correctes     : {correctes}/{total}  →  {global_acc:.1f}%")
    print(f"  ── OCR sobre detectades ─────────────────────────────")
    print(f"     Correctes / detectades   : {correctes_detectades}/{detectades}  →  {ocr_acc:.1f}%")
    print(f"     Acc. per caràcter (avg)  : {char_acc:.1f}%")
    print(f"     Distància edició (avg)   : {edit_mitja:.2f} caràcters")
    print(f"  ── Detecció bbox (IoU ≥ {args.iou}) ─────────────────────")
    print(f"     Bbox correctes           : {bbox_bona}/{total}  →  {bbox_rate:.1f}%")
    print(f"     OCR correcte | bbox bona : {bbox_i_ocr}/{bbox_bona}  →  {bbox_ocr:.1f}%")
    print('═' * 60)


if __name__ == '__main__':
    main()
