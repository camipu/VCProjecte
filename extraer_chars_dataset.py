"""
Extreu caràcters individuals de les ROIs del dataset i els guarda a LetrasGeneradas/
per ampliar el conjunt d'entrenament del KNN.

Procés:
  Per cada imatge a resultats/{nom}/07_roi_matricula_retallada.jpg:
    1. Llegeix el text ground truth de dataset/{nom}.txt
    2. Segmenta els caràcters amb components connexes
    3. Si # components == longitud del GT, guarda cada caràcter etiquetat

Ús:
  python3 extraer_chars_dataset.py
  python3 extraer_chars_dataset.py --resultats resultats --min_area 50
"""

import cv2
import os
import argparse
from ocr_utils import IMG_WIDTH, IMG_HEIGHT, segmentar_caracteres


DATASET   = "dataset"
RESULTATS = "resultats"
LETRAS    = "LetrasGeneradas"


def llegir_gt_text(txt_path):
    try:
        with open(txt_path) as f:
            parts = f.readline().strip().split('\t')
        if len(parts) >= 6:
            return parts[5].strip().upper()
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   default=DATASET)
    parser.add_argument('--resultats', default=RESULTATS)
    parser.add_argument('--letras',    default=LETRAS)
    parser.add_argument('--min_area',  type=int, default=50)
    args = parser.parse_args()

    gt_files = sorted(f for f in os.listdir(args.dataset) if f.endswith('.txt'))

    no_roi      = 0
    mismatch    = 0
    aprofitades = 0
    extrets     = 0
    duplicats   = 0
    per_classe  = {}

    print(f"Processant {len(gt_files)} imatges del dataset...\n")

    for txt_f in gt_files:
        base    = txt_f.replace('.txt', '')
        gt_text = llegir_gt_text(os.path.join(args.dataset, txt_f))
        if not gt_text:
            continue

        roi_path = os.path.join(args.resultats, base, "07_roi_matricula_retallada.jpg")
        if not os.path.exists(roi_path):
            no_roi += 1
            continue

        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            no_roi += 1
            continue

        thresh, components = segmentar_caracteres(roi, args.min_area)

        if len(components) != len(gt_text):
            mismatch += 1
            continue

        aprofitades += 1

        for i, (x, y, w, h, area) in enumerate(components):
            char        = gt_text[i]
            carpeta     = os.path.join(args.letras, char)
            os.makedirs(carpeta, exist_ok=True)

            nom_fitxer = f"ext_{base}_{i}.jpg"
            dest_path  = os.path.join(carpeta, nom_fitxer)

            if os.path.exists(dest_path):
                duplicats += 1
                continue

            recorte = thresh[y:y+h, x:x+w]
            recorte = cv2.copyMakeBorder(recorte, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            recorte = cv2.resize(recorte, (IMG_WIDTH, IMG_HEIGHT))
            cv2.imwrite(dest_path, recorte)

            extrets += 1
            per_classe[char] = per_classe.get(char, 0) + 1

    # ── Resum ───────────────────────────────────────────────────────────
    print(f"{'─'*44}")
    print(f"  Imatges sense ROI           : {no_roi}")
    print(f"  Mismatch # caràcters        : {mismatch}")
    print(f"  Imatges aprofitades         : {aprofitades}")
    print(f"  Caràcters ja existien       : {duplicats}")
    print(f"  Caràcters nous guardats     : {extrets}")
    print(f"{'─'*44}")

    if per_classe:
        print("\nNous caràcters extrets per classe:")
        for char in sorted(per_classe):
            bar = '█' * min(per_classe[char], 30)
            print(f"  {char}  {bar:30s}  +{per_classe[char]}")

        total_per_class = {}
        for char in sorted(os.listdir(args.letras)):
            carpeta = os.path.join(args.letras, char)
            if os.path.isdir(carpeta):
                imgs = [f for f in os.listdir(carpeta)
                        if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
                total_per_class[char] = len(imgs)
        if total_per_class:
            print(f"\nTotal mostres per classe (original + extrets):")
            for char, n in sorted(total_per_class.items()):
                print(f"  {char}: {n}")

    if extrets > 0:
        print(f"\nAra executa: python3 entrenar_ocr.py")
    else:
        print(f"\nNo s'han extret caràcters nous. Comprova que resultats/ té les ROIs generades.")


if __name__ == '__main__':
    main()
