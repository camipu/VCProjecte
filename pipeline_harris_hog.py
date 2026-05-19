import cv2
import numpy as np
import os
import sys


# ---------------------------------------------------------------------------
# PREPROCESSAMENT  (idèntic a pipeline_millorat.py)
# ---------------------------------------------------------------------------

def preprocessar(img_bgr, target_w=800):
    h, w = img_bgr.shape[:2]
    scale = target_w / w
    img = cv2.resize(img_bgr, (target_w, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# GENERACIÓ DE CANDIDATS  (idèntic a pipeline_millorat.py)
# ---------------------------------------------------------------------------

def extreure_candidats(gris):
    filtered = cv2.bilateralFilter(gris, d=11, sigmaColor=17, sigmaSpace=17)
    sobelx   = cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3)
    abs_sob  = cv2.convertScaleAbs(sobelx)
    _, thresh = cv2.threshold(abs_sob, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated  = cv2.dilate(thresh, kernel_h, iterations=2)
    kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated  = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_c)
    contorns, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contorns, abs_sob


# ---------------------------------------------------------------------------
# CRITERIS DE PUNTUACIÓ  — tècniques del curs
# ---------------------------------------------------------------------------

def score_ratio(w, h):
    """
    Gaussiana centrada al ratio EU 4.35 (520×112 mm).
    Entrega 18: característica geomètrica discriminativa.
    """
    return float(np.exp(-((w / h - 4.35) ** 2) / (2 * 1.2 ** 2)))


def score_contrast_inout(gris, x, y, w, h, marge=8):
    """
    IDEA DE L'USUARI + Descriptors de Haar (Entrega 22).

    Una matrícula EU és un rectangle blanc/groc (brillant) sobre carrosseria
    (fosca). Comparem la brillantor entre l'INTERIOR del bbox i una franja
    EXTERIOR de 'marge' píxels als quatre costats.

         ┌──── exterior fosc ────────────────────┐
         │  ┌──── interior clar ───────────────┐ │
         │  │  MATRICULA BLANCA / GROGA        │ │
         │  └──────────────────────────────────┘ │
         └───────────────────────────────────────┘

    Score = clamp( (μ_interior − μ_exterior) / 127 , 0, 1 )
    """
    H, W = gris.shape
    yi1, yi2 = max(y, 0), min(y + h, H)
    xi1, xi2 = max(x, 0), min(x + w, W)
    if yi2 <= yi1 or xi2 <= xi1:
        return 0.0

    mean_inner = float(np.mean(gris[yi1:yi2, xi1:xi2]))

    franges = []
    if x - marge >= 0:
        franges.append(gris[yi1:yi2, x - marge:x].ravel())
    if x + w + marge <= W:
        franges.append(gris[yi1:yi2, x + w:x + w + marge].ravel())
    if y - marge >= 0:
        franges.append(gris[y - marge:y, xi1:xi2].ravel())
    if y + h + marge <= H:
        franges.append(gris[y + h:y + h + marge, xi1:xi2].ravel())

    if not franges:
        return 0.0
    mean_outer = float(np.mean(np.concatenate(franges)))
    return max(0.0, min((mean_inner - mean_outer) / 127.0, 1.0))


def score_harris_vertexs(gris, x, y, w, h, k=0.04, radi=10):
    """
    Detector de Harris (Entrega 20):  R = det(M) − k·(trace M)²

    Una matrícula real té cantonades fortes als 4 vèrtexs (punt on la placa
    contrasta amb la carrosseria en dues direccions alhora).

        TL ●──────────────────────● TR
           │                      │
        BL ●──────────────────────● BR

    Busquem el màxim de R en un entorn de radi px al voltant de cada vèrtex.
    Score = mitjana dels 4 màxims normalitzats  →  [0, 1]
    """
    H, W = gris.shape
    pad  = radi + 2
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)

    roi = gris[y0:y1, x0:x1].astype(np.float32)
    if roi.size < 9:
        return 0.0

    R    = cv2.cornerHarris(roi, blockSize=3, ksize=3, k=k)
    Rpos = np.clip(R, 0, None)
    mx   = Rpos.max()
    if mx < 1e-9:
        return 0.0
    Rn = Rpos / mx

    vertexs = [
        (y - y0,     x - x0),
        (y - y0,     x + w - x0),
        (y + h - y0, x - x0),
        (y + h - y0, x + w - x0),
    ]

    total = 0.0
    for (vr, vc) in vertexs:
        r0 = max(vr - radi, 0);  r1 = min(vr + radi, Rn.shape[0])
        c0 = max(vc - radi, 0);  c1 = min(vc + radi, Rn.shape[1])
        patch = Rn[r0:r1, c0:c1]
        total += float(patch.max()) if patch.size > 0 else 0.0

    return min(total / 4.0, 1.0)


def score_hog_text(gris, x, y, w, h):
    """
    HOG (Entrega 18): el text de les lletres genera gradients predominantment
    quasi-verticals (|Gx| > |Gy|) perquè les lletres son barres verticals.

    Passos del curs:
      1. Calculem Gx = Sobel X, Gy = Sobel Y.
      2. Magnitud mag = √(Gx²+Gy²). Direcció α = atan2(Gy, Gx).
      3. Seleccionem gradients rellevants (mag > percentil 65).
      4. Proporció amb angle quasi-vertical (|Gx| > |Gy|).

    Score = densitat × proporció_vertical  →  [0, 1]
    """
    H, W = gris.shape
    roi  = gris[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)].astype(np.float32)
    if roi.size == 0:
        return 0.0

    gx  = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    umbral = max(float(np.percentile(mag, 65)), 5.0)
    mask   = mag > umbral
    if mask.sum() < 20:
        return 0.0

    prop_vert = float(np.sum(np.abs(gx[mask]) > np.abs(gy[mask]))) / float(mask.sum())
    densitat  = min(float(mask.sum()) / float(roi.size) / 0.30, 1.0)

    return min(densitat * prop_vert * 2.0, 1.0)


def score_area(w, h, img_h, img_w):
    rel = (w * h) / (img_h * img_w)
    if rel < 0.004 or rel > 0.15:
        return 0.0
    return max(0.0, 1.0 - abs(rel - 0.03) / 0.06)


def puntuar(x, y, w, h, gris, img_h, img_w, verbose=False):
    """
    Score combinat amb tècniques del curs de Visió per Computador:

      30% — Ratio gaussià             (E18, geomètria)
      25% — Contrast interior/extern  (E22 Haar + idea usuari)
      20% — HOG gradients verticals   (E18)
      15% — Harris als 4 vèrtexs      (E20)
      10% — Àrea relativa
    """
    sr = score_ratio(w, h)
    sc = score_contrast_inout(gris, x, y, w, h)
    sh = score_hog_text(gris, x, y, w, h)
    sv = score_harris_vertexs(gris, x, y, w, h)
    sa = score_area(w, h, img_h, img_w)
    total = sr * 0.30 + sc * 0.25 + sh * 0.20 + sv * 0.15 + sa * 0.10

    if verbose:
        print(f"    ratio={sr:.2f}  contrast={sc:.2f}  hog={sh:.2f}"
              f"  harris={sv:.2f}  area={sa:.2f}  →  {total:.3f}")
    return total


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def detectar_matricula(img_bgr, verbose=False):
    h_orig, w_orig = img_bgr.shape[:2]
    img_proc = preprocessar(img_bgr)
    h_proc, w_proc = img_proc.shape[:2]
    sx, sy = w_orig / w_proc, h_orig / h_proc

    gris = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    candidats, _ = extreure_candidats(gris)

    margin_x = max(4, int(0.015 * w_proc))
    margin_y = max(4, int(0.015 * h_proc))

    millor_score = 0.0
    millor = None

    for c in candidats:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        if (x < margin_x or y < margin_y
                or x + w > w_proc - margin_x
                or y + h > h_proc - margin_y):
            continue
        ratio = w / h
        if not (2.5 < ratio < 7.0):
            continue
        if w / w_proc < 0.08 or w / w_proc > 0.55:
            continue
        if h / h_proc < 0.02 or h / h_proc > 0.15:
            continue

        score = puntuar(x, y, w, h, gris, h_proc, w_proc, verbose=verbose)
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

def processar_directori(directori_entrada="images",
                        directori_sortida="resultats_harris_hog",
                        verbose=False):
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    ext    = ('.jpg', '.jpeg', '.png', '.bmp')
    arxius = sorted([f for f in os.listdir(directori_entrada)
                     if f.lower().endswith(ext)])
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

        res = detectar_matricula(img, verbose=verbose)

        if res:
            x, y, w, h, score = res
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, f"Score:{score:.2f} R:{w/h:.1f}",
                        (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detectades += 1
            print(f"  [OK] {nom}  score={score:.3f}  "
                  f"bbox=({x},{y},{w},{h})  ratio={w/h:.2f}")
        else:
            cv2.putText(img, "SENSE DETECCIO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print(f"  [ ] {nom}  — sense detecció")

        cv2.imwrite(os.path.join(directori_sortida, f"res_{nom}"), img)

    pct = 100 * detectades / len(arxius)
    print(f"\nDetectades: {detectades}/{len(arxius)} ({pct:.1f}%)")
    print(f"Resultats a: {directori_sortida}/")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verbose   = "--verbose" in sys.argv or "-v" in sys.argv
    args_rest = [a for a in sys.argv[1:] if not a.startswith("-")]
    dir_in    = args_rest[0] if len(args_rest) > 0 else "images"
    dir_out   = args_rest[1] if len(args_rest) > 1 else "resultats_harris_hog"
    processar_directori(dir_in, dir_out, verbose=verbose)
