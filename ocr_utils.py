"""
Utilitats compartides entre entrenar_ocr.py i segment_cc.py.

IMPORTANT: extraer_caracteristicas() ha de ser idèntica en entrenament i inferència.
Qualsevol canvi aquí afecta els dos scripts i requereix re-entrenar el model.
"""

import cv2
import numpy as np

IMG_WIDTH, IMG_HEIGHT = 20, 30


def extraer_caracteristicas(img_bin, aspect_ratio=1.0):
    """
    Descriptor de 655 característiques: píxels plans + projeccions + densitats + aspect ratio.
    img_bin: imatge 20×30 binaritzada (blanc=caràcter, negre=fons).
    """
    pixeles = img_bin.flatten() / 255.0                              # 600
    proj_h  = np.sum(img_bin, axis=1) / (255.0 * IMG_WIDTH)         #  30
    proj_v  = np.sum(img_bin, axis=0) / (255.0 * IMG_HEIGHT)        #  20
    hm, wm  = IMG_HEIGHT // 2, IMG_WIDTH // 2
    total   = float(np.sum(img_bin)) or 1.0
    dens    = np.array([                                             #   4
        np.sum(img_bin[0:hm, 0:wm]),
        np.sum(img_bin[0:hm, wm:]),
        np.sum(img_bin[hm:,  0:wm]),
        np.sum(img_bin[hm:,  wm:]),
    ]) / total
    return np.concatenate([pixeles, proj_h, proj_v, dens, [aspect_ratio]]).astype(np.float32)


def binarizar_char(img_gray):
    """Binaritza una ROI de caràcter: caràcter blanc sobre fons negre."""
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


def segmentar_caracteres(roi_gray, min_area=50, min_h_w_ratio=0.4):
    """
    Segmenta una ROI de matrícula en components connexes vàlids.
    Retorna (thresh_binary, [(x, y, w, h, area), ...]) ordenats d'esquerra a dreta.
    """
    thresh = binarizar_char(roi_gray)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    components = []
    for i in range(1, num_labels):
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and h > w * min_h_w_ratio:
            components.append((x, y, w, h, area))

    components.sort(key=lambda c: c[0])
    return thresh, components
