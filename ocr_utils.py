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
    Descriptor millorat: píxels plans + projeccions + densitats 3x3 + aspect ratio amb PES.
    """
    pixeles = img_bin.flatten() / 255.0                              
    proj_h  = np.sum(img_bin, axis=1) / (255.0 * IMG_WIDTH)         
    proj_v  = np.sum(img_bin, axis=0) / (255.0 * IMG_HEIGHT)        
    

    h3, w3 = IMG_HEIGHT // 3, IMG_WIDTH // 3
    total  = float(np.sum(img_bin)) or 1.0
    
    dens = []
    for i in range(3):
        for j in range(3):
            y0, y1 = i * h3, (i + 1) * h3 if i < 2 else IMG_HEIGHT
            x0, x1 = j * w3, (j + 1) * w3 if j < 2 else IMG_WIDTH
            dens.append(np.sum(img_bin[y0:y1, x0:x1]))
    dens = np.array(dens) / total
    
    peso_aspect_ratio = aspect_ratio * 15.0

    return np.concatenate([pixeles, proj_h, proj_v, dens, [peso_aspect_ratio]]).astype(np.float32)


def binarizar_char(img_gray):
    """Binaritza una ROI de caràcter: caràcter blanc sobre fons negre."""
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


def segmentar_caracteres(roi_gray, min_area=50):
    """
    Segmenta una ROI de matrícula en components connexes vàlids.
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

        if area >= min_area and h > (w * 0.75) and h > 15:
            components.append((x, y, w, h, area))

    components.sort(key=lambda c: c[0])
    return thresh, components