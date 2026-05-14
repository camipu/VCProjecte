import cv2
import numpy as np
import os

def executar_pipeline_worldwide(directori_entrada="images", directori_sortida="resultats_global"):
    # 1. PREPARACIÓ MULTI-MODEL (Templates de diferents parts del món)
    # Hauries de tenir una petita carpeta amb 2 o 3 models: 'eu.jpg', 'us.jpg', 'asia.jpg'
    path_models = "models_worldwide"
    models_sift = []
    sift = cv2.SIFT_create()

    if os.path.exists(path_models):
        for f in os.listdir(path_models):
            img_m = cv2.imread(os.path.join(path_models, f), cv2.IMREAD_GRAYSCALE)
            if img_m is not None:
                kp, des = sift.detectAndCompute(img_m, None)
                models_sift.append((kp, des, f))
    
    # Configuració del Matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    arxius = [f for f in os.listdir(directori_entrada) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for nom_arxiu in arxius:
        img_bgr = cv2.imread(os.path.join(directori_entrada, nom_arxiu))
        if img_bgr is None: continue
        
        img_color = cv2.resize(img_bgr, (600, 600))
        gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris) # Crucial per a matrícules de diferents colors

        # FASE 2: ROI ultra-permissiva (Qualsevol cosa que pugui ser una placa)
        vores = cv2.Canny(gris, 50, 150)
        vores = cv2.dilate(vores, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contorns, _ = cv2.findContours(vores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        millor_match_global = 0
        guanyador = None
        tipus_detectat = ""

        # FASE 3: Matching contra tots els models del món
        for c in sorted(contorns, key=cv2.contourArea, reverse=True)[:25]:
            x, y, w, h = cv2.boundingRect(c)
            if w < 30 or h < 10: continue # Només filtrem soroll extrem

            roi = gris[y:y+h, x:x+w]
            kp_roi, des_roi = sift.detectAndCompute(roi, None)

            if des_roi is not None:
                for kp_m, des_m, nom_m in models_sift:
                    matches = flann.knnMatch(des_m, des_roi, k=2)
                    bons = [m for m, n in matches if m.distance < 0.75 * n.distance]
                    
                    if len(bons) > millor_match_global:
                        millor_match_global = len(bons)
                        guanyador = (x, y, w, h)
                        tipus_detectat = nom_m

        # FASE 4: Resultat
        if guanyador and millor_match_global > 7:
            x, y, w, h = guanyador
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_color, f"WORLDWIDE ({tipus_detectat})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(directori_sortida, f"res_{nom_arxiu}"), img_color)

executar_pipeline_worldwide()