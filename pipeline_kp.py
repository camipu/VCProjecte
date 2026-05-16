import cv2
import numpy as np
import os
import sys

# Forcem l'ús de xcb per evitar errors de Wayland en sistemes Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

def executar_pipeline_worldwide(directori_entrada="images", directori_sortida="resultats_global", path_models="models_worldwide"):
    """
    Pipeline de detecció de matrícules Worldwide usant ROI + SIFT Keypoints.
    """
    # 1. PREPARACIÓ DELS MODELS (Templates de diferents països)
    sift = cv2.SIFT_create()
    models_sift = []

    if not os.path.exists(path_models):
        os.makedirs(path_models)
        print(f"S'ha creat la carpeta '{path_models}'. Posa-hi retallades de matrícules (ex: eu.jpg, us.jpg).")
        return

    # Carreguem tots els models de la carpeta
    for f in os.listdir(path_models):
        ruta_m = os.path.join(path_models, f)
        img_m = cv2.imread(ruta_m, cv2.IMREAD_GRAYSCALE)
        if img_m is not None:
            kp, des = sift.detectAndCompute(img_m, None)
            if des is not None:
                models_sift.append((kp, des, f))
                print(f"Model carregat: {f}")

    if not models_sift:
        print("Error: No hi ha models SIFT vàlids a la carpeta de models.")
        return

    # Configuració del Matcher (FLANN)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Preparació directori de sortida
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    arxius = [f for f in os.listdir(directori_entrada) if f.lower().endswith(extensions)]

    print(f"\nProcessant {len(arxius)} imatges...\n")

    for nom_arxiu in arxius:
        ruta_img = os.path.join(directori_entrada, nom_arxiu)
        img_bgr = cv2.imread(ruta_img)
        if img_bgr is None: continue

        # Redimensionem per a un processament homogeni
        img_display = cv2.resize(img_bgr, (800, 600))
        gris = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris) # Robustesa davant diferents colors de placa

        # 2. GENERACIÓ DE ROI (Regions d'Interès) PERMISSIVA
        vores = cv2.Canny(gris, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        vores = cv2.dilate(vores, kernel, iterations=1)
        
        contorns, _ = cv2.findContours(vores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        millor_match_global = 0
        guanyador = None
        nom_model_guanyador = ""

        # 3. FILTRATGE PER KEYPOINTS DINS DE CADA CONTORN
        # Ordenem per àrea per analitzar primer els objectes més rellevants
        for c in sorted(contorns, key=cv2.contourArea, reverse=True)[:30]:
            x, y, w, h = cv2.boundingRect(c)
            
            # Filtre de mida mínima per evitar processar soroll insignificant
            if w < 40 or h < 12: continue

            # Extraiem la ROI i busquem descriptors
            roi = gris[y:y+h, x:x+w]
            kp_roi, des_roi = sift.detectAndCompute(roi, None)

            # SOLUCIÓ A L'ERROR: Comprovem que tenim prou descriptors (k=2)
            if des_roi is not None and len(des_roi) >= 2:
                for kp_m, des_m, nom_m in models_sift:
                    # Matching contra cada model (EU, US, etc.)
                    matches = flann.knnMatch(des_m, des_roi, k=2)
                    
                    # Lowe's ratio test
                    bons = [m for m, n in matches if m.distance < 0.7 * n.distance]
                    
                    # Guardem si és la regió que més s'assembla a una matrícula
                    if len(bons) > millor_match_global:
                        millor_match_global = len(bons)
                        guanyador = (x, y, w, h)
                        nom_model_guanyador = nom_m

        # 4. DIBUIX I RESULTAT FINAL
        # Posem un llindar mínim de 10 matches per considerar-ho matrícula
        if guanyador and millor_match_global > 10:
            x, y, w, h = guanyador
            cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"Match: {nom_model_guanyador} ({millor_match_global} pts)"
            cv2.putText(img_display, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f" [OK] {nom_arxiu} -> Detectada ({nom_model_guanyador})")
        else:
            print(f" [X] {nom_arxiu} -> No s'ha detectat cap matrícula fiable")

        # Guardar el resultat
        cv2.imwrite(os.path.join(directori_sortida, f"res_{nom_arxiu}"), img_display)

if __name__ == "__main__":
    # Pots passar els directoris com a arguments o deixar-los per defecte
    executar_pipeline_worldwide()