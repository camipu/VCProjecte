import cv2
import numpy as np
import os

def pipeline_pas_a_pas(directori_entrada="dataset", directori_sortida="resultats_sobel"):
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    arxius = [f for f in os.listdir(directori_entrada) if f.lower().endswith(extensions)]

    print(f"Processant {len(arxius)} imatges i desant cada pas...\n")

    for nom_arxiu in arxius:
        # Creem una subcarpeta per a cada imatge per no barrejar els passos
        nom_sense_ext = os.path.splitext(nom_arxiu)[0]
        carpeta_pas = os.path.join(directori_sortida, nom_sense_ext)
        if not os.path.exists(carpeta_pas):
            os.makedirs(carpeta_pas)

        ruta_img = os.path.join(directori_entrada, nom_arxiu)
        img = cv2.imread(ruta_img) # [cite: 11]
        if img is None: continue

        print(f"--- Processant: {nom_arxiu} ---")

        # ---------------------------------------------------------------------
        # PAS 1: Grayscale 
        # ---------------------------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        cv2.imwrite(os.path.join(carpeta_pas, "01_escala_grisos.jpg"), gray)

        # ---------------------------------------------------------------------
        # PAS 2: Bilateral Filter 
        # ---------------------------------------------------------------------
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
        cv2.imwrite(os.path.join(carpeta_pas, "02_filtre_bilateral.jpg"), bfilter)

        # ---------------------------------------------------------------------
        # PAS 3: Canny Edge Detection 
        # ---------------------------------------------------------------------
        edged = cv2.Canny(bfilter, 30, 200) 
        cv2.imwrite(os.path.join(carpeta_pas, "03_vores_canny.jpg"), edged)

        # ---------------------------------------------------------------------
        # PAS 4: Find Contours & Selection 
        # ---------------------------------------------------------------------
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # [cite: 60]
        contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] 

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True) 
            if len(approx) == 4: 
                location = approx 
                break

        # Imatge de diagnòstic per veure TOTS els 10 contorns candidats en vermell
        img_contorns = img.copy()
        cv2.drawContours(img_contorns, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(carpeta_pas, "04_tots_els_contorns.jpg"), img_contorns)

        if location is not None:
            # -----------------------------------------------------------------
            # PAS 5: Aplicar Màscara
            # -----------------------------------------------------------------
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [location], 0, 255, -1)
            cv2.imwrite(os.path.join(carpeta_pas, "05_mascara_binaria.jpg"), mask)

            # Imatge emmascarada (Només es veu la zona de la matrícula)
            new_image = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(os.path.join(carpeta_pas, "06_imatge_emmascarada.jpg"), new_image)

            # -----------------------------------------------------------------
            # PAS 6: Retall de la ROI
            # -----------------------------------------------------------------
            x_indices, y_indices = np.where(mask == 255)
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            cropped_image = gray[x1:x2+1, y1:y2+1]
            cv2.imwrite(os.path.join(carpeta_pas, "07_roi_matricula_retallada.jpg"), cropped_image)

            # Guardem el bounding box detectat perquè check.py el pugui llegir
            bx, by, bw, bh = cv2.boundingRect(location)
            with open(os.path.join(carpeta_pas, "bbox.txt"), "w") as f:
                f.write(f"{bx} {by} {bw} {bh}\n")

            # -----------------------------------------------------------------
            # PAS 7: Render Final
            # -----------------------------------------------------------------
            img_final = img.copy()
            cv2.rectangle(img_final, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
            cv2.imwrite(os.path.join(carpeta_pas, "07_resultat_final.jpg"), img_final)
        else:
            print(" -> [X] No s'ha trobat cap contorn de 4 vèrtices per a aquesta imatge.")

    print(f"\nProcés completat. Revisa les subcarpetes a: '{directori_sortida}/'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--entrada', default='dataset')
    parser.add_argument('--sortida', default='resultats_sobel')
    args = parser.parse_args()
    pipeline_pas_a_pas(args.entrada, args.sortida)