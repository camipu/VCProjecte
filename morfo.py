import cv2
import numpy as np
import os

def processar_millorat(directori_entrada="images", directori_sortida="resultats_top"):
    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    for nom_arxiu in os.listdir(directori_entrada):
        if not nom_arxiu.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        
        img = cv2.imread(os.path.join(directori_entrada, nom_arxiu))
        alt_orig, ample_orig = img.shape[:2]
        
        # 1. RESIZE ESTRATÈGIC (Treballem a una escala major per a detalls petits)
        img_res = cv2.resize(img, (ample_orig * 2, alt_orig * 2), interpolation=cv2.INTER_LANCZOS4)
        gris = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        # 2. MILLORA DE CONTRAST LOCAL (Top-Hat / Black-Hat)
        # Això ressalta les lletres fosques sobre fons clar o viceversa
        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel_th)
        tophat = cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, kernel_th)
        # Combinem per tenir el màxim contrast de detalls
        processada = cv2.add(gris, tophat)
        processada = cv2.subtract(processada, blackhat)

        # 3. SOBEL VERTICAL I SUAVITZAT
        # Busquem gradients en X per detectar les línies verticals de les lletres
        sobelx = cv2.Sobel(processada, cv2.CV_16S, 1, 0, ksize=3)
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobelx = cv2.GaussianBlur(abs_sobelx, (5, 5), 0)

        # 4. BINARITZACIÓ + DILATACIÓ DIRECCIONAL (Clau per matrícules estirades)
        _, thresh = cv2.threshold(abs_sobelx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Dilatem molt en horitzontal per unir les lletres (fins i tot si la matrícula és llarga)
        kernel_unir = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3)) 
        morf = cv2.dilate(thresh, kernel_unir, iterations=2)
        # Tanquem verticalment per donar consistència al bloc
        morf = cv2.morphologyEx(morf, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)))

        # 5. FILTRATGE GEOMÈTRIC AVANÇAT
        contorns, _ = cv2.findContours(morf.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidats = []

        for c in contorns:
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            area = cv2.contourArea(c)
            solidity = area / float(w * h) # Quant omple el rectangle

            # Filtres Worldwide:
            # - Ratio: des de 2.0 (EUA/Quadrades) fins a 7.0 (Molt estirades)
            # - Solidity: Una matrícula sol ser un bloc força sòlid (>0.4)
            if 1.8 < ratio < 7.5 and area > 1500 and solidity > 0.45:
                candidats.append((x, y, w, h, area))

        # 6. SELECCIÓ FINAL I DIBUIX
        if candidats:
            # Triem el candidat més gran (normalment la matrícula real)
            millor = max(candidats, key=lambda x: x[4])
            x, y, w, h, _ = millor
            cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(img_res, f"RATIO: {w/h:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Retornem a mida original per guardar
        img_final = cv2.resize(img_res, (ample_orig, alt_orig))
        cv2.imwrite(os.path.join(directori_sortida, f"final_{nom_arxiu}"), img_final)
        print(f"Processada: {nom_arxiu}")

if __name__ == "__main__":
    processar_millorat()