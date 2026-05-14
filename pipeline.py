import cv2
import numpy as np
import os
import sys


def canny_adaptatiu(img_gris, sigma=0.33):
    """
    Calcula els llindars de Canny de forma adaptativa
    basant-se en la mediana de la imatge.
    """
    v = np.median(img_gris)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_gris, lower, upper)


def executar_pipeline_imatges_compost(directori_entrada="images", directori_sortida="resultats_pipeline_compost"):
    """
    Processa imatges per detectar matrícules usant ràtio d'aspecte i àrea.
    Genera un compost 2x2 amb tota la informació de depuració.

    Millores aplicades respecte la versió anterior:
      - Canny adaptatiu (llindars basats en la mediana de la imatge)
      - Equalització d'histograma per robustesa davant canvis d'il·luminació
      - Dilatació morfològica per tancar contorns trencats
      - Eliminació de la condició len(aprox)==4 (massa restrictiva)
      - Ús de boundingRect directament per calcular ràtio i àrea
      - Fins a 50 candidats (en lloc de 15)
      - Filtre per àrea mínima absoluta (en lloc de tall fix per rànquing)
      - RETR_EXTERNAL en comptes de RETR_TREE (més net i ràpid)
    """
    mida_taulell = (400, 400)  # Mida per a cada quadrant

    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)
        print(f"Directori creat: {directori_sortida}")

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    arxius = [f for f in os.listdir(directori_entrada) if f.lower().endswith(extensions)]

    if not arxius:
        print(f"No s'han trobat imatges a {directori_entrada}")
        return

    print(f"Processant {len(arxius)} imatges...\n")

    for nom_arxiu in arxius:
        ruta_completa = os.path.join(directori_entrada, nom_arxiu)
        img_bgr = cv2.imread(ruta_completa)
        if img_bgr is None:
            print(f"[SKIP] No s'ha pogut llegir: {nom_arxiu}")
            continue

        # Redimensionar per al compost
        img_color_original = cv2.resize(img_bgr, mida_taulell)

        # -----------------------------------------------------------
        # FASE 1: PREPROCESSAT ROBUST
        # -----------------------------------------------------------
        # 1a. Convertir a grisos
        gris = cv2.cvtColor(img_color_original, cv2.COLOR_BGR2GRAY)

        # 1b. Equalització d'histograma: millora el contrast en
        #     condicions d'il·luminació adverses (nit, contallums, etc.)
        gris = cv2.equalizeHist(gris)

        # 1c. Suavitzat gaussià per reduir soroll
        blurred = cv2.GaussianBlur(gris, (5, 5), 0)

        # 1d. Canny adaptatiu: els llindars s'ajusten a cada imatge
        #     evitant valors fixos que fallen amb il·luminació variable
        vores = canny_adaptatiu(blurred)

        # 1e. Dilatació morfològica: tanca els contorns trencats
        #     que genera el text interior de la matrícula
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        vores = cv2.dilate(vores, kernel, iterations=1)

        vores_bgr = cv2.cvtColor(vores, cv2.COLOR_GRAY2BGR)

        # -----------------------------------------------------------
        # FASE 2: CERCA DE CONTORNS
        # -----------------------------------------------------------
        # RETR_EXTERNAL: només contorns externs (més net que RETR_TREE)
        contorns, _ = cv2.findContours(vores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtre per àrea mínima absoluta (evita microsorolls)
        # i limitem a 50 candidats (en lloc de 15)
        contorns_filtrats = [c for c in contorns if cv2.contourArea(c) > 200]
        contorns_ordenats = sorted(contorns_filtrats, key=cv2.contourArea, reverse=True)[:50]

        # -----------------------------------------------------------
        # FASE 3: FILTRAT GEOMÈTRIC — CANDIDATS
        # -----------------------------------------------------------
        # MILLORA CLAU: usem boundingRect directament i NO exigim
        # len(aprox)==4. Això evita descartar matrícules amb cantonades
        # lleugerament arrodonides, deformació per perspectiva, etc.
        img_candidats_f3 = img_color_original.copy()
        img_final_f4 = img_color_original.copy()
        detectat_f4 = False
        millor_candidat = None  # Guardem el millor per si n'hi ha més d'un

        for c in contorns_ordenats:
            x, y, w, h = cv2.boundingRect(c)

            # Evitar divisions per zero
            if h == 0:
                continue

            ratio = w / float(h)
            area = cv2.contourArea(c)

            # --- FASE 3: mostrar tots els candidats amb ràtio plausible ---
            # Rang ampli per no perdre res en la visualització de debug
            if 1.0 < ratio < 7.0 and 200 < area < 60000:
                cv2.rectangle(img_candidats_f3, (x, y), (x + w, y + h), (255, 150, 0), 2)
                label_cand = f"R:{ratio:.1f} A:{int(area)}"
                cv2.putText(img_candidats_f3, label_cand, (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 0), 1)

                # --- FASE 4: filtre estricte per a la selecció final ---
                # Ràtio d'una matrícula europea: aprox. 4.7:1 (520x112 mm)
                # Rang permissiu per cobrir angles i retalls parcials
                if 1.5 < ratio < 5.5 and area > 500 and not detectat_f4:
                    millor_candidat = (x, y, w, h, ratio, area)
                    detectat_f4 = True

        # Dibuixar la detecció final (Fase 4)
        if detectat_f4 and millor_candidat:
            x, y, w, h, ratio, area = millor_candidat
            cv2.rectangle(img_final_f4, (x, y), (x + w, y + h), (0, 255, 0), 3)
            info_final = f"MATRICULA R:{ratio:.2f} A:{int(area)}"
            cv2.putText(img_final_f4, info_final, (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        else:
            # Missatge visible quan no es detecta res
            cv2.putText(img_final_f4, "SENSE DETECCIO", (80, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # -----------------------------------------------------------
        # ASSEMBLEATGE DEL COMPOST 2x2
        # -----------------------------------------------------------
        fila_sup = np.hstack((img_color_original, vores_bgr))
        fila_inf = np.hstack((img_candidats_f3, img_final_f4))
        compost = np.vstack((fila_sup, fila_inf))

        # Textos dels quadrants
        overlay_params = [
            ("1. Original",                    10,  25),
            ("2. Canny Adaptatiu + Dilatacio", 410, 25),
            ("3. Candidats (BoundingRect)",    10,  425),
            ("4. Seleccio Final",              410, 425),
        ]
        for text, tx, ty in overlay_params:
            # Ombra negra per llegibilitat
            cv2.putText(compost, text, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(compost, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Separador visual entre quadrants
        cv2.line(compost, (0, 400), (800, 400), (200, 200, 200), 1)
        cv2.line(compost, (400, 0), (400, 800), (200, 200, 200), 1)

        # Guardar resultat
        ruta_guardat = os.path.join(directori_sortida, f"compost_{nom_arxiu}")
        cv2.imwrite(ruta_guardat, compost)

        estat = f"Deteccio OK (R:{millor_candidat[4]:.2f}, A:{int(millor_candidat[5])})" if detectat_f4 else "Sense deteccio"
        print(f"  [{nom_arxiu}] -> {estat}")

    print(f"\nResultats guardats a: {directori_sortida}/")


if __name__ == "__main__":
    dir_in  = sys.argv[1] if len(sys.argv) > 1 else 'images'
    dir_out = sys.argv[2] if len(sys.argv) > 2 else 'resultats_pipeline_compost'

    if os.path.isdir(dir_in):
        executar_pipeline_imatges_compost(dir_in, dir_out)
    else:
        print(f"Error: No s'ha trobat el directori '{dir_in}'")
        print(f"Ús: python {sys.argv[0]} <directori_entrada> [directori_sortida]")