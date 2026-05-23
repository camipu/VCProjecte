import cv2
import numpy as np
import os
import easyocr

# Forcem l'ús de xcb per evitar errors de visualització en sistemes Linux/Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"

def pipeline_identic_pdf(directori_entrada="images", directori_sortida="resultats_sobel"):
    # Inicialitzem el lector d'EasyOCR en anglès (tal com fa el PDF)
    print("Inicialitzant EasyOCR...")
    reader = easyocr.Reader(['en'])

    if not os.path.exists(directori_sortida):
        os.makedirs(directori_sortida)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    arxius = [f for f in os.listdir(directori_entrada) if f.lower().endswith(extensions)]

    print(f"Processant {len(arxius)} imatges utilitzant NOMÉS el mètode del PDF...\n")

    for nom_arxiu in arxius:
        ruta_img = os.path.join(directori_entrada, nom_arxiu)
        img = cv2.imread(ruta_img)
        if img is None: continue

        # 1. Read in Image, Grayscale and Blur (Pàg. 1 del PDF)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Apply filter and find edges for localization (Pàg. 2 del PDF)
        # Filtre bilateral per reduir soroll mantenint les vores nítides
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        # Detecció de vores amb Canny (llindars fixos 30 i 200)
        edged = cv2.Canny(bfilter, 30, 200)

        # 3. Find Contours and Apply Mask (Pàg. 2 i 3 del PDF)
        # Busquem la jerarquia completa de contorns amb RETR_TREE
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Grab contours (ajust de compatibilitat de versions de OpenCV que fa imutils)
        contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
        # Ordenem per àrea de major a menor i ens quedem amb els 10 primers
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            # Aproximació poligonal amb un límit de precisió de 10
            approx = cv2.approxPolyDP(contour, 10, True)
            # Condició estricta: ha de tenir exactament 4 vèrtices
            if len(approx) == 4:
                location = approx
                break

        # Si hem trobat un polígon de 4 vèrtices, procedim a extreure la ROI i l'OCR
        if location is not None:
            # Creem la màscara binària buida (tot negre)
            mask = np.zeros(gray.shape, np.uint8)
            # Dibuixem el contorn de 4 punts en blanc (255) a la màscara
            cv2.drawContours(mask, [location], 0, 255, -1)
            # Operació bitwise_and per aïllar la placa
            new_image = cv2.bitwise_and(img, img, mask=mask)

            # Busquem les coordenades de tall mitjançant els píxels blancs (Pàg. 3 del PDF)
            x_indices, y_indices = np.where(mask == 255)
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            cropped_image = gray[x1:x2+1, y1:y2+1]

            # 4. Use Easy OCR To Read Text (Pàg. 4 del PDF)
            try:
                result = reader.readtext(cropped_image)
                
                if result:
                    # Extraiem el text detectat (penúltim element de la tupla de resultat)
                    text = result[0][-2]
                    print(f" [OK] {nom_arxiu} -> Text detectat: {text}")
                else:
                    text = "NO LLEGIBLE"
                    print(f" [?] {nom_arxiu} -> Matrícula trobada, però l'OCR no llegeix el text")
            except Exception as e:
                text = "ERROR OCR"
                print(f" [!] Error en processar l'OCR a {nom_arxiu}: {e}")

            # 5. Render Result (Pàg. 5 del PDF)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Dibuixem el text verd sobre la imatge (posicionat segons els vèrtices de l'approx)
            res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60),
                              fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # Dibuixem el rectangle directament utilitzant els punts extrems oposats de l'aproximació
            res = cv2.rectangle(res, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

            # Guardem el resultat pintat
            cv2.imwrite(os.path.join(directori_sortida, f"pdf_{nom_arxiu}"), res)
        else:
            print(f" [X] {nom_arxiu} -> El contorn no ha donat exactament 4 vèrtices. Saltant.")
            # Guardem la imatge original sense marcar per saber que ha fallat la segmentació
            cv2.imwrite(os.path.join(directori_sortida, f"pdf_{nom_arxiu}"), img)

if __name__ == "__main__":
    pipeline_identic_pdf()