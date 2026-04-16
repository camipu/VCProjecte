import cv2
import numpy as np

def procesar_matricula(ruta_imagen):
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # 1. Preprocesamiento (Fase 1: Detección) [cite: 17]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicamos un filtro para reducir ruido 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detección de bordes
    edged = cv2.Canny(blurred, 50, 200)

    # 2. Buscar el contorno de la placa [cite: 18]
    contornos, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        
        # Si tiene 4 puntos, es nuestra ROI (Region of Interest) [cite: 19]
        if len(aprox) == 4:
            x, y, w, h = cv2.boundingRect(aprox)
            roi = img[y:y+h, x:x+w]
            
            # Dibujar sobre la imagen original para el Checkpoint [cite: 25, 26]
            cv2.drawContours(img, [aprox], -1, (0, 255, 0), 3)
            
            # Mostrar resultados
            cv2.imshow("Deteccion en Imagen Original", img)
            cv2.imshow("ROI Extraida (Matricula)", roi)
            print("Matrícula detectada correctamente.")
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- CAMBIA ESTO ---
mi_imagen = "img/sample.png" 
procesar_matricula(mi_imagen)
