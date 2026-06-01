import cv2
import numpy as np
import os

letras_nuevas = ['A', 'E', 'I', 'O', 'U', 'Q']
output_dir = "LetrasGeneradas"

# Fuentes clásicas de OpenCV
fuentes = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX
]

for letra in letras_nuevas:
    carpeta = os.path.join(output_dir, letra)
    os.makedirs(carpeta, exist_ok=True)
    
    # Generamos 4 variantes cambiando la fuente y el grosor
    for idx, fuente in enumerate(fuentes, start=1):
        # Crear un lienzo en blanco (fondo blanco, 100x100 para tener margen)
        img = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Calcular el tamaño del texto para centrarlo
        grosor = 2 if idx % 2 == 0 else 3
        escala = 1.8
        (txt_w, txt_h), _ = cv2.getTextSize(letra, fuente, escala, grosor)
        
        x = (100 - txt_w) // 2
        y = (100 + txt_h) // 2
        
        # Pintar la letra en negro (0) para calcar tu dataset original
        cv2.putText(img, letra, (x, y), fuente, escala, 0, grosor, cv2.LINE_AA)
        
        # Guardar en el formato .tif que usan tus otros archivos
        nombre_archivo = f"{letra}_{idx}.tif"
        cv2.imwrite(os.path.join(carpeta, nombre_archivo), img)

print("¡Letras generadas correctamente con tipografías de OpenCV!")