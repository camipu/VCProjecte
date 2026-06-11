Aquest repositori conté el codi font i els recursos d'un sistema modular de Reconeixement Automàtic de Matrícules (ANPR/LPR) desenvolupat per a l'assignatura de Visió per Computador.

## 1. Descripció dels fitxers i carpetes

### Fitxers de lògica principal
* **canny.py**: Primera fase del pipeline. Converteix la imatge a escala de grisos, aplica un filtre bilateral i utilitza l'algorisme de Canny com a filtre de vores amb cribratge geomètric per extreure la Regió d'Interès (ROI) de la matrícula.
* **segment_cc.py**: Segona fase del pipeline. Binaritza la ROI extreta (mètode d'Otsu) i aïlla de forma dinàmica cada caràcter utilitzant l'etiquetatge de components connexes.
* **entrenar_ocr.py**: Fase final d'entrenament. Aplica augmentació de dades (multiplicació x11) sobre els caràcters del dataset i entrena el classificador K-Nearest Neighbors (KNN), generant el fitxer `modelo_knn.xml`.
* **ocr_utils.py**: Mòdul centralitzat de suport que conté les funcions i constants compartides (`extraer_caracteristicas`, `binarizar_char`, `segmentar_caracteres`) per garantir la consistència entre l'entrenament i la inferència.

### Fitxers auxiliars i d'avaluació
* **generador_vocals.py**: Script de generació sintètica que utilitza fonts vectorials d'OpenCV per crear les lletres vocals i la Q, resolent el problema del dataset incomplet.
* **extraer_chars_dataset.py**: Script que retalla i extreu caràcters reals directament de les matrícules per alimentar el dataset amb condicions de llum, resolució i soroll del món real.
* **check_ocr.py** / **check.py**: Utilitats d'avaluació que mesuren el rendiment global del pipeline de l'OCR comparant-lo amb el ground truth i check.py de la primera fase de detecció de canny.

### Carpetes i dependències
* **resultats/**: Carpeta destinada a emmagatzemar els retalls de les matrícules i els renders amb el rectangle verd de comprovació.
* **resultats_ocr/**: Carpeta destinada a emmagatzemar els resultats de la etapa de segmentació+ocr.
* **requirements.txt**: Fitxer amb les llibreries i dependències necessàries (OpenCV, NumPy...) per a l'execució del projecte.

## 2. Flux d'execució

```bash
# 1. Instal·lar les dependències del projecte

pip install -r requirements.txt

# 2. Executar la detecció i localització de la placa (Fase 1). Carpeta: resultats/

python canny.py

per a un altre dataset d'imatges pots posar-li quina carpeta vols que agafi amb: 
python canny.py --entrada NOM_CARPETA

# 3. Evaluar el canny

python check.py

# 4. Generar sintèticament les lletres mancants (vocals i Q). Carpeta: LetrasGeneradas

python generador_vocals.py

# 5. Extreure caràcters reals de matrícules per complementar el dataset

python extraer_chars_dataset.py

# 6. Entrenar el model KNN i generar el fitxer 'modelo_knn.xml'

python entrenar_ocr.py

# 7. Executar la segmentació i el reconeixement de caràcters (Fases 2 i 3). Carpeta: resutlats_ocr/

python segment_cc.py

# 8. Avaluar el rendiment i calcular les mètriques d'accuracy globals del sistema
python check_ocr.py

