# src/data_processing.py

import os
import re
import fitz  # PyMuPDF
import traceback
from collections import defaultdict
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
from math import sqrt
# Importamos nuestro archivo de configuración
from . import config

# Importamos la clase Document de LangChain para el formato de salida estándar
from langchain_core.documents import Document

# --- Descarga de NLTK (necesario para el chunking por frases) ---
# CÓDIGO NUEVO Y CORREGIDO
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:  # <--- Este es el cambio clave
    print("Descargando el tokenizador 'punkt' de NLTK por primera vez...")
    nltk.download('punkt', quiet=True)

def clean_pdf_text_robust(text):
    """Limpia texto de PDF de forma MÁS robusta para RAG, atacando patrones específicos."""
    if not text: return ""
    # --- PASOS DE LIMPIEZA GENERAL ---
    ligatures = {'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl'}
    for lig, repl in ligatures.items(): text = text.replace(lig, repl)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text) # Unir palabras con guión
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text) # Segunda pasada
    text = re.sub(r'^\s*Página\s+\d+(\s+de\s+\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE) # Paginación
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text) # Paginación X / Y
    text = re.sub(r'https?://[^\s/$.?#].[^\s]*', '', text, flags=re.IGNORECASE) # URLs http/https
    text = re.sub(r'\bwww\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b(?!\.)', '', text) # URLs www
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text) # Emails

    # --- REGLAS ESPECÍFICAS MEJORADAS ---
    text = re.sub(r'https?://opo\.cl/[a-zA-Z0-9]+', '', text, flags=re.IGNORECASE) # URLs opo.cl
    text = re.sub(r'\bopositatest\.com\b', '', text, flags=re.IGNORECASE) # Dominio específico
    text = re.sub(r'\bv\d+\.\d+\.\d+\b', '', text, flags=re.IGNORECASE) # Versión vX.Y.Z
    text = re.sub(r'/?\s*\+34\s*(\d{1,3}\s*){2,4}', '', text) # Teléfono +34
    text = re.sub(r'^\s*\d+\s+TEMARIO\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*Accede a los recursos.*$', '', text, flags=re.MULTILINE | re.IGNORECASE) # Línea recursos
    text = re.sub(r'^\s*Comprueba si tu temario.*$', '', text, flags=re.MULTILINE | re.IGNORECASE) # Línea actualizado
    text = re.sub(r'^\s*ORGANIZACIÓN DEL ESTADO\s*\|\s*TEMA\s*\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE) # Cabecera específica
    text = re.sub(r'^\s*RECURSOS\s*\n?\s*(GRÁFICOS)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE) # Cabecera Recursos
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) # Líneas solo con número (experimental)
    # Eliminar bloque explicativo iconos (más agresivo)
    text = re.sub(r'^\s*RECURSOS\s+GRÁFICOS.*?simple vistazo\.', '', text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    text = re.sub(r'^\s*PLAZOS\s+Sabemos que.*?simple vistazo\.', '', text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    text = re.sub(r'^\s*(PLAZOS|Destacados|Pregunta de examen|Datos importantes|Negrita)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE) # Títulos sueltos iconos

    # --- PASOS DE NORMALIZACIÓN FINAL ---
    text = re.sub(r'[ \t\f\v]+', ' ', text) # Normalizar espacios horizontales
    text = re.sub(r' +\n', '\n', text) # Espacios antes de salto
    text = re.sub(r'\n +', '\n', text) # Espacios después de salto
    text = re.sub(r'\n{3,}', '\n\n', text) # Reducir saltos múltiples a 2
    text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE) # Eliminar líneas vacías residuales
    text = re.sub(r'^\s*[-•*o»·]\s+', '- ', text, flags=re.MULTILINE) # Normalizar viñetas
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text) # Caracteres de control
    text = text.strip() # Limpiar inicio/fin
    if text: text = text.rstrip('\n') + '\n\n' # Asegurar que termine con dos saltos
    return text

# --- DETECCIÓN PORTADA ---
def is_likely_cover(page_text, page_number, num_total_pages):
    """Heurística para detectar portadas."""
    lines = [line for line in page_text.split('\n') if line.strip()]
    line_count = len(lines)
    text_length = len(page_text.strip())
    if page_number == 0 and (line_count < 15 or text_length < 200):
         return True
    if page_number < 2 and line_count < 25:
        if re.search(r'\b(temario|edición|editorial|reservados todos los derechos|oposici[oó]n)\b', page_text, re.IGNORECASE):
            return True
    return False


def detectar_secciones_matematicas(texto_chunk):
    """
    Detecta menciones a secciones matemáticas típicas CUANDO APARECEN COMO
    ENCABEZADOS/INICIO DE SECCIÓN.

    Retorna un dict **SPARSE**: solo incluye claves para las secciones encontradas.
    Si no encuentra ninguna, retorna un diccionario vacío.
    """
    if not texto_chunk:
        return {} # Retorna dict vacío si no hay texto

    # --- (Copiamos la lógica de patrones y find_section_starts de la v2) ---
    def find_section_starts(pattern, text):
        matches = []
        for match in re.finditer(pattern, text):
            keyword = match.group(1) or ""
            number = match.group(2) or ""
            matches.append((keyword + number).strip())
        return sorted(list(set(matches))) if matches else None

    base_pattern = r'(?:^|\n)\s*({keyword_es}|{keyword_en})(\s+\d+(\.\d+)*)?[:\s]?\b'
    teorema_pattern = base_pattern.format(keyword_es='Teorema|Teo', keyword_en='Theorem')
    definicion_pattern = base_pattern.format(keyword_es='Definici[oó]n|Def', keyword_en='Definition')
    demostracion_pattern = base_pattern.format(keyword_es='Demostraci[oó]n|Dem|Prueba', keyword_en='Proof')
    lema_pattern = base_pattern.format(keyword_es='Lema', keyword_en='Lemma')
    proposicion_pattern = base_pattern.format(keyword_es='Proposici[oó]n|Prop', keyword_en='Proposition')
    corolario_pattern = base_pattern.format(keyword_es='Corolario|Cor', keyword_en='Corollary')
    observacion_pattern = base_pattern.format(keyword_es='Observaci[oó]n|Obs', keyword_en='Observation|Remark')
    nota_pattern = base_pattern.format(keyword_es='Nota', keyword_en='Note')
    # --- (Fin de la copia de patrones) ---

    # Detecta todas las secciones posibles
    detected_sections = {
        "teorema": find_section_starts(teorema_pattern, texto_chunk),
        "definicion": find_section_starts(definicion_pattern, texto_chunk),
        "demostracion": find_section_starts(demostracion_pattern, texto_chunk),
        "lema": find_section_starts(lema_pattern, texto_chunk),
        "proposicion": find_section_starts(proposicion_pattern, texto_chunk),
        "corolario": find_section_starts(corolario_pattern, texto_chunk),
        "observacion": find_section_starts(observacion_pattern, texto_chunk),
        "nota": find_section_starts(nota_pattern, texto_chunk),
    }

    # --- Construye el diccionario de salida dinámicamente ---
    math_sections_sparse = {}
    for section_type, mentions_list in detected_sections.items():
        if mentions_list:  # Solo si se encontró algo para este tipo
            bool_key = f"is_{section_type}"
            list_key = f"{section_type}s_mencionados" # Añade 's' para pluralizar

            # Ajuste manual para plurales irregulares si los hubiera (no aplica aquí)
            # if section_type == '...': list_key = '...'

            math_sections_sparse[bool_key] = True
            math_sections_sparse[list_key] = mentions_list

    return math_sections_sparse # Devuelve el diccionario (posiblemente vacío)

def detectar_paginas_resumen_biblio(pdf_path, max_paginas_finales_a_revisar=10):
    """
    Detecta páginas que contienen RESUMEN o BIBLIOGRAFÍA por separado.
    """
    paginas_resumen = []
    paginas_biblio = []

    resumen_keywords = ['RESUMEN', 'CONCLUSIÓ']
    biblio_keywords = ['BIBLIOGRAFÍA', 'REFERENCIAS', 'WEBGRAFÍA']

    try:
        doc = fitz.open(pdf_path)
        num_total_pages = len(doc)
        start_page_index = max(0, num_total_pages - max_paginas_finales_a_revisar)

        for page_num in range(start_page_index, num_total_pages):
            page = doc.load_page(page_num)
            text = page.get_text("text").upper()

            if not text or text.isspace():
                continue

            # Detectar resumen
            if any(re.search(r'(?:^[ \t]*|\n[ \t]*)' + kw + r'\b', text) for kw in resumen_keywords):
                paginas_resumen.append(page_num)

            # Detectar bibliografía
            if any(re.search(r'(?:^[ \t]*|\n[ \t]*)' + kw + r'\b', text) for kw in biblio_keywords):
                paginas_biblio.append(page_num)

        doc.close()

    except Exception as e:
        print(f"WARN (detectar_resumen_biblio): Error procesando {pdf_path}: {e}")

    return paginas_resumen, paginas_biblio


def detect_image_regions_on_page(
    page: Any,
    merge_close_distance: int = 5,
    min_area: int = 1000,
    detect_drawings: bool = False,
    debug: bool = False
) -> list:
    """
    Detecta regiones probables de imágenes (y opcionalmente dibujos vectoriales)
    en una página de PyMuPDF, retornando bounding boxes fusionadas y filtradas.

    Args:
        page (fitz.Page): Página de PyMuPDF sobre la que se detectan imágenes.
        merge_close_distance (int): Distancia máxima (en puntos) para fusionar
            bounding boxes que se solapan o están muy cerca.
        min_area (int): Área mínima (en puntos^2) para no descartar regiones pequeñas.
        detect_drawings (bool): Si True, intentará detectar regiones vectoriales
            (get_drawings()) y tratarlas como imágenes.
        debug (bool): Si True, muestra mensajes de debug.

    Returns:
        list[dict]: Lista de regiones detectadas, cada una con:
            {
              "bbox": (x0, y0, x1, y1),
              "type": "image" | "drawing"
            }
    """
    all_regions = []
    try:
        # --------------------------------------------------------
        # 1. DETECCIÓN DE IMÁGENES BITMAP
        # --------------------------------------------------------
        images_info = page.get_images(full=True)
        for img_info in images_info:
            xref = img_info[0]
            if xref == 0:
                continue  # ignorar imágenes inline o inválidas
            try:
                # Obtener los rectángulos donde se dibuja esta imagen (puede haber varios)
                img_rects = page.get_image_rects(xref)
                for rect in img_rects:
                    bbox = rect.irect  # (x0, y0, x1, y1) con coords enteras
                    x0, y0, x1, y1 = bbox
                    area = (x1 - x0) * (y1 - y0)
                    if area >= min_area:
                        all_regions.append({"bbox": bbox, "type": "image"})
                    elif debug:
                        print(f"DEBUG: Descartando imagen muy pequeña bbox={bbox}, area={area}")
            except Exception as err_rects:
                if debug:
                    print(f"DEBUG: No se pudo obtener rects de imagen xref={xref}: {err_rects}")

        # --------------------------------------------------------
        # 2. DETECCIÓN DE "DRAWINGS" VECTORIALES (OPCIONAL)
        # --------------------------------------------------------
        if detect_drawings:
            try:
                drawings = page.get_drawings()
                for d in drawings:
                    # 'type' puede ser: 'l' (line), 're' (rectangle),
                    # 'f' (fill?), 'cs' (curves?), etc.
                    # Ajusta según tus necesidades de filtrado.
                    # Aquí descartamos líneas simples:
                    if d['type'] == 'l':
                        continue
                    bbox = d['rect'].irect
                    x0, y0, x1, y1 = bbox
                    area = (x1 - x0) * (y1 - y0)
                    if area >= min_area:
                        all_regions.append({"bbox": bbox, "type": "drawing"})
                    elif debug:
                        print(f"DEBUG: Descartando dibujo pequeño bbox={bbox}, area={area}")
            except Exception as err_draw:
                if debug:
                    print(f"DEBUG: Error detectando dibujos vectoriales: {err_draw}")

        # --------------------------------------------------------
        # 3. FUSIÓN DE BBOXES CERCANOS O SOLAPADOS
        # --------------------------------------------------------
        merged_regions = _merge_bounding_boxes(all_regions, merge_close_distance, debug=debug)

        if debug:
            print(f"DEBUG: detect_image_regions_on_page => {len(all_regions)} sin fusionar, {len(merged_regions)} tras fusión")

        return merged_regions

    except Exception as e:
        print(f"WARN: Error detectando imágenes/dibujos en página: {e}")
        return []
    
def _merge_bounding_boxes(regions: list, close_dist: int, debug: bool = False) -> list:
    """
    Funde bounding boxes que se solapan o están muy cerca, retornando una
    nueva lista de regiones. Cada región es un dict con:
      { "bbox": (x0, y0, x1, y1), "type": "image" / "drawing" }.

    - close_dist: se considerará 'cerca' si la distancia entre 2 rects
      es menor o igual a close_dist.
    - Este método hace un loop iterativo hasta que no haya merges nuevos.

    Return: lista de dicts con bboxes fusionadas.
    """

    # Para fusionar rects, necesitamos una pequeña función de "check solape" y "unión"
    def rects_are_close_or_overlap(r1, r2, threshold):
        """Retorna True si r1 y r2 se solapan o la distancia entre ellos es <= threshold."""
        (x0a, y0a, x1a, y1a) = r1
        (x0b, y0b, x1b, y1b) = r2

        # 1) Si se solapan en x e y (overlap check)
        overlap_x = not (x1a < x0b or x1b < x0a)
        overlap_y = not (y1a < y0b or y1b < y0a)
        if overlap_x and overlap_y:
            return True

        # 2) Si no solapan, calculamos distancia mínima entre los rects
        #    Si es <= threshold, consideramos "cerca".
        dist = _min_dist_between_rects(r1, r2)
        return dist <= threshold

    def merge_rects(r1, r2):
        """Devuelve el bounding box que cubre ambos rects."""
        (x0a, y0a, x1a, y1a) = r1
        (x0b, y0b, x1b, y1b) = r2
        return (
            min(x0a, x0b),
            min(y0a, y0b),
            max(x1a, x1b),
            max(y1a, y1b)
        )

    changed = True
    while changed:
        changed = False
        merged_list = []
        skip_indices = set()
        n = len(regions)

        for i in range(n):
            if i in skip_indices:
                continue
            r1 = regions[i]
            merged = False
            for j in range(i+1, n):
                if j in skip_indices:
                    continue
                r2 = regions[j]

                if r1["type"] == r2["type"] or True:
                    # Si quisieras mantener separado "image" vs "drawing", podrías
                    # fusionar solo si r1["type"] == r2["type"]. O fusionar siempre.
                    if rects_are_close_or_overlap(r1["bbox"], r2["bbox"], close_dist):
                        # Merge them
                        new_bbox = merge_rects(r1["bbox"], r2["bbox"])
                        # Podríamos unificar el 'type'; aquí escogemos la del primero
                        # o creamos algo como "mixed"
                        new_type = r1["type"] if r1["type"] == r2["type"] else "mixed"
                        merged_list.append({"bbox": new_bbox, "type": new_type})
                        skip_indices.add(j)
                        merged = True
                        if debug:
                            print(f"DEBUG: Merged {r1['bbox']} + {r2['bbox']} => {new_bbox}")
                        break
            if not merged:
                # No fusionamos r1 con nadie
                merged_list.append(r1)

        if len(merged_list) < len(regions):
            # Hubo fusión => repetimos
            regions = merged_list
            changed = True
        else:
            # Sin cambio => terminamos
            regions = merged_list

    return regions




def _min_dist_between_rects(r1, r2):
    """
    Calcula la distancia mínima entre dos rects (x0, y0, x1, y1)
    si no se solapan.
    """
    x0a, y0a, x1a, y1a = r1
    x0b, y0b, x1b, y1b = r2

    # Si se solapan en x, la distancia en x es 0; de lo contrario,
    # es la diferencia entre los bordes más cercanos.
    if x1a < x0b:
        dx = x0b - x1a
    elif x1b < x0a:
        dx = x0a - x1b
    else:
        dx = 0

    # Lo mismo para y.
    if y1a < y0b:
        dy = y0b - y1a
    elif y1b < y0a:
        dy = y0a - y1b
    else:
        dy = 0

    # Distancia euclidiana
    return sqrt(dx*dx + dy*dy)



def detectar_paginas_indice(pdf_path, max_paginas_a_revisar=None, umbral_min_lineas=5):
    """
    Intenta detectar las páginas del índice (Tabla de Contenido) en un PDF.

    Utiliza heurísticas mejoradas basadas en patrones de texto comunes,
    combinando numeración jerárquica, palabras clave y opcionalmente
    la presencia de números de página al final de la línea.

    Args:
        pdf_path (str): Ruta al archivo PDF.
        max_paginas_a_revisar (int): Número máximo de páginas iniciales a revisar.
        umbral_min_lineas (int): Mínimo de líneas de texto requeridas en una página
                                 para siquiera considerarla como índice.

    Returns:
        list: Una lista de índices de página (basados en 0) que probablemente
              contienen el índice. Lista vacía si no se detecta ninguno o hay error.
    """
    paginas_indice_detectadas = []
    if not os.path.exists(pdf_path):
         print(f"ERROR: (detectar_paginas_indice) Archivo no encontrado: {pdf_path}")
         return paginas_indice_detectadas

    doc = None
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"WARN: (detectar_paginas_indice) Error al abrir PDF '{pdf_path}': {e}")
        return paginas_indice_detectadas

    # --- Heurísticas ---
    # Regex para numeración como 1., 1.1, 1.1.1., CAPÍTULO 1, TEMA 2, etc. (más flexible)
    patron_numeracion_jerarquica = re.compile(
        r"^\s*([0-9]+(\.[0-9]+)*\.?\s+|"  # 1., 1.1, 1.1.
        r"(CAP[IÍ]TULO|TEMA|SECCI[OÓ]N|PARTE)\s+[0-9IVXLCDM]+\b\.?\s*).*",
        re.IGNORECASE
    )
    # Regex para palabras clave comunes en índices/sumarios (usando search)
    patron_palabras_clave = re.compile(
        r"^\s*(INTRODUCCI[OÓ]N|PR[OÓ]LOGO|CONCLUSI[OÓ]N|EP[IÍ]LOGO|BIBLIOGRAF[IÍ]A|WEBGRAF[IÍ]A|REFERENCIAS|RESUMEN|[IÍ]NDICE|CONTENIDO|SUMARIO|ANEXO|GLOSARIO)\b",
        re.IGNORECASE
    )
    # Regex para líneas que probablemente terminan en un número de página (puede estar precedido por puntos o espacios)
    patron_linea_con_pagina = re.compile(r".*[.\s]\s*(\d+)\s*$")

    num_paginas_a_escanear = min(max_paginas_a_revisar, doc.page_count)
    posible_indice_activo = False # Flag para detectar índices multi-página

    print(f"INFO: Escaneando hasta {num_paginas_a_escanear} páginas para índice en '{os.path.basename(pdf_path)}'")

    for num_pagina in range(num_paginas_a_escanear):
        try:
            pagina = doc.load_page(num_pagina)
            # Usar bloques puede ser un poco más robusto para la separación de líneas
            bloques = pagina.get_text("blocks")
            lineas = []
            for b in bloques:
                # b[4] contiene el texto del bloque, puede tener \n internos
                block_text = b[4]
                # Dividir por nueva línea y limpiar
                lineas.extend(line.strip() for line in block_text.split('\n') if line.strip())

            num_total_lineas = len(lineas)

            # Ignorar páginas casi vacías o portadas detectadas
            if num_total_lineas < umbral_min_lineas or is_likely_cover("\n".join(lineas), num_pagina, doc.page_count):
                # print(f"DEBUG P{num_pagina+1}: Ignorada (líneas={num_total_lineas} < {umbral_min_lineas} or portada)")
                posible_indice_activo = False # Si no es índice, rompe la cadena
                continue

            contador_lineas_patron = 0
            contador_palabras_clave = 0
            contador_lineas_con_pagina = 0

            for linea in lineas:
                if patron_numeracion_jerarquica.match(linea):
                    contador_lineas_patron += 1
                # Usamos search para palabras clave, más flexible a indentación
                if patron_palabras_clave.search(linea):
                    contador_palabras_clave += 1
                if patron_linea_con_pagina.match(linea):
                    # Verificación adicional: asegurarse de que el número no sea parte de la numeración inicial
                    match_num_inicial = patron_numeracion_jerarquica.match(linea)
                    num_final_match = patron_linea_con_pagina.match(linea)
                    if num_final_match:
                         num_final_str = num_final_match.group(1)
                         # Evitar contar si el número final es el mismo que el inicial (p.ej., "1. Título 1")
                         if not (match_num_inicial and linea.strip().endswith(num_final_str) and len(linea.split()) < 4):
                              contador_lineas_con_pagina += 1


            ratio_lineas_patron = contador_lineas_patron / num_total_lineas
            ratio_lineas_con_pagina = contador_lineas_con_pagina / num_total_lineas

            # --- Lógica de Decisión Mejorada ---
            es_pagina_indice = False
            score = 0.0

            # Puntuación base por estructura de numeración (alta importancia)
            score += ratio_lineas_patron * 0.6

            # Puntuación por líneas terminando en número (media importancia)
            score += ratio_lineas_con_pagina * 0.3

            # Bonus por presencia de palabras clave (menor importancia individual, pero ayuda)
            if contador_palabras_clave > 0:
                score += 0.1 # Bonus fijo pequeño si hay al menos una
            if contador_palabras_clave > 2:
                score += 0.1 # Bonus adicional si hay varias

            # Umbral base para considerar índice
            umbral_score_base = 0.25 # Ajustar según sea necesario

            # Umbral más bajo si la página anterior fue índice (continuación)
            umbral_score_continuacion = 0.18

            if posible_indice_activo:
                if score >= umbral_score_continuacion:
                    es_pagina_indice = True
            else:
                 if score >= umbral_score_base:
                    es_pagina_indice = True

            # Refinamiento: Una página con muchas palabras clave pero CERO estructura podría ser un falso positivo
            # O una página con ALTA estructura pero pocas líneas podría no serlo.
            # (La comprobación de umbral_min_lineas ya ayuda con lo segundo)
            if contador_palabras_clave > 1 and contador_lineas_patron == 0 and contador_lineas_con_pagina == 0:
                 # Si SOLO tiene palabras clave y ninguna otra estructura, probablemente no sea índice (podría ser intro/conclusión)
                 # A menos que tenga MUCHAS líneas con palabras clave? Podría ser un índice simple.
                 if num_total_lineas > 10 and (contador_palabras_clave / num_total_lineas > 0.3): # Si >30% de lineas son keywords
                     pass # Probablemente un índice simple basado en keywords, mantener es_pagina_indice si score fue suficiente
                 else:
                     es_pagina_indice = False # Descartar si no cumple la condición anterior


            # DEBUGGING INTERNO
            print(f"  Pág {num_pagina + 1}: Lines={num_total_lineas}, "
                  f"RatioPatron={ratio_lineas_patron:.2f} ({contador_lineas_patron}), "
                  f"Keywords={contador_palabras_clave}, "
                  f"RatioPgNum={ratio_lineas_con_pagina:.2f} ({contador_lineas_con_pagina}), "
                  f"Score={score:.3f} -> Índice? {es_pagina_indice} (ActivoPrev? {posible_indice_activo})")

            if es_pagina_indice:
                if num_pagina not in paginas_indice_detectadas:
                     paginas_indice_detectadas.append(num_pagina)
                posible_indice_activo = True
            else:
                # Si la página no cumple, se rompe la posible cadena de índice
                posible_indice_activo = False

        except Exception as e:
            print(f"WARN: (detectar_paginas_indice) Error procesando página {num_pagina} del PDF: {e}")
            posible_indice_activo = False # Resetear en caso de error
            continue

    if doc:
        doc.close()

    # Post-procesamiento: a veces puede detectar una página suelta entre otras.
    # Si tenemos [0, 2], pero no 1, es menos probable que 2 sea índice.
    # Podríamos requerir bloques contiguos, pero por simplicidad lo dejamos así por ahora.

    print(f"INFO: Páginas de índice detectadas: {[p+1 for p in paginas_indice_detectadas]}") # Mostrar páginas base 1
    return paginas_indice_detectadas

def detect_formulas_in_text(text):
    """
    Detecta fórmulas tipo LaTeX en un texto (cuando existen los delimitadores):
      - Inline: $ ... $
      - Display: $$ ... $$
      - Entorno: \(...\), \[...\]
      - Entornos: \begin{equation}, \begin{align}, etc.

    Retorna lista de dicts con campos:
      {
        "latex_content": str,
        "full_match": str,
        "delimiter_type": str,  # '$', '$$', '\(\)', '\[\]', 'equation', etc.
        "start_pos": int,
        "end_pos": int
      }
    """
    # Para ignorar dólares escapados (\$), reemplazamos temporalmente
    # por algo inofensivo, de modo que no interfieran en la búsqueda.
    text_ignoring_escaped = re.sub(r'\\\$', '__ESCAPED_DOLLAR__', text)

    formulas = []

    # Patrones de LaTeX
    inline_pattern = re.compile(r'\$([^\$]+?)\$', re.DOTALL)
    display_pattern = re.compile(r'\$\$([^\$]+?)\$\$', re.DOTALL)
    backslash_paren_pattern = re.compile(r'\\\((.*?)\\\)', re.DOTALL)
    backslash_bracket_pattern = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
    begin_env_pattern = re.compile(
        r'\\begin\{(equation|align|align\*|gather|gather\*|multline|multline\*)\}(.*?)\\end\{\1\}',
        re.DOTALL
    )

    pattern_list = [
        (display_pattern, lambda m: '$$'),
        (inline_pattern, lambda m: '$'),
        (backslash_paren_pattern, lambda m: r'\(\)'),
        (backslash_bracket_pattern, lambda m: r'\[\]'),
        (begin_env_pattern, lambda m: m.group(1))
    ]

    for pattern, delim_fn in pattern_list:
        for match in pattern.finditer(text_ignoring_escaped):
            content = match.group(1).strip()
            if content:
                formulas.append({
                    "latex_content": content,
                    "full_match": match.group(0),
                    "delimiter_type": delim_fn(match),
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })

    # Ordenar por posición
    formulas.sort(key=lambda f: f['start_pos'])
    return formulas

# ----------------------------------------------------------------
# 2. Detector heurístico para expresiones matemáticas “simples”
#    (cuando NO hay delimitadores LaTeX en el PDF)
# ----------------------------------------------------------------
def detect_heuristic_math_expressions(text):
    """
    Retorna True si 'text' aparenta contener “algo matemático” basado en
    patrones heurísticos MÁS REFINADOS: subíndices, superíndices, símbolos
    griegos comunes, funciones matemáticas conocidas.
    """

    # Regex para subíndices/superíndices: Mantenida igual
    subsuper_regex = re.compile(
        r'[A-Za-z0-9]\s*(\^|_)\s*(\{[^}]+\}|\w+|\d+)',
    )

    # Regex para caracteres matemáticos comunes: Mantenida igual
    # (Podrías añadir más si es necesario: ∈∉⊂⊃⊆⊇∧∨¬∀∃...)
    math_symbols_regex = re.compile(
        r'[∂∑∏√∞±≤≥≈≠÷⋅→←⇒⇔∆∫∑−∂∇αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]'
    )

    # Regex REFINADA para funciones con paréntesis:
    # - Busca funciones matemáticas comunes (sin, cos, log, exp, etc.)
    # - O busca el patrón simple f(x), g(y, z), etc. (letra seguida de paréntesis
    #   con variables/números simples dentro)
    # - Ignora mayúsculas/minúsculas para los nombres de función
    # - Es menos probable que coincida con texto descriptivo como "específicas(..)"
    common_math_funcs = r'\b(sin|cos|tan|sec|csc|cot|log|ln|exp|sqrt|abs|max|min|det|dim|ker|im|gcd|lcm|mod)\b'
    simple_func_call = r'\b[a-zA-Z]\s*\(\s*[a-zA-Z0-9_,\s]+?\s*\)' # Ej: f(x), g(a, b), h(t_1)

    function_call_regex = re.compile(f'({common_math_funcs}\s*\(.*?\)|{simple_func_call})', re.IGNORECASE)


    # Verifica si alguno de los patrones se cumple
    if subsuper_regex.search(text):
        # print("DEBUG: Found sub/superscript") # Debug opcional
        return True
    if math_symbols_regex.search(text):
        # print("DEBUG: Found math symbol") # Debug opcional
        return True
    if function_call_regex.search(text):
        # print(f"DEBUG: Found potential function call: {function_call_regex.search(text).group()}") # Debug opcional
        return True

    return False

def extract_and_clean_pdf_smart(pdf_path,
                                     use_ocr_threshold=50,
                                     language='spa', # 'language' no se usa directamente aquí, quizás en OCR
                                     max_index_pages_to_scan=15,
                                     max_summary_biblio_pages_to_scan=10,
                                     debug_prints=False): # Añadido parámetro debug_prints
    """
    Extrae texto de PDF, detecta fórmulas/imágenes,
    omite portadas/índices, limpia texto, y elimina solo el bloque de
    bibliografía sin eliminar páginas enteras.

    - Aplica detect_formulas_in_text() para ver si hay LaTeX literal.
    - Marca si el texto parece contener expresiones matemáticas (heurística).
    - Detecta secciones matemáticas (teorema, definición, demostración, etc.)
      a nivel global.
    """
    all_formulas_detected = []
    all_image_regions = {}
    valid_pages_text = []
    omitted_pages_info = []
    # debug_prints = False # Se recibe como parámetro ahora

    try:
        if not os.path.exists(pdf_path):
            print(f"Error GRAVE: No se encontró el archivo PDF: {pdf_path}")
            # Devolver None o un dict vacío con error es mejor que solo None
            return {"error": f"File not found: {pdf_path}", "cleaned_text": "", "detected_formulas": [], "detected_image_regions": {}, "omitted_pages": [], "heuristic_math_detected": False, "detected_math_sections": {}}


        pdf_basename = os.path.basename(pdf_path)

        # --- Pre-detección de ÍNDICE ---
        if debug_prints: print(f"DEBUG ({pdf_basename}): Pre-detectando páginas de índice (hasta {max_index_pages_to_scan} págs)...")
        paginas_indice_detectadas = detectar_paginas_indice(
            pdf_path, max_paginas_a_revisar=max_index_pages_to_scan
        )
        if paginas_indice_detectadas:
            if debug_prints: print(f"DEBUG ({pdf_basename}): Posibles páginas de ÍNDICE (0-based): {paginas_indice_detectadas}")

        # --- Pre-detección de RESUMEN/BIBLIO ---
        if debug_prints: print(f"DEBUG ({pdf_basename}): Pre-detectando págs Resumen/Biblio (últimas {max_summary_biblio_pages_to_scan})...")
        paginas_resumen_detectadas, paginas_biblio_detectadas = detectar_paginas_resumen_biblio(
            pdf_path, max_paginas_finales_a_revisar=max_summary_biblio_pages_to_scan
        )
        if paginas_resumen_detectadas:
            if debug_prints: print(f"DEBUG ({pdf_basename}): Páginas con posible RESUMEN (0-based): {paginas_resumen_detectadas}")
        if paginas_biblio_detectadas:
            if debug_prints: print(f"DEBUG ({pdf_basename}): Páginas con posible BIBLIOGRAFÍA (0-based): {paginas_biblio_detectadas}")

        # --- Crear conjunto de páginas a omitir solo para índice ---
        # Convertir a 0-based si las funciones de detección devuelven 1-based
        # Asumiendo que devuelven 0-based:
        paginas_a_omitir_previamente = set(paginas_indice_detectadas)

        doc = fitz.open(pdf_path)
        num_total_pages = len(doc)
        if debug_prints: print(f"DEBUG ({pdf_basename}): Procesando {num_total_pages} páginas (Modo STEM).")

        for page_num in range(num_total_pages):
            page_num_real = page_num + 1 # Para logs y referencias (1-based)
            if debug_prints: print(f"DEBUG ({pdf_basename}): Procesando pág {page_num_real}/{num_total_pages}...")

            # --- Omisión de páginas índice ---
            if page_num in paginas_a_omitir_previamente:
                reason = "Índice (pre-detectado)"
                omitted_pages_info.append((page_num_real, reason))
                if debug_prints:
                    print(f"  -> OMITIDA ({reason}).")
                continue

            # --- Extracción básica de texto ---
            page_raw_text = ""
            page_raw_text_strip = ""
            try:
                page = doc.load_page(page_num) # Cargar página dentro del bucle
                page_raw_text = page.get_text("text", sort=True)
                page_raw_text_strip = page_raw_text.strip() if page_raw_text else ""
                if debug_prints and not page_raw_text_strip:
                    print(f"  -> WARN: get_text devolvió vacío o solo espacios.")
            except Exception as getTextErr:
                print(f"WARN ({pdf_basename}): get_text falló pág {page_num_real}: {getTextErr}.")
                omitted_pages_info.append((page_num_real, f"Error get_text: {getTextErr}"))
                continue # Saltar página si falla la extracción básica

            # --- Lógica de OCR (Opcional, si se requiere) ---
            used_ocr = False
            if not page_raw_text_strip or len(page_raw_text_strip) < use_ocr_threshold:
                # Aquí iría la llamada a una función OCR si decides implementarla
                # page_raw_text_ocr = apply_ocr_to_page(page, language=language)
                # if page_raw_text_ocr and len(page_raw_text_ocr.strip()) > len(page_raw_text_strip):
                #     page_raw_text = page_raw_text_ocr
                #     page_raw_text_strip = page_raw_text.strip()
                #     used_ocr = True
                #     if debug_prints: print(f"  -> INFO: OCR aplicado (resultado > {use_ocr_threshold} chars).")
                # elif debug_prints:
                #     print(f"  -> INFO: Texto < {use_ocr_threshold} chars, OCR no aplicado o sin mejora.")
                pass # Placeholder para OCR

            # --- Comprobación de texto vacío (Post-OCR si aplica) ---
            if not page_raw_text_strip:
                reason = "Sin texto válido (post-OCR)" if used_ocr else "Sin texto válido"
                omitted_pages_info.append((page_num_real, reason))
                if debug_prints: print(f"  -> OMITIDA ({reason}).")
                continue

            # --- Heurística de portada ---
            if is_likely_cover(page_raw_text_strip, page_num, num_total_pages):
                reason = "Portada (heurística)"
                omitted_pages_info.append((page_num_real, reason))
                if debug_prints: print(f"  -> OMITIDA ({reason}).")
                continue

            # --- Cortar bibliografía si corresponde (Solo si la página fue pre-detectada) ---
            final_page_text = page_raw_text_strip # Usar texto strip para búsqueda
            original_length_before_bib_cut = len(final_page_text)
            bibliography_cut_applied = False

            if page_num in paginas_biblio_detectadas:
                # Buscar keywords en mayúsculas para robustez, pero cortar el original
                biblio_keywords_regex = r'^\s*(BIBLIOGRAFÍA|REFERENCIAS|WEBGRAFÍA)\s*$' # Más específico, inicio de línea
                # Intentar buscar desde el final de la página hacia atrás podría ser más robusto
                lines = final_page_text.splitlines()
                cut_index = -1
                for i in range(len(lines) - 1, -1, -1):
                     if re.search(biblio_keywords_regex, lines[i].strip().upper()):
                         # Encontrar la posición de inicio de esta línea en el texto original
                         try:
                            cut_index = final_page_text.rindex(lines[i])
                            break
                         except ValueError:
                            pass # Seguir buscando si la línea no se encuentra exactamente

                if cut_index != -1:
                    final_page_text = final_page_text[:cut_index].strip()
                    bibliography_cut_applied = True
                    if debug_prints:
                        print(f"  -> INFO: Texto cortado por keyword de bibliografía encontrada.")

                    # Si tras cortar no queda texto, omitir la página
                    if not final_page_text:
                        reason = "Texto eliminado por contenido de bibliografía"
                        omitted_pages_info.append((page_num_real, reason))
                        if debug_prints: print(f"  -> OMITIDA ({reason}).")
                        continue

            # --- Detección de fórmulas e imágenes (en el texto final de la página) ---
            page_formulas = detect_formulas_in_text(final_page_text) # Usar texto potencialmente cortado
            if page_formulas:
                # Ajustar posiciones si se cortó la bibliografía? No, las pos son relativas a final_page_text
                all_formulas_detected.extend(page_formulas)
                if debug_prints: print(f"  -> INFO: Detectadas {len(page_formulas)} fórmulas LaTeX.")

            # Detectar imágenes (asumiendo que la función existe)
            page_image_bboxes = detect_image_regions_on_page(page)
            if page_image_bboxes:
                all_image_regions[page_num_real] = page_image_bboxes
                if debug_prints: print(f"  -> INFO: Detectadas {len(page_image_bboxes)} regiones de imagen.")

            # --- Página Aceptada (añadir texto final) ---
            valid_pages_text.append(final_page_text)
            if debug_prints: print(f"  -> ACEPTADA (len: {len(final_page_text)} chars).")

        # --- Fin del bucle de páginas ---
        doc.close()

        # --- Resumen de omisiones ---
        print("\n" + "-"*20 + f" Resumen Omisiones ({pdf_basename}) " + "-"*20)
        if not omitted_pages_info:
            print("INFO: No se omitió ninguna página.")
        else:
            omitted_by_reason = defaultdict(list)
            for page, reason in omitted_pages_info:
                omitted_by_reason[reason].append(page)
            print(f"INFO: Omitidas {len(omitted_pages_info)}/{num_total_pages} páginas:")
            for reason, pages in sorted(omitted_by_reason.items()):
                pages.sort()
                # Agrupar páginas consecutivas para mejor lectura
                grouped_pages = []
                if pages:
                    start_range = pages[0]
                    end_range = pages[0]
                    for i in range(1, len(pages)):
                        if pages[i] == end_range + 1:
                            end_range = pages[i]
                        else:
                            if start_range == end_range:
                                grouped_pages.append(str(start_range))
                            else:
                                grouped_pages.append(f"{start_range}-{end_range}")
                            start_range = end_range = pages[i]
                    # Añadir el último rango/página
                    if start_range == end_range:
                        grouped_pages.append(str(start_range))
                    else:
                        grouped_pages.append(f"{start_range}-{end_range}")
                print(f"  - Razón: '{reason}', Páginas: {', '.join(grouped_pages)}")
        print("-"*(42 + len(f" Resumen Omisiones ({pdf_basename}) ")))

        # --- Limpieza final del texto concatenado ---
        if not valid_pages_text:
            print(f"ERROR ({pdf_basename}): No se aceptó ninguna página válida.")
            return {
                "cleaned_text": "",
                "detected_formulas": [],
                "detected_image_regions": {},
                "omitted_pages": omitted_pages_info,
                "heuristic_math_detected": False,
                "detected_math_sections": {}
            }

        full_raw_text = "\n\n".join(valid_pages_text) # Unir páginas aceptadas
        if debug_prints: print(f"DEBUG ({pdf_basename}): {len(valid_pages_text)} págs aceptadas. Limpiando texto concatenado...")

        # Asumiendo que clean_pdf_text_robust está definido.
        cleaned_text = clean_pdf_text_robust(full_raw_text)
        if debug_prints: print(f"DEBUG ({pdf_basename}): Limpieza completada. Longitud final: {len(cleaned_text)} chars.")

        if not cleaned_text or cleaned_text.isspace():
            print(f"WARN ({pdf_basename}): Texto final limpio vacío.")
            # Devolver el estado aunque el texto esté vacío
            return {
                "cleaned_text": "",
                "detected_formulas": all_formulas_detected,
                "detected_image_regions": all_image_regions,
                "omitted_pages": omitted_pages_info,
                "heuristic_math_detected": False, # No hay texto para analizar
                "detected_math_sections": {}
            }

        # --- Análisis Global del Texto Limpio ---
        # Heurística global de fórmulas en el texto
        has_any_heuristic_math = detect_heuristic_math_expressions(cleaned_text)
        if debug_prints: print(f"DEBUG ({pdf_basename}): Heurística matemática global detectada: {has_any_heuristic_math}")

        # Detección de secciones matemáticas (teorema, definición, etc.)
        math_sections_info = detectar_secciones_matematicas(cleaned_text)
        if debug_prints: print(f"DEBUG ({pdf_basename}): Secciones matemáticas detectadas: {math_sections_info}")


        # --- Devolver Resultados ---
        return {
            "cleaned_text": cleaned_text,
            "detected_formulas": all_formulas_detected,       # LaTeX literal
            "detected_image_regions": all_image_regions,
            "omitted_pages": omitted_pages_info,
            "heuristic_math_detected": has_any_heuristic_math,  # True/False si hay expresiones "sospechosas"
            "detected_math_sections": math_sections_info        # Dict con las listas y booleanos
        }

    except FileNotFoundError:
 # Ser más específico con la excepción
        print(f"Error GRAVE: No se encontró el PDF: {pdf_path}")
        return {"error": f"File not found: {pdf_path}", "cleaned_text": "", "detected_formulas": [], "detected_image_regions": {}, "omitted_pages": [], "heuristic_math_detected": False, "detected_math_sections": {}}
    except Exception as e:
        print(f"Error GRAVE procesando PDF {os.path.basename(pdf_path)} ({type(e).__name__}): {e}")
        traceback.print_exc()
        return {"error": f"Processing error: {e}", "cleaned_text": "", "detected_formulas": [], "detected_image_regions": {}, "omitted_pages": [], "heuristic_math_detected": False, "detected_math_sections": {}}
    




def chunk_text_by_sentences(text, max_words=200, overlap_sentences=1):
    """Divide el texto en chunks basados en frases, con límite de palabras y solapamiento."""
    if not text: return []
    overlap_sentences = max(0, overlap_sentences)
    try:
        sentences = sent_tokenize(text, language='spanish')
        if not sentences: # Si sent_tokenize devuelve vacío, usar fallback
             raise ValueError("sent_tokenize devolvió lista vacía")
    except Exception as e:
        # print(f"DEBUG: sent_tokenize falló o vacío ({e}), usando fallback por saltos de línea.")
        # Fallback: dividir por dobles saltos de línea, luego simples
        sentences = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not sentences:
             sentences = [p.strip() for p in text.split('\n') if p.strip()]
        # Fallback final: dividir por puntos si todo lo demás falla
        if not sentences:
             sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

    chunks = []
    i = 0
    n_sentences = len(sentences)

    while i < n_sentences:
        current_chunk_sentences = []
        word_count = 0
        start_sentence_index = i

        j = i
        while j < n_sentences:
            current_sentence = sentences[j].strip()
            if not current_sentence:
                j += 1; continue

            sentence_words = len(current_sentence.split())

            # Añadir la primera frase siempre, o si cabe
            if not current_chunk_sentences or (word_count + sentence_words <= max_words):
                current_chunk_sentences.append(current_sentence)
                word_count += sentence_words
                j += 1
            else:
                break # La frase actual no cabe

        # Procesar el chunk acumulado
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            # Calcular índice de inicio del siguiente chunk con solapamiento
            # Retroceder `overlap_sentences` desde `j` (la frase que no cupo)
            # pero sin ir más atrás que `start_sentence_index + 1`
            next_start_index = max(start_sentence_index + 1, j - overlap_sentences)
            i = min(next_start_index, n_sentences) # Asegurar que no se pase del final
        # Manejar caso de frase inicial demasiado larga
        elif j < n_sentences and j == i: # No se avanzó, la primera frase ya era > max_words
            long_sentence = sentences[j].strip()
            if long_sentence:
                 chunks.append(long_sentence)
                 # print(f"DEBUG: Frase inicial larga (idx {j}) añadida como chunk.")
            i = j + 1 # Avanzar al siguiente
        # Caso raro: no se añadieron frases y no era la inicial (ej, bucle infinito?)
        elif j < n_sentences:
             # print(f"DEBUG: Avanzando índice de {i} a {j} sin añadir chunk.")
             i = j
        else: # Fin de las frases
            break
    return chunks



# ==============================================================================
#  NUEVA FUNCIÓN "PÚBLICA" DE ENCAPSULAMIENTO
# ==============================================================================

def load_and_process_pdf(file_path: str) -> List[Document]:
    """
    Punto de entrada principal del módulo.
    Carga, extrae, limpia y trocea un PDF, devolviendo una lista de Documentos LangChain.
    """
    print("--- Iniciando pipeline de procesamiento de datos ---")
    
    # 1. Extraer y limpiar el texto del PDF usando tu lógica avanzada
    extraction_result = extract_and_clean_pdf_smart(file_path, debug_prints=False)
    cleaned_text = extraction_result.get("cleaned_text")
    
    if not cleaned_text:
        print("ADVERTENCIA: No se pudo extraer texto del documento. No se generarán chunks.")
        return []

    # 2. Dividir el texto limpio en chunks usando tu lógica de frases
    #    (Usamos parámetros de config.py, aunque podrías añadirlos aquí)
    text_chunks = chunk_text_by_sentences(
        cleaned_text,
        max_words=250, # Podrías mover esto a config.py
        overlap_sentences=2 # Y esto también
    )
    
    # 3. Convertir los chunks de texto en objetos Document de LangChain
    #    (Aquí es donde podrías añadir los metadatos de fórmulas, etc. a cada chunk)
    documents = []
    for i, chunk_text in enumerate(text_chunks):
        # Por ahora, metadatos simples. Aquí puedes integrar tu lógica de metadatos.
        metadata = {
            "source": os.path.basename(file_path),
            "chunk_index": i
        }
        doc = Document(page_content=chunk_text, metadata=metadata)
        documents.append(doc)

    print(f"--- Pipeline completado. Se generaron {len(documents)} documentos (chunks). ---")
    return documents


# ==============================================================================
#  BLOQUE DE PRUEBA (SOLO SE EJECUTA AL CORRER ESTE ARCHIVO DIRECTAMENTE)
# ==============================================================================

if __name__ == '__main__':
    print("--- Ejecutando data_processing.py como un script de prueba ---")
    
    # Usamos la ruta del PDF desde nuestro archivo de configuración centralizado
    pdf_path_to_test = config.PDF_SOURCE_PATH
    
    if not os.path.exists(pdf_path_to_test):
        print(f"ERROR DE PRUEBA: El archivo no se encuentra en '{pdf_path_to_test}'")
        print("Asegúrate de que la ruta en 'src/config.py' es correcta y el archivo existe.")
    else:
        # Llamamos a nuestra nueva función principal
        processed_documents = load_and_process_pdf(pdf_path_to_test)
        
        if processed_documents:
            print(f"\nSe han procesado {len(processed_documents)} documentos.")
            print("\n--- Ejemplo del primer documento procesado ---")
            print("Contenido (primeros 300 caracteres):")
            print(processed_documents[0].page_content[:300] + "...")
            print("\nMetadatos:")
            print(processed_documents[0].metadata)
            
            print("\n--- Ejemplo del último documento procesado ---")
            print("Contenido (primeros 300 caracteres):")
            print(processed_documents[-1].page_content[:300] + "...")
            print("\nMetadatos:")
            print(processed_documents[-1].metadata)
        else:
            print("La prueba finalizó pero no se generaron documentos.")