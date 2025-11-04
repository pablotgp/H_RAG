




# src/retrieval.py (Versión Dinámica Multi-Documento Completa)

import os
import faiss
import pickle
import numpy as np
import traceback
import re
from typing import List

# --- Importaciones de Terceros ---
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- Importaciones de nuestros módulos ---
from src import config
from src.data_processing import load_and_process_pdf
from src.utils import calculate_checksum

# ==============================================================================
#  Funciones Auxiliares para Rutas Dinámicas
# ==============================================================================

def get_index_dir_for_pdf(pdf_path: str) -> str:
    """
    Genera una ruta de directorio de índice única para un PDF específico.
    Ejemplo: 'data/Biologia.pdf' -> 'index/Biologia_index'
    """
    pdf_filename = os.path.basename(pdf_path)
    # Crea un nombre de carpeta seguro a partir del nombre del archivo
    dir_name = os.path.splitext(pdf_filename)[0].replace(" ", "_").replace("-", "_") + "_index"
    return os.path.join(config.INDEX_PARENT_DIR, dir_name)

# ==============================================================================
#  Construcción del Índice (Ahora es Dinámico)
# ==============================================================================

def build_index(pdf_path: str):
    pdf_filename = os.path.basename(pdf_path)
    print(f"--- Iniciando la construcción de índices para: {pdf_filename} ---")
    
    index_dir = get_index_dir_for_pdf(pdf_path)
    os.makedirs(index_dir, exist_ok=True)
    
    documents = load_and_process_pdf(pdf_path)
    if not documents:
        raise ValueError("El procesamiento del PDF no generó documentos.")
        
    chunks = [doc.page_content for doc in documents]
    metadata = [doc.metadata for doc in documents]
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    print(f"Generando embeddings para {len(chunks)} chunks...")
    index = faiss.IndexFlatL2(1536)
    BATCH_SIZE = 100
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_texts = chunks[i : i + BATCH_SIZE]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch_texts)
        batch_embeddings = [d.embedding for d in response.data]
        index.add(np.array(batch_embeddings, dtype=np.float32))
        print(f"  Batch {i//BATCH_SIZE + 1} procesado.")

    # ===> CORRECCIÓN CLAVE: Usamos 'index_dir' para todas las rutas <===
    faiss_index_path = os.path.join(index_dir, "faiss_index")
    texts_path = os.path.join(index_dir, "texts.pkl")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    bm25_index_path = os.path.join(index_dir, "bm25_index.pkl")

    faiss.write_index(index, faiss_index_path)
    with open(texts_path, "wb") as f: pickle.dump(chunks, f)
    with open(metadata_path, "wb") as f: pickle.dump(metadata, f)

    print(f"Índice FAISS guardado en '{faiss_index_path}'.")
    
    tokenized_docs = [doc.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_docs)
    with open(bm25_index_path, "wb") as f: pickle.dump(bm25, f)
    print(f"Índice BM25 guardado en '{bm25_index_path}'.")

    checksum = calculate_checksum(pdf_path)
    if checksum:
        checksum_path = os.path.join(index_dir, "index.checksum")
        with open(checksum_path, "w") as f: f.write(checksum)
        print(f"Huella digital guardada: {checksum[:10]}...")
        
    print("--- Construcción de todos los índices completada. ---")

# ==============================================================================
#  Búsqueda Híbrida (Dinámica)
# ==============================================================================

_retriever_cache = {}

def _initialize_retriever_components(index_dir: str):
    if index_dir in _retriever_cache:
        return

    print(f"INFO: Inicializando componentes del retriever para el índice: {index_dir}")
    try:
        components = {}
        
        # ===> CORRECCIÓN CLAVE: Usamos 'index_dir' para cargar todo <===
        faiss_path = os.path.join(index_dir, "faiss_index")
        texts_path = os.path.join(index_dir, "texts.pkl")
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")
        
        components["client"] = OpenAI(api_key=config.OPENAI_API_KEY)
        components["faiss_index"] = faiss.read_index(faiss_path)
        with open(texts_path, "rb") as f: components["texts"] = pickle.load(f)
        with open(metadata_path, "rb") as f: components["metadata"] = pickle.load(f)
        with open(bm25_path, "rb") as f: components["bm25"] = pickle.load(f)
        components["reranker"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        _retriever_cache[index_dir] = components
        print(f"INFO: Componentes para '{index_dir}' inicializados y cacheados.")
    except Exception as e:
        print(f"ERROR FATAL inicializando el retriever para '{index_dir}': {e}")
        raise RuntimeError(f"Fallo al inicializar para '{index_dir}'.") from e

# --- Funciones Auxiliares para la Búsqueda ---

def simple_tokenizer(text):
    if not isinstance(text, str): return []
    return text.lower().split()

def norm_score(score, min_val, max_val):
    if min_val == max_val: return 1.0 if score > 0 else 0.0
    return (score - min_val) / (max_val - min_val)

def print_pesos_info(razon_principal, detalles_razon, peso_bm25, peso_emb):
    print(f"  INFO DinamicWeights: Razón Principal = {razon_principal}")
    if detalles_razon:
        for detalle in detalles_razon: print(f"    - {detalle}")
    print(f"  INFO DinamicWeights: Pesos Asignados -> BM25={peso_bm25:.2f}, Embedding={peso_emb:.2f}")

def calcular_pesos_dinamicos(query: str, subject: str = None) -> tuple[float, float]:
    """
    Analiza la query educativa y el tema (opcional) y ajusta pesos entre BM25 y Embeddings.
    Devuelve (peso_bm25, peso_emb).
    """
    query_lower = query.lower()
    query_original = query # Para checks de mayúsculas

    # --- Pesos Base ---
    peso_bm25 = 0.4
    peso_emb = 0.6
    razon_principal = "Default (ligero sesgo Embedding)"
    detalles_razon = []

    # --- 1. Indicadores de ALTA ESPECIFICIDAD (Prioridad Alta para BM25) ---

    # 1.1. Citas exactas (texto entre comillas)
    if re.search(r'"[^"]+"', query_original): # Busca texto entre comillas dobles
        peso_bm25 = 0.85
        peso_emb = 0.15
        razon_principal = "Cita Exacta"
        detalles_razon.append("BM25 priorizado para coincidencia literal.")
        print_pesos_info(razon_principal, detalles_razon, peso_bm25, peso_emb)
        return peso_bm25, peso_emb

    # 1.bis. Definición de Término Clave Específico (Ej: "elipsis", "hipérbaton")
    definicion_keywords_specific_term = [
        "define", "definición de", "definir", "significa",
        "qué es", "que es", "cuál es el significado de",
        "concepto de"
    ]
    term_to_define_specific = ""
    for keyword in definicion_keywords_specific_term:
        # Patrón para "keyword X" o "keyword 'X'" o "keyword "X""
        # o para "X keyword" (menos común para estas keywords pero podría pasar)
        # Priorizamos "keyword X"
        if query_lower.startswith(keyword + " "):
            potential_term = query_lower[len(keyword)+1:].strip()
            # Quitar comillas y signos de interrogación del término
            potential_term = re.sub(r"['\"?¿!¡]$", "", potential_term).strip()
            potential_term = re.sub(r"^['\"]", "", potential_term).strip()

            # Si la query original tenía el término entre comillas, es buena señal
            if f"'{potential_term}'" in query_original or f'"{potential_term}"' in query_original:
                 term_to_define_specific = potential_term
                 break
            # Si no, tomarlo si es corto
            elif len(potential_term.split()) <= 3:
                 term_to_define_specific = potential_term
                 break

    if term_to_define_specific and len(term_to_define_specific.split()) <= 3 and len(query.split()) < 8 : # Término corto, query no demasiado larga
        # Evitar que una pregunta conceptual larga que casualmente empieza con "qué es la vida..." caiga aquí
        # Si la query es más larga, es probable que sea más conceptual.
        peso_bm25 = 0.80 # Alta prioridad para BM25 para encontrar el término exacto
        peso_emb = 0.20
        razon_principal = "Definición de Término Clave Específico"
        detalles_razon.append(f"Término detectado: '{term_to_define_specific}'. BM25 fuertemente priorizado.")
        print_pesos_info(razon_principal, detalles_razon, peso_bm25, peso_emb)
        return peso_bm25, peso_emb


    # 1.2. Búsqueda de Leyes, Artículos, Teoremas específicos
    if re.search(r'\b(ley|artículo|teorema|postulado|axioma|principio)\s+([0-9]+|[xviíclmd]+|[A-Za-z\s]+)\b', query_lower, re.IGNORECASE):
        peso_bm25 = 0.75
        peso_emb = 0.25
        razon_principal = "Ley/Artículo/Teorema Específico"
        detalles_razon.append("BM25 priorizado para identificadores exactos.")
        print_pesos_info(razon_principal, detalles_razon, peso_bm25, peso_emb)
        return peso_bm25, peso_emb

    # 1.3. Fórmulas o Ecuaciones
    if re.search(r'\b[a-zA-Z]\s*=\s*[a-zA-Z0-9]|\b[a-zA-Z]\w*\([a-zA-Z\d,\s]*\)|[a-zA-Z]\w*_[a-zA-Z\d]|\w\^[2-9]\b', query_original):
        if subject in ["Física", "Biología", "Matemáticas", "Química"]: # Más probable que sea una fórmula
            peso_bm25 = 0.70
            peso_emb = 0.30
            razon_principal = "Posible Fórmula/Ecuación"
            detalles_razon.append(f"BM25 priorizado en {subject} para coincidencia estructural.")
            print_pesos_info(razon_principal, detalles_razon, peso_bm25, peso_emb)
            return peso_bm25, peso_emb

    # --- 2. Indicadores de ESPECIFICIDAD MEDIA (Favorecen BM25, pero con espacio para semántica) ---

    # 2.1. Nombres Propios
    nombres_propios_candidatos = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{1,})*\b', query_original)
    if nombres_propios_candidatos:
        if not (len(nombres_propios_candidatos) == 1 and query_original.startswith(nombres_propios_candidatos[0]) and len(query.split()) > 3):
            peso_bm25 = max(peso_bm25, 0.65) # Aumenta si el default era menor, o lo establece
            peso_emb = 1.0 - peso_bm25
            if razon_principal.startswith("Default"): razon_principal = "Nombre Propio Detectado"
            detalles_razon.append(f"Candidatos NP: {nombres_propios_candidatos}. BM25 priorizado.")

    # 2.2. Fechas, Años, Siglos
    if re.search(r'\b\d{3,4}\b', query_lower) or \
       re.search(r'\bsiglo\s+(?:[xviíclmd]+|[0-9]+)\b', query_lower) or \
       re.search(r'\b(año|fecha)\s+\d{1,4}\b', query_lower) or \
       re.search(r'\b\d{1,2}(?:/| de |-| del )\w+(?:/| de |-| del )\d{2,4}\b', query_lower):
        peso_bm25 = max(peso_bm25, 0.70)
        peso_emb = 1.0 - peso_bm25
        if razon_principal.startswith("Default") or "Nombre Propio" in razon_principal: razon_principal = "Fecha/Año/Siglo Detectado"
        detalles_razon.append("BM25 priorizado para especificidad temporal.")
        if subject == "Historia":
            peso_bm25 = max(peso_bm25, 0.75) # Aún más para Historia
            peso_emb = 1.0 - peso_bm25
            detalles_razon.append("Alta prioridad BM25 en Historia.")

    # 2.3. Acrónimos y Términos Técnicos Muy Específicos
    acronimos_candidatos = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{2,}\b', query_original)
    if acronimos_candidatos and not query_original.isupper():
        if not (len(acronimos_candidatos) == 1 and query_original.startswith(acronimos_candidatos[0])):
            peso_bm25 = max(peso_bm25, 0.60)
            peso_emb = 1.0 - peso_bm25
            if razon_principal.startswith("Default") or "Nombre Propio" in razon_principal or "Fecha" in razon_principal:
                razon_principal = "Acrónimo/Término Técnico Específico Detectado"
            detalles_razon.append(f"Candidatos Acrónimo: {acronimos_candidatos}. BM25 con peso incrementado.")


    # --- 3. Indicadores de BÚSQUEDA DE DEFINICIONES (Equilibrio, si no es ya muy específico) ---
    # Esta regla se aplica si las de ALTA ESPECIFICIDAD (incluida 1.bis) no se activaron y retornaron.
    definicion_keywords_general = ["define", "definición de", "definir", "significa", "concepto de"]
    que_es_keywords_general = ["qué es", "que es", "cual es el significado de", "cuál es el significado de"]

    is_general_definition_request = False
    if any(keyword in query_lower for keyword in definicion_keywords_general) or \
       any(query_lower.startswith(keyword) for keyword in que_es_keywords_general):
        is_general_definition_request = True

    if is_general_definition_request:
        # Si ya se marcó como muy específico (nombre propio, fecha, acrónimo), mantenemos BM25 alto,
        # pero si la razón principal aún es "Default" o algo menos específico.
        if peso_bm25 < 0.6: # Solo ajusta si no es ya específico por reglas anteriores
            peso_bm25 = 0.55
            peso_emb = 0.45
            razon_principal = "Petición de Definición General"
            detalles_razon.append("Pesos ligeramente inclinados a BM25 para literalidad, pero con semántica.")
        else:
            detalles_razon.append("Petición de definición, pero query ya tenía especificidad media/alta.")


    # --- 4. Indicadores de CONCEPTUALIDAD (Prioridad para Embeddings) ---
    concept_keywords_strong = ["explica", "describe el proceso de", "analiza las causas de", "compara y contrasta",
                               "cuál es la importancia de", "interpreta", "relación entre", "impacto de",
                               "evolución de", "fundamentos de", "teoría de"]
    concept_keywords_medium = ["cómo funciona", "por qué ocurre", "cuáles son las características",
                               "tipos de", "función de", "origen de", "propiedades de"]

    is_conceptual = False
    conceptual_keyword_found = ""
    for keyword in concept_keywords_strong:
        if keyword in query_lower:
            is_conceptual = True
            conceptual_keyword_found = keyword
            detalles_razon.append(f"Palabra clave conceptual fuerte detectada: '{keyword}'.")
            break
    if not is_conceptual:
        for keyword in concept_keywords_medium:
            if keyword in query_lower:
                is_conceptual = True
                conceptual_keyword_found = keyword
                detalles_razon.append(f"Palabra clave conceptual media detectada: '{keyword}'.")
                break
    
    if is_conceptual:
        # Si es una pregunta conceptual sobre un término muy específico (ya capturado por NP, Fecha, Acrónimo)
        # Ej: "Explica el impacto de la Peste Negra" -> Peste Negra (NP) + Explica (Conceptual)
        if peso_bm25 >= 0.65 : # Ya era muy específico
            peso_bm25 = 0.55 # Mantenemos algo de BM25 para el término, pero damos espacio a la explicación
            peso_emb = 0.45
            razon_principal = "Pregunta Conceptual Muy Específica"
            detalles_razon.append(f"Término específico combinado con petición conceptual ('{conceptual_keyword_found}').")
        elif peso_bm25 >= 0.55 and peso_bm25 < 0.65: # Especificidad media
            peso_bm25 = 0.40
            peso_emb = 0.60
            razon_principal = "Pregunta Conceptual con Especificidad Media"
            detalles_razon.append(f"Término con especificidad media combinado con petición conceptual ('{conceptual_keyword_found}').")
        else: # Pregunta conceptual más general
            peso_bm25 = 0.25
            peso_emb = 0.75
            razon_principal = "Pregunta Conceptual General"
            detalles_razon.append(f"Mayor peso para Embeddings debido a '{conceptual_keyword_found}'.")


    # --- 5. Ajustes por Asignatura (si se proporciona y no hay una regla fuerte dominante) ---
    if subject and (razon_principal.startswith("Default") or "Petición de Definición General" in razon_principal):
        original_razon_principal = razon_principal # Guardar por si no se modifica
        if subject == "Lengua Castellana":
            if "analiza el poema" in query_lower or "figuras retóricas" in query_lower or "estilo de" in query_lower or "comentario de texto" in query_lower:
                peso_bm25 = 0.3
                peso_emb = 0.7
                razon_principal = f"Conceptual (Lengua - Análisis Literario)"
            elif "regla gramatical" in query_lower or "ortografía de" in query_lower or "sintaxis de" in query_lower:
                peso_bm25 = 0.6
                peso_emb = 0.4
                razon_principal = f"Específico (Lengua - Gramática/Ortografía)"
        elif subject == "Historia":
            if "batalla de" in query_lower or "tratado de" in query_lower or "reinado de" in query_lower or "guerra de" in query_lower:
                if peso_bm25 < 0.65: # Solo si no fue ya capturado por NP/Fecha con alta prioridad
                    peso_bm25 = 0.65
                    peso_emb = 0.35
                    razon_principal = f"Evento Específico (Historia)"
        
        if original_razon_principal != razon_principal: # Si se aplicó una regla de asignatura
             detalles_razon.append(f"Ajuste por asignatura '{subject}'.")


    # --- 6. Ajuste final por longitud de la query (si aún es default o poco definido) ---
    # Se aplica si ninguna regla fuerte o de especificidad media/conceptual clara dominó
    if razon_principal.startswith("Default") or \
       ("Petición de Definición General" in razon_principal and peso_bm25 == 0.55) or \
       (peso_bm25 >= 0.35 and peso_bm25 <= 0.45 and not is_conceptual): # Default o ligeramente inclinado a Emb sin ser conceptual fuerte

        num_words_query = len(query.split())
        if num_words_query > 10:
            peso_bm25 = 0.30
            peso_emb = 0.70
            razon_principal = "Ajuste por Longitud (Larga -> Conceptual)"
            detalles_razon.append(f"Query larga ({num_words_query} palabras), favoreciendo semántica.")
        elif num_words_query < 4:
            peso_bm25 = 0.50 # Si era default (0.4), lo sube un poco para términos cortos
            peso_emb = 0.50
            razon_principal = "Ajuste por Longitud (Corta -> Equilibrio/Específica)"
            detalles_razon.append(f"Query corta ({num_words_query} palabras), buscando equilibrio o término.")


    print_pesos_info(razon_principal, detalles_razon, peso_bm25, peso_emb)
    return peso_bm25, peso_emb

# ===> AHORA ACEPTA 'index_dir' COMO ARGUMENTO <===
def hybrid_retriever(query: str, index_dir: str) -> List[dict]:
    _initialize_retriever_components(index_dir)
    
    components = _retriever_cache[index_dir]
    client, faiss_index, texts, metadatas, bm25, reranker = (
        components["client"], components["faiss_index"], components["texts"],
        components["metadata"], components["bm25"], components["reranker"]
    )

    print(f"\n--- Buscando contexto para: '{query}' en el índice '{index_dir}' ---")
    
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_embedding_np = np.array([response.data[0].embedding], dtype=np.float32)

    distances, faiss_indices = faiss_index.search(query_embedding_np, config.K_FAISS_INITIAL)
    faiss_results = {idx: 1.0 / (1.0 + dist) for idx, dist in zip(faiss_indices[0], distances[0]) if idx != -1}

    tokenized_query = simple_tokenizer(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:config.K_BM25_INITIAL]
    bm25_results = {idx: bm25_scores[idx] for idx in bm25_top_indices if bm25_scores[idx] > 0}

    peso_bm25, peso_emb = calcular_pesos_dinamicos(query)
    candidate_ids = set(faiss_results.keys()) | set(bm25_results.keys())
    
    faiss_scores_list = list(faiss_results.values())
    min_faiss, max_faiss = (min(faiss_scores_list), max(faiss_scores_list)) if faiss_scores_list else (0.0, 0.0)
    bm25_scores_list = list(bm25_results.values())
    min_bm25, max_bm25 = (min(bm25_scores_list), max(bm25_scores_list)) if bm25_scores_list else (0.0, 0.0)

    hybrid_scores = {}
    for idx in candidate_ids:
        norm_f = norm_score(faiss_results.get(idx, 0.0), min_faiss, max_faiss)
        norm_b = norm_score(bm25_results.get(idx, 0.0), min_bm25, max_bm25)
        hybrid_scores[idx] = (peso_emb * norm_f) + (peso_bm25 * norm_b)

    top_hybrid_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:config.K_RERANK]
    
    rerank_pairs = [(query, texts[doc_id]) for doc_id in top_hybrid_ids if 0 <= doc_id < len(texts)]
    if not rerank_pairs: return []
        
    reranker_scores = reranker.predict(rerank_pairs)
    
    reranked_docs_info = []
    for i, doc_id in enumerate(top_hybrid_ids):
        if 0 <= doc_id < len(texts):
            reranked_docs_info.append({
                "doc_id": doc_id, "text": texts[doc_id], "metadata": metadatas[doc_id],
                "reranker_score": float(reranker_scores[i])
            })
    reranked_docs_info.sort(key=lambda x: x["reranker_score"], reverse=True)

    if getattr(config, 'USE_DYNAMIC_K', True):
        selected_for_dynamic_k = [doc for doc in reranked_docs_info if doc["reranker_score"] >= config.RERANKER_SCORE_THRESHOLD]
        if len(selected_for_dynamic_k) < config.MIN_CHUNKS_DYNAMIC:
            final_top_docs = reranked_docs_info[:config.MIN_CHUNKS_DYNAMIC]
        elif len(selected_for_dynamic_k) > config.MAX_CHUNKS_DYNAMIC:
            final_top_docs = selected_for_dynamic_k[:config.MAX_CHUNKS_DYNAMIC]
        else:
            final_top_docs = selected_for_dynamic_k
    else:
        final_top_docs = reranked_docs_info[:config.K_FINAL]

    print(f"--- Contexto Final Generado ({len(final_top_docs)} chunks) ---")
    return final_top_docs