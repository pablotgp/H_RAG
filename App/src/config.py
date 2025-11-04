# src/config.py (Versión Final Multi-Documento)

import os
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# ==============================================================================
#  1. RUTAS DE DIRECTORIOS (Ahora son genéricas)
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorio que contiene TODOS los archivos PDF
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Directorio PADRE que contendrá las subcarpetas de los índices para cada PDF
INDEX_PARENT_DIR = os.path.join(PROJECT_ROOT, "index")

# ==============================================================================
#  2. CLAVES DE API Y CONFIGURACIÓN DE MODELOS
# ==============================================================================

# Claves de API (cargadas desde .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Modelo de LLM de OpenAI a utilizar
LLM_MODEL_NAME = "gpt-4o"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# ==============================================================================
#  3. PARÁMETROS DE BÚSQUEDA Y RETRIEVAL
# ==============================================================================

# Parámetros para la búsqueda híbrida y el reranking
K_FAISS_INITIAL = 100
K_BM25_INITIAL = 100
K_RERANK = 100
K_FINAL = 3  # Número final de chunks a devolver si K dinámico está desactivado

# Parámetros para la selección dinámica de K
USE_DYNAMIC_K = True
RERANKER_SCORE_THRESHOLD = 1.75
MIN_CHUNKS_DYNAMIC = 3
MAX_CHUNKS_DYNAMIC = 5

# Modelo para el Re-ranker
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'