# app.py (Versión Final con Fuentes Reconectadas)

import streamlit as st
import os
import time
# --- MODIFICACIÓN CLAVE PARA CORREGIR IMPORTACIONES ---
# Añade la carpeta 'app' al path de Python para que encuentre el paquete 'src'
sys.path.append(os.path.abspath('app'))

# --- Importaciones del Backend RAG ---
try:
    from src.utils import calculate_checksum
    from src.retrieval import build_index, _initialize_retriever_components
    from src.chain import get_rag_response
    from src.config import DATA_DIR, INDEX_PARENT_DIR
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.stop()

# ==============================================================================
#  Funciones Auxiliares de la App
# ==============================================================================

def get_available_pdfs():
    if not os.path.exists(DATA_DIR): return []
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

def get_index_dir_for_pdf(pdf_filename):
    dir_name = os.path.splitext(pdf_filename)[0].replace(" ", "_").replace("-", "_") + "_index"
    return os.path.join(INDEX_PARENT_DIR, dir_name)

@st.cache_resource
def load_rag_components(index_dir):
    print(f"--- Cargando componentes del RAG en caché para {index_dir} ---")
    _initialize_retriever_components(index_dir)

def check_index_status(pdf_filename):
    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    index_dir = get_index_dir_for_pdf(pdf_filename)
    checksum_path = os.path.join(index_dir, "index.checksum")
    faiss_path = os.path.join(index_dir, "faiss_index")
    pdf_checksum = calculate_checksum(pdf_path)
    if not os.path.exists(faiss_path) or not os.path.exists(checksum_path): return "MISSING"
    with open(checksum_path, "r") as f: indexed_checksum = f.read()
    return "READY" if pdf_checksum == indexed_checksum else "MISMATCH"

# ==============================================================================
#  Configuración de la Página e Inicialización de Estado
# ==============================================================================

st.set_page_config(page_title="Asistente Multi-Documento", page_icon="🤖", layout="wide")

if "selected_pdf" not in st.session_state: st.session_state.selected_pdf = None
if "messages" not in st.session_state: st.session_state.messages = []

# ==============================================================================
#  Interfaz de Usuario
# ==============================================================================

st.title("🤖 Asistente Multi-Documento RAG")

with st.sidebar:
    st.header("1. Selecciona un Documento")
    available_pdfs = get_available_pdfs()
    if not available_pdfs:
        st.warning("No se encontraron PDFs en la carpeta 'data'.")
    else:
        for pdf_file in available_pdfs:
            if st.button(pdf_file, use_container_width=True, type="secondary" if st.session_state.selected_pdf != pdf_file else "primary"):
                st.session_state.selected_pdf = pdf_file
                st.session_state.messages = []
                st.rerun()
    st.header("2. Opciones")
    if st.button("Limpiar Historial", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Historial limpiado. ¿En qué puedo ayudarte?"}]
        st.rerun()

# --- Lógica de Renderizado Principal ---
if st.session_state.selected_pdf is None:
    st.info("Por favor, selecciona un documento de la barra lateral para comenzar.")
else:
    pdf_name = st.session_state.selected_pdf
    st.header(f"Conversando con: `{pdf_name}`")
    index_status = check_index_status(pdf_name)

    # MODO DE INDEXACIÓN
    if index_status != "READY":
        st.warning(f"**Estado del Índice:** `{index_status}`")
        if index_status == "MISSING": st.info("Este documento no ha sido indexado.")
        elif index_status == "MISMATCH": st.error("El documento ha cambiado. El índice es obsoleto.")
        if st.button("Construir Índice Ahora", type="primary", use_container_width=True):
            with st.spinner(f"Indexando {pdf_name}... (Puede tardar varios minutos)"):
                pdf_path = os.path.join(DATA_DIR, pdf_name)
                build_index(pdf_path)
                st.success("¡Índice construido! La página se recargará.")
                time.sleep(2); st.rerun()
    
    # MODO CHAT
    else:
        index_dir = get_index_dir_for_pdf(pdf_name)
        load_rag_components(index_dir)

        if not st.session_state.messages:
            st.session_state.messages = [{"role": "assistant", "content": f"¡Hola! El índice para '{pdf_name}' está listo. ¿Qué quieres saber?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # ===> CORRECCIÓN 1: Mostrar fuentes de mensajes ANTIGUOS <===
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("Ver fuentes utilizadas"):
                        for i, chunk in enumerate(message["sources"]):
                            st.info(f"**Fuente {i+1}:**\n" + chunk.get("text", "Contenido no disponible."))

        if prompt := st.chat_input(f"Haz tu pregunta sobre {pdf_name}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Buscando y generando respuesta..."):
                    result = get_rag_response(prompt, index_dir=index_dir)
                    response = result["answer"]
                    context_chunks = result["context"]

                    st.markdown(response)
                    
                    # ===> CORRECCIÓN 2: Mostrar fuentes para el mensaje NUEVO <===
                    if context_chunks:
                        with st.expander("Ver fuentes utilizadas"):
                            for i, chunk in enumerate(context_chunks):
                                st.info(f"**Fuente {i+1}:**\n" + chunk.get("text", "Contenido no disponible."))

            # ===> CORRECCIÓN 3: Guardar las fuentes en el historial <===
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": context_chunks
            })
