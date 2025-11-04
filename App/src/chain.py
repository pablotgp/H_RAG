# src/chain.py (Versión Final y Corregida para Multi-Documento)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import traceback

from . import config
from .retrieval import hybrid_retriever

_llm_instance = None
def _initialize_llm():
    """Inicializa el LLM una sola vez."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(model=config.LLM_MODEL_NAME, api_key=config.OPENAI_API_KEY, temperature=config.LLM_TEMPERATURE)
    return _llm_instance

EDUCATIONAL_ASSISTANT_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente educativo experto y amigable. Tu objetivo es ayudar al usuario a comprender un tema basándote estricta y únicamente en el contexto proporcionado.
---
Contexto Proporcionado:
{context_string}
---
Pregunta del Usuario:
{query}
---
Tu Respuesta Detallada:
""")

def format_docs(docs: list) -> str:
    """Une el contenido de los documentos en un solo string para el LLM."""
    return "\n\n---\n\n".join([doc.get("text", "") for doc in docs])

# ===> CORRECCIÓN CLAVE: Simplificamos la cadena de generación <===
# Esta cadena ahora solo se encarga de generar la respuesta,
# asumiendo que ya tiene el contexto y la pregunta.
generation_chain = (
    EDUCATIONAL_ASSISTANT_PROMPT
    | _initialize_llm()
    | StrOutputParser()
)

def get_rag_response(query: str, index_dir: str) -> dict:
    """
    Punto de entrada principal. Orquesta los pasos de forma explícita.
    1. Llama al retriever.
    2. Formatea el contexto.
    3. Llama a la cadena de generación.
    """
    print(f"--- Ejecutando cadena RAG para la consulta: '{query}' ---")
    try:
        # ===> PASO 1: Llamar al retriever directamente (¡Aquí estaba el error!) <===
        # Le pasamos explícitamente la consulta y el directorio del índice.
        context_docs = hybrid_retriever(query=query, index_dir=index_dir)
        if not context_docs:
            return {"answer": "No se encontró información relevante en el documento para esta pregunta.", "context": []}

        # ===> PASO 2: Formatear los documentos para el LLM <===
        context_string = format_docs(context_docs)

        # ===> PASO 3: Invocar la cadena de generación con toda la información <===
        answer = generation_chain.invoke({
            "query": query,
            "context_string": context_string
        })
        
        return {"answer": answer, "context": context_docs}

    except Exception as e:
        print(f"ERROR durante la invocación de la cadena RAG: {e}")
        traceback.print_exc()
        return {"answer": "Lo siento, ha ocurrido un error al procesar tu pregunta.", "context": []}