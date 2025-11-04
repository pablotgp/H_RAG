# main.py (Versión Corregida y Definitiva)

import argparse
import os
import sys

# --- INSTRUCCIONES DE USO ---
# 1. Construir el índice: python main.py build-index
# 2. Hacer una pregunta:  python main.py ask "¿Tu pregunta aquí?"
# ------------------------------------------------------------------------------------

# --- Importaciones Corregidas ---
# En lugar de modificar el 'sys.path', importamos directamente desde el paquete 'src'.
# Esto es más robusto y es la forma estándar de hacerlo. Python buscará automáticamente
# una carpeta llamada 'src' en el mismo directorio que main.py.

try:
    from src.config import PDF_SOURCE_PATH, FAISS_INDEX_PATH
    from src.retrieval import build_index
    from src.chain import get_rag_response
except ImportError as e:
    print("Error: No se pudieron importar los módulos desde la carpeta 'src'.")
    print(f"Detalle del error: {e}")
    print("Asegúrate de que la estructura de tu proyecto es correcta y que tienes un archivo __init__.py en la carpeta 'src'.")
    sys.exit(1)
except ModuleNotFoundError as e:
    print(f"Error: No se encontró un módulo necesario: {e}")
    print("Asegúrate de que has instalado todas las dependencias con 'pip install -r requirements.txt'.")
    sys.exit(1)


def main():
    """
    Función principal que maneja los argumentos de la línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Sistema RAG para consultar documentos PDF.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Comandos disponibles"
    )

    # --- Comando: build-index ---
    build_parser = subparsers.add_parser(
        "build-index",
        help="Procesa el PDF fuente y construye los índices necesarios."
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la reconstrucción del índice aunque ya exista."
    )

    # --- Comando: ask ---
    ask_parser = subparsers.add_parser(
        "ask",
        help="Realiza una pregunta al sistema RAG."
    )
    ask_parser.add_argument(
        "query",
        type=str,
        help="La pregunta que deseas hacer, entre comillas."
    )

    args = parser.parse_args()

    # --- Lógica de Ejecución de Comandos ---
    if args.command == "build-index":
        print("--- Iniciando el comando 'build-index' ---")
        
        if not os.path.exists(PDF_SOURCE_PATH):
            print(f"\n[ERROR] El archivo PDF no se encuentra en: {PDF_SOURCE_PATH}")
            return

        if os.path.exists(FAISS_INDEX_PATH) and not args.force:
            print(f"\n[INFO] El índice ya existe. Usa '--force' para reconstruirlo.")
            return
        
        print("\nConstruyendo los índices... Esto puede tardar varios minutos.")
        try:
            build_index()
            print("\n[ÉXITO] Los índices se han construido y guardado correctamente.")
        except Exception as e:
            print(f"\n[ERROR] Ocurrió un error durante la construcción del índice: {e}")
            
    elif args.command == "ask":
        print("--- Iniciando el comando 'ask' ---")
        
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"\n[ERROR] No se ha encontrado el índice. Ejecuta 'python main.py build-index' primero.")
            return
            
        try:
            final_answer = get_rag_response(args.query)
            
            print("\n" + "="*20 + " Respuesta del Asistente " + "="*20)
            print(final_answer)
            print("=" * (42 + len(" Respuesta del Asistente ")))

        except Exception as e:
            print(f"\n[ERROR] Ocurrió un error al procesar la pregunta: {e}")


if __name__ == "__main__":
    main()