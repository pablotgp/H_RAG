# src/utils.py
import hashlib

def calculate_checksum(file_path: str) -> str | None:
    """
    Calcula el checksum SHA256 de un archivo.
    Devuelve el checksum como un string hexadecimal o None si el archivo no se encuentra.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Leer el archivo en trozos para no consumir mucha memoria con archivos grandes
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except FileNotFoundError:
        return None