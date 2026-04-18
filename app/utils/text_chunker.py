def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Full extracted text
        chunk_size (int): Maximum characters per chunk
        overlap (int): Number of characters to overlap between chunks

    Returns:
        list[str]: List of chunked text
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks