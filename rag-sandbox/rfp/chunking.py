import re
import PyPDF2

# Note: will assume are pdf not .txt, following parsonsgpt


def load_pdf(pdf_path, chunk_type, chunk_size):
    """
    Extracts and chunks text from 1 PDF file.

    Parameters
    ----------
    pdf_path: Path to the PDF file.
    chunk_type: The type of chunking to perform ('characters', 'words', 'sentences', 'paragraphs', 'pages').
    chunk_size: The size of each chunk (number of characters, words, etc.).

    Returns
    -------
    List of document chunks (strings).
    """
    if chunk_type == "pages":
        return chunk_by_pages(pdf_path, chunk_size)

    else:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return chunk_document(text, chunk_type, chunk_size)


# Chunk by: characters, words, sentences, paragraphs, pages
# Pages is special since we need different way of reading it
def chunk_by_pages(pdf_path, chunk_size):
    """
    Chunk a PDF document by pages, with a fixed number of pages in each chunk
    """

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        chunks = []
        text = ""

        # Iterate over each page in the PDF
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

            # If the chunk reaches the specified size, add it to the chunks list and reset the chunk
            if (page_num+1) % chunk_size == 0:
                chunks.append(text)
                text = ""

        # Add any remaining pages that didn't form a full chunk
        if text:
            chunks.append(text)

    return chunks


def chunk_document(doc_text, chunk_type, chunk_size):
    """
    Chunk a document based on the specified chunk type and size, other than by pages
    Parameters
    ----------
    doc_text: The full document text.
    chunk_type: The type of chunking to perform ('characters', 'words', 'sentences', 'paragraphs').
    chunk_size: The size of each chunk (number of characters, words, etc.).
    Returns
    -------
    List of document chunks (strings).
    """
    if chunk_type == "characters":
        return chunk_by_characters(doc_text, chunk_size)
    elif chunk_type == "words":
        return chunk_by_words(doc_text, chunk_size)
    elif chunk_type == "sentences":
        return chunk_by_sentences(doc_text, chunk_size)
    elif chunk_type == "paragraphs":
        return chunk_by_paragraphs(doc_text, chunk_size)
    else:
        raise ValueError(f"Unsupported chunk type: {chunk_type}")


def chunk_by_characters(doc_text, char_per_chunk=500):
    """
    Chunk text into fixed-size character chunks
    """
    chunks = [doc_text[i:i + char_per_chunk]
              for i in range(0, len(doc_text), char_per_chunk)]
    return chunks


def chunk_by_words(doc_text, words_per_chunk=100):
    """
    Chunk text into fixed-size word chunks
    """
    words = doc_text.split()  # Split text into words
    chunks = [" ".join(words[i:i + words_per_chunk])
              for i in range(0, len(words), words_per_chunk)]
    return chunks


# Matches parsonsgpt in splitting by { '.', '!', '?' }
def chunk_by_sentences(doc_text, lines_per_chunk=10):
    """
    Chunk a document by lines, with a fixed number of lines in each chunk
    """
    lines = re.split(r'(?<=[.!?])', doc_text.strip())
    chunks = []

    # Group lines into chunks of size `lines_per_chunk`
    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i:i + lines_per_chunk])
        chunks.append(chunk)

    return chunks


# TODO: follows parsons gpt in recognising as \n, see if correct (more usually for lines?)
# Also note that our current morehop_context pdf does not differentiate between lines and paras. might want to change this
def chunk_by_paragraphs(doc_text, chunk_size):
    """
    Chunk text into groups of paragraphs based on the specified chunk size.
    """
    paragraphs = doc_text.split("\n \n")

    # Create chunks of paragraphs, each containing `chunk_size` paragraphs
    chunks = [paragraphs[i:i + chunk_size]
              for i in range(0, len(paragraphs), chunk_size)]

    # Join paragraphs in each chunk into a single string
    return ['\n\n'.join(chunk) for chunk in chunks]
