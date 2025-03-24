import os
import argparse
import chromadb
from openai import OpenAI
import tiktoken
from chunking import load_pdf


OPENAI_KEY = "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"
client = OpenAI(api_key=OPENAI_KEY)

# specifying dir
DOCS_DIR = "./documents"

# arg parse!
parser = argparse.ArgumentParser(description="Process and embed PDF documents using OpenAI embeddings and ChromaDB.")
parser.add_argument("--chunk_type", type=str, choices=["characters", "words", "sentences", "paragraphs", "pages"], default="words", help="Chunking method to use.")
parser.add_argument("--chunk_size", type=int, default=1000, help="Size of each chunk.")
args = parser.parse_args()

CHUNK_TYPE = args.chunk_type
CHUNK_SIZE = args.chunk_size

# chroma db setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = f"rag_documents_{CHUNK_TYPE}_{CHUNK_SIZE}"
collection = chroma_client.get_or_create_collection(collection_name)

# OpenAI Embedding Model settings
EMBEDDING_MODEL = "text-embedding-3-large"
ENCODER = tiktoken.get_encoding("cl100k_base")

def load_documents(directory, chunk_type, chunk_size):
    """
    Loads PDFs from a directory, chunks them,
    and returns a list of (doc_id, chunk_list)
    """
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = os.path.join(directory, filename)
            chunks = load_pdf(path, chunk_type, chunk_size)
            docs.append((filename, chunks))
    return docs

def embed_texts(texts):
    """
    Generate OpenAI embeddings for a list of texts
    """
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]

def count_tokens(text):
    """
    Estimating the token count (for future use)
    """
    return len(ENCODER.encode(text))

if __name__ == "__main__":
    # load and chunk documents
    documents = load_documents(DOCS_DIR, CHUNK_TYPE, CHUNK_SIZE)
    print(f"Loaded {len(documents)} documents using chunking method: {CHUNK_TYPE}")

    # embedding + store in ChromaDB
    for doc_id, chunks in documents:
        print(f"Processing {doc_id} with {len(chunks)} chunks...")

        # embed in batches (to respect OpenAI API limits)
        batch_size = 10  # modify as needed
        for i in range(0, len(chunks), batch_size):
            chunk_batch = chunks[i:min(i + batch_size, len(chunks) - 1)]
            embeddings = embed_texts(chunk_batch)  # get the embeddings

            # store chunk + corresponding embedding in ChromaDB
            for j, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i+j}"
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "document": doc_id,
                        "chunk_type": CHUNK_TYPE,
                        "chunk_index": i + j,
                        "text": chunk,
                        "token_count": count_tokens(chunk)
                    }]
                )
        print(f"Finished storing {doc_id} in ChromaDB")

    print("All documents successfully processed and stored!")
