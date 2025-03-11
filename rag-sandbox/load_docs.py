# this file loads text files into local memory and embeds them

import os
from sentence_transformers import SentenceTransformer
import chromadb

# getting credentials

DOCS_DIR = "./documents"
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_documents")


def load_documents(directory):
    """
    loads txt files from a directory into a list of (doc_id, content)
    """
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append((filename, f.read()))
    return docs


def embed_texts(texts):
    """
    create embeddings from a list of documents
    """
    return EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)


# testing it out - sweet this works!
documents = load_documents(DOCS_DIR)
print(f"loaded {len(documents)} documents")
doc_texts = [doc[1] for doc in documents]
embeddings = embed_texts(doc_texts)
print("embeddings generated")

# storing in chromadb
# NOTE: right now storing the raw text as metadata but will probably stop this
for (doc_id, text), embedding in zip(documents, embeddings):
    collection.add(ids=[doc_id], embeddings=[embedding.tolist()],
                   metadatas=[{"text": text}])

print("documents in chroma")
