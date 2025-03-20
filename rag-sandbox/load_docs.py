# this file loads text files into local memory and embeds them

import os
from sentence_transformers import SentenceTransformer
import chromadb
from chunking import load_pdf

# getting credentials
DOCS_DIR = "./documents"
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_documents")

# TODO: make this an input arg maybe later
chunk_type = "pages"
chunk_size = 1


def load_documents(directory, chunk_type, chunk_size):
    """
    loads pdf files from a directory and chunks them, returns list of (doc_id, list of content chunks)
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
    create embeddings from a list of documents
    """
    return EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)


# # testing it out - sweet this works!
# documents = load_documents(DOCS_DIR)
# print(f"loaded {len(documents)} documents")
# TODO Change the load_docs + the pre-embeddings bit
documents = load_documents(DOCS_DIR, chunk_type, chunk_size)
doc_texts = [doc[1] for doc in documents]
# embeddings = embed_texts(doc_texts)
all_embeddings = []
for text in doc_texts:
    toAdd = embed_texts(text)
    all_embeddings.append(toAdd)
print("embeddings generated")

# storing in chromadb
# NOTE: right now storing the raw text as metadata but will probably stop this
# TODO: change the database to store chunk_id, or do we only want to store chunk id
for (doc_id, text), embeddings in zip(documents, all_embeddings):
    i = 0
    for embedding in embeddings:
        collection.add(doc_id=[doc_id], chunk_id=[doc_id+"_"+i], embeddings=[embedding.tolist()],
                       metadatas=[{"text": text}])
        i += 1

print("documents in chroma")
