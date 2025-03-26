import openai
import json
import re
import chromadb
import time
import csv
import datetime
import argparse


# arg parse!
parser = argparse.ArgumentParser(description="Process RFP queries with OpenAI and ChromaDB.")
parser.add_argument("--chunk_type", type=str, required=True, help="Type of chunking (e.g., 'words' or 'sentences').")
parser.add_argument("--chunk_size", type=int, required=True, help="Size of chunks for document processing.")
parser.add_argument("--top_k", type=int, required=True, help="K chunks retrieved during search.")

args = parser.parse_args()
CHUNK_TYPE = args.chunk_type
CHUNK_SIZE = args.chunk_size
TOP_K = args.top_k


# TODO: set the api key
OPENAI_KEY = "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"
openai.api_key = OPENAI_KEY

# initialize chroma db for searching 
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(
    name=f"rag_documents_{CHUNK_TYPE}_{CHUNK_SIZE}"
)


def is_collection_empty(collection):
    # function for debugging
    """double checking collection (embedding store) isn't empty"""
    return collection.count() == 0


def embed_queries(queries, batch_size=10):
    """Generates embeddings using OpenAI's text-embedding-3-large in batches"""
    # TODO: make it so that embedding model is not hard coded
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        response = openai.embeddings.create(model="text-embedding-3-large", input=batch)
        embeddings.extend([data.embedding for data in response.data])
    return embeddings


def retrieve_similar_docs(queries, rfp_ids, top_k=TOP_K):
    """Finds the top k most relevant documents for each query, searching only
    in the relevant document
    ParsonsGPT makes us pass in a relevant RFP, so this by default is doing
    that because we filter on the
    metadata to ensure we're not searching all of the chunks, only chunks in
    the relevant documents"""
    query_embeddings = embed_queries(queries)
    results_list = [
        collection.query(query_embeddings=[embedding], n_results=top_k, where={"document": rfp_id})
        for embedding, rfp_id in zip(query_embeddings, rfp_ids)
    ]

    # print(len(results_list))
    # print(results_list)

    return [[res["text"] for res in results["metadatas"][0]] if results["metadatas"] else [] for results in results_list]


def batch_process(items, batch_size=10):
    """Split items into batches"""
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]


def ask_llm(query, context):
    """Generates answer for a single query with proper message formatting"""

    prompt = """ You are a helpful assistant. Answer only the given question concisely.
            Do not generate any extra responses or questions, or generate the context. 
            Summarize the context into one correct response.
            Stop immediately after answering.
            If you do not have sufficient information to answer the question, return only
            'Insufficient information to answer question based on given context'"""
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing query: {query[:50]}... - {str(e)}")
        return "Error generating response"


if __name__ == "__main__":
    # load in json question data
    with open("subset-final.json", "r") as f: # TODO: change this file as needed
        qa_data = json.load(f)

    # xtracting queries + corresponding RFP IDs
    queries = [item["question"] for item in qa_data]

    # this is needed because I was dumb and used underscores in one place and hyphens in another
    rfp_ids = [re.sub(r'_', '-', item["RFP_id"]) + ".pdf" for item in qa_data]

    # retrieving relevant context
    retrieve_start = time.time()
    retrieved_docs_batch = retrieve_similar_docs(queries, rfp_ids)
    retrieve_time = time.time() - retrieve_start

    # checking to make sure some output was received
    contexts = [" ".join(docs) if docs else "No relevant context found" for docs in retrieved_docs_batch]

    """sending -> LLM"""

    # get the llm responses
    results = []
    for idx, (qa, context) in enumerate(zip(qa_data, contexts)):
        # get llm responses with timing
        llm_start = time.time()
        if idx % 10 == 0:
            print(idx)
            time.sleep(3)
        answer = ask_llm(qa["question"], context)
        llm_time = time.time() - llm_start

        # calc total time for query
        total_time = (retrieve_time/len(qa_data)) + llm_time

        results.append({
            "question": qa["question"],
            "llm_response": answer,
            "ground_truth_answer": qa["answer"],
            "retrieved_context": retrieved_docs_batch[idx],
            "ground_truth_context": qa["context"],
            "metadata": {
                "question_id": qa["question_id"],
                "RFP_id": qa["RFP_id"],
                "RFP_type": qa["RFP_type"],
                "chunks": qa["chunks"],
                "manually_edited": qa["manually_edited"]
            },
            "timing": {
                "retrieve_context": retrieve_time/len(qa_data),
                "llm_response": llm_time,
                "total": total_time
            }
        })

    # Save CSV
    with open('full-experiment-results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "LLM Response"])
        writer.writerows([(r["question"], r["llm_response"]) for r in results])

    # Save JSON
    experiment_output = {
        "hyperparameters": {
            "k": TOP_K,
            "chunk_type": CHUNK_TYPE,
            "chunk_size": CHUNK_SIZE
        },
        "chroma_db_size": collection.count(),
        "total_time": retrieve_time + sum(r["timing"]["llm_response"] for r in results),
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results
    }

    filename = f"exp_output_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.json"

    with open(filename, 'w') as f:
        json.dump(experiment_output, f, indent=2)