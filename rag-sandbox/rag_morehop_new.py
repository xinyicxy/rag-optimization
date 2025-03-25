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

args = parser.parse_args()
CHUNK_TYPE = args.chunk_type
CHUNK_SIZE = args.chunk_size


# TODO: set the api key
OPENAI_KEY = "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"
openai.api_key = OPENAI_KEY
TOP_K = 2 # set to 2 because morehop needs exactly 2 contexts

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


def retrieve_similar_docs(queries, top_k=TOP_K):
    """Finds the top k most relevant documents for each query"""
    query_embeddings = embed_queries(queries)
    results_list = [
        collection.query(query_embeddings=[embedding], n_results=top_k)
        for embedding in query_embeddings
    ]
    # print(len(results_list))
    # print(results_list)
    return [[res["text"] for res in results["metadatas"][0]] if results["metadatas"] else [] for results in results_list]


def batch_process(items, batch_size=10):
    """Split items into batches"""
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]


def ask_llm(query, context):
    """Generates answer for a single query with proper message formatting"""

    prompt = """You are a helpful assistant. Answer only the given question concisely.
            Answer ONLY with the facts given in the context.
            Do not generate any extra responses or questions, or generate the context.
            If the answer is a string or character, output only the string or character.
            If the answer is a date, format it as follows: YYYY-MM-DD (ISO standard)
            If the answer is a datetime, format it is as follows: YYYY-MM-DD hh:mm
            If the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.
            Answer as short as possible in the formats specified.
            Stop immediately after answering."""

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
    with open("morehop_original_test.json", "r") as f: # ../multihop-data/data/morehopqa_final.json
        qa_data = json.load(f)

    # extracting queries
    queries = [item["question"] for item in qa_data]

    # retrieving relevant context
    retrieve_start = time.time()
    retrieved_docs_batch = retrieve_similar_docs(queries)
    retrieve_time = time.time() - retrieve_start

    # checking to make sure some output was received
    contexts = [" ".join(docs) if docs else "No relevant context found" for docs in retrieved_docs_batch] # Change to negative rejection?

    """sending -> LLM"""

    # Get LLM responses
    results = []
    for idx, (qa, context) in enumerate(zip(qa_data, contexts)):
        # Get LLM response with timing
        llm_start = time.time()
        if idx % 10 == 0:
            print(idx)
            time.sleep(3)
        answer = ask_llm(qa["question"], context)
        llm_time = time.time() - llm_start

        # Calculate total time for this query
        total_time = (retrieve_time/len(qa_data)) + llm_time

        results.append({
            "question": qa["question"],
            "llm_response": answer,
            "ground_truth_answer": qa["answer"],
            "retrieved_context": retrieved_docs_batch[idx],
            "ground_truth_context": qa["context"],
            "metadata": {
                "question_id": qa["_id"],
                "answer_type": qa["answer_type"],
                "reasoning_type": qa["reasoning_type"]
                # "chunks": qa["chunks"], - can change to context label instead
            },
            "timing": {
                "retrieve_context": retrieve_time/len(qa_data),
                "llm_response": llm_time,
                "total": total_time
            }
        })

    # Save CSV
    with open('morehop_new_results.csv', 'w', newline='') as f:
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

    with open('morehop_new_output.json', 'w') as f:
        json.dump(experiment_output, f, indent=2)
