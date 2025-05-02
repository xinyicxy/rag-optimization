import openai
import json
import chromadb
import time
import csv
import datetime
import argparse
from credentials import OPENAI_KEY


# arg parse!
parser = argparse.ArgumentParser(description="Process MoreHop queries with OpenAI and ChromaDB.")
parser.add_argument("chunk_type", type=str, choices=["characters", "words", "sentences", "paragraphs", "pages"],
                    default="words", help="Chunking method to use.")
parser.add_argument("chunk_size", type=int, default=1000,
                    help="Size of each chunk.")
parser.add_argument("top_k", type=int, default=2, help="K chunks retrieved during search.")

args = parser.parse_args()
CHUNK_TYPE = args.chunk_type
CHUNK_SIZE = args.chunk_size
TOP_K = args.top_k

# set the api key
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
            If the answer is a name, format it as follows: Firstname Lastname
            If the answer is a place or organization, give only the name of the place or organization.
            If the answer is a year, give only the year in the format: YYYY
            If the answer is a date, format it as follows: YYYY-MM-DD (ISO standard)
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
    with open("../../multihop-data/morehopqa_120.json", "r") as f:
        qa_data = json.load(f)

    # extracting queries
    queries = [item["previous_question"] for item in qa_data]

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
        answer = ask_llm(qa["previous_question"], context)
        llm_time = time.time() - llm_start

        # Calculate total time for this query
        total_time = (retrieve_time/len(qa_data)) + llm_time

        results.append({
            "question": qa["previous_question"],
            "llm_response": answer,
            "ground_truth_answer": qa["previous_answer"],
            "retrieved_context": retrieved_docs_batch[idx],
            "ground_truth_context": qa["context"],
            "metadata": {
                "question_id": qa["_id"],
                "answer_type": qa["previous_answer_type"]
                # "chunks": qa["chunks"], - can change to context label instead
            },
            "timing": {
                "retrieve_context": retrieve_time/len(qa_data),
                "llm_response": llm_time,
                "total": total_time
            }
        })

    # Save CSV
    csv_filename = f"outputs/morehop_csv_exp_output_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.csv"
    with open(csv_filename, 'w', newline='') as f:
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

    filename = f"outputs/morehop_exp_output_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.json"
    with open(filename, 'w') as f:
        json.dump(experiment_output, f, indent=2)
