import openai
import json
import re
import chromadb
import time
import csv
import datetime
import argparse
import copy
from credentials import OPENAI_KEY

#from nltk.corpus import wordnet
# TODO: sort out mac install issues with nltk

# arg parse!
parser = argparse.ArgumentParser(
    description="Process RFP queries with OpenAI and ChromaDB.")
parser.add_argument("chunk_type", type=str,
                    help="Type of chunking (e.g., 'words' or 'sentences').")
parser.add_argument("chunk_size", type=int,
                    help="Size of chunks for document processing.")
parser.add_argument("top_k", type=int,
                    help="K chunks retrieved during search.")
parser.add_argument("method", type=str,
                    choices=["baseline", "filtering", "reranking", "reranking_filtering"],
                    help="Retrieval method to use.")


args = parser.parse_args()
CHUNK_TYPE = args.chunk_type
CHUNK_SIZE = args.chunk_size
TOP_K = args.top_k
METHOD = args.method


# TODO: set the api key
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
        response = openai.embeddings.create(
            model="text-embedding-3-large", input=batch)
        embeddings.extend([data.embedding for data in response.data])
    return embeddings


def old_retrieve_similar_docs(queries, rfp_ids, top_k=TOP_K):
    """Finds the top k most relevant documents for each query, searching only
    in the relevant document
    ParsonsGPT makes us pass in a relevant RFP, so this by default is doing
    that because we filter on the
    metadata to ensure we're not searching all of the chunks, only chunks in
    the relevant documents"""
    query_embeddings = embed_queries(queries)
    results_list = [
        collection.query(query_embeddings=[embedding], n_results=top_k, where={
                         "document": rfp_id})
        for embedding, rfp_id in zip(query_embeddings, rfp_ids)
    ]

    # print(len(results_list))
    # print(results_list)

    return [[res["text"] for res in results["metadatas"][0]] if results["metadatas"] else [] for results in results_list]


def retrieve_similar_docs(queries, rfp_ids, top_k=TOP_K):
    """Retrieve documents with scores from ChromaDB"""
    query_embeddings = embed_queries(queries)

    results_list = [
        collection.query(query_embeddings=[embedding],
                         n_results=top_k,
                         where={"document": rfp_id})       
        for embedding, rfp_id in zip(query_embeddings, rfp_ids)
    ]

    output = []
    for results in results_list:
        if results["metadatas"]:
            chunks = [metadata["text"] for metadata in results["metadatas"][0]]
            scores = results["distances"][0]
            output.append(zip(chunks, scores))

    return output


def llm_rerank(query, documents):
    """LLM-assisted reranking returning new document order"""
    if not documents:
        return []

    doc_list = "\n".join([f"{idx}: {doc}" for idx, doc in enumerate(documents)])
    prompt = f"""Re-rank these documents by relevance to the query. Return ONLY a JSON array of original indices (0-based) in new order.

Query: {query}

Documents:
{doc_list}

Reranked indices:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        response_text = response.choices[0].message.content.strip()

        # Extract JSON array from response
        match = re.search(r'\[.*\]', response_text)
        if not match:
            raise ValueError("No JSON array found")

        ranked_indices = json.loads(match.group(0))

        # Validate indices
        if sorted(ranked_indices) != list(range(len(documents))):
            print(sorted(ranked_indices))
            print(list(range(len(documents))))
            raise ValueError("Invalid indices")

        return ranked_indices
    except Exception as e:
        print(f"Reranking error: {str(e)} - Using original order")
        return list(range(len(documents)))


def process_results(doc_score_pairs, method, query):
    """Process documents through filtering and reranking pipeline"""
    if not doc_score_pairs:
        return []

    # Baseline: No processing
    if method == "baseline":
        return [doc for doc, _ in doc_score_pairs]

    # Initial Chroma score filtering
    if method == "filtering" or method == "reranking_filtering":
        sorted_by_score = sorted(doc_score_pairs, key=lambda x: x[1])
        #filtered_docs = sorted_by_score[:len(sorted_by_score)//2]
        filtered_docs = [(doc, score) for doc, score in doc_score_pairs if score >= 0.7]

        if method == "filtering":
            return [doc for doc, _ in filtered_docs]
    else:
        # we are reranking
        filtered_docs = doc_score_pairs

    docs_to_rerank = [doc for doc, _ in filtered_docs]

    # LLM reranking
    ranked_indices = llm_rerank(query, docs_to_rerank)
    reranked_docs = [docs_to_rerank[i] for i in ranked_indices]
    if method == "reranking":
        return reranked_docs

    # Final filtering after reranking
    return reranked_docs[:len(reranked_docs)//2]


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
    with open("subset_final.json", "r") as f:  # TODO: change this file as needed
        qa_data = json.load(f)

    # xtracting queries + corresponding RFP IDs
    queries = [item["question"] for item in qa_data]
    # this is needed because I was dumb and used underscores in one place and hyphens in another
    rfp_ids = [re.sub(r'_', '-', item["RFP_id"]) + ".pdf" for item in qa_data]

    # retrieving relevant context
    retrieve_start = time.time()
    retrieved_docs = retrieve_similar_docs(queries, rfp_ids)
    retrieve_time = time.time() - retrieve_start

    """sending -> LLM"""

    # get the llm responses
    results = []
    for idx, (qa_pair, doc_pairs) in enumerate(zip(qa_data, retrieved_docs)):
        query = qa_pair["question"]

        og_context = copy.deepcopy(doc_pairs)
        retrieved_context_for_json = [random_idx for random_idx, _ in og_context]
        # process based on method
        process_start = time.time()
        processed_docs = process_results(doc_pairs, METHOD, query)
        context = " ".join(processed_docs) if processed_docs else ""
        process_time = time.time() - process_start

        # generate answer
        llm_start = time.time()
        if idx % 5 == 0:
            print(idx)
            time.sleep(3)
        answer = ask_llm(query, context)
        llm_time = time.time() - llm_start

        # calc total time for query
        total_time = (retrieve_time/len(qa_data)) + process_time + llm_time

        results.append({
            "question": query,
            "llm_response": answer,
            "ground_truth_answer": qa_pair["answer"],
            "retrieved_context": retrieved_context_for_json,
            "processed_context": processed_docs,
            "ground_truth_context": qa_pair["context"],
            "metadata": {
                "question_id": qa_pair["question_id"],
                "RFP_id": qa_pair["RFP_id"],
                "RFP_type": qa_pair["RFP_type"],
                "chunks": qa_pair["chunks"],
                "manually_edited": qa_pair["manually_edited"]
            },
            "timing": {
                "retrieve_context": retrieve_time/len(qa_data) + process_time,
                "llm_response": llm_time,
                "total": total_time
            }
        })

    # Save CSV
    base_name = f"{METHOD}_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}"
    csv_filename = f"outputs/reranking/csv_{base_name}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "LLM Response"])
        writer.writerows([(r["question"], r["llm_response"]) for r in results])

    # Save JSON
    experiment_output = {
        "hyperparameters": {
            "method": METHOD,
            "k": TOP_K,
            "chunk_type": CHUNK_TYPE,
            "chunk_size": CHUNK_SIZE
        },
        "chroma_db_size": collection.count(),
        "total_time": retrieve_time + sum(r["timing"]["llm_response"] for r in results),
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results
    }

    filename = f"outputs/reranking/exp_output_{base_name}.json"

    with open(filename, 'w') as f:
        json.dump(experiment_output, f, indent=2)
