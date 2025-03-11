# this file does retrieval and generation 
from transformers import AutoModelForCausalLM, AutoTokenizer
from load_docs import EMBEDDING_MODEL, embed_texts
from load_docs import collection

# had to uninstall and redownload pytorch cpu to support cpu-only inference
import torch
print(torch.cuda.is_available())  # should return false
print(torch.device("cpu"))  # should print cpu

# load mistral-7b
# device = torch.device("cpu")
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float16,
                                             device_map="cpu")
#model = model.to(device)
print("model on the cpu")


def retrieve_similar_docs(query, top_k=1):
    """finds the top k relative to query"""
    query_embedding = embed_texts([query])[0].tolist()

    results = collection.query(query_embeddings=[query_embedding],
                               n_results=top_k)
    return [res["text"] for res in results["metadatas"][0]]


def ask_llm(query, context):
    """generating answer from context"""
    prompt = f"""
    You are a helpful assistant. Answer only the given question concisely.
    Do not generate any extra responses or questions, or generate the context. 
    Summarize the context into one correct response.
    Stop immediately after answering.

            Context:
            {context}

            Question: {query}
            Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(output[0], skip_special_tokens=True)


query = "What is the capital of france? "
retrieved_docs = retrieve_similar_docs(query)
# print("Retrieved documents:", retrieved_docs)

#print(type(retrieved_docs))
#print(retrieved_docs)
context = " ".join([doc for doc in retrieved_docs])
#print(context)
answer = ask_llm(query, context)
print("LLM Response:", answer)
