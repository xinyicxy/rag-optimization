"""This script processes a PDF file, breaks it into manageable chunks of text, 
and then uses a language model via OpenRouter to generate question-answer (QA) pairs for each chunk. 
The QA pairs are saved to a .txt file. """

import guidance
from PyPDF2 import PdfReader
import re
import json
from guidance import gen, models, user, assistant, system
import tiktoken
import multiprocessing
import time
from datetime import datetime
from credentials import API_KEY


PDF_NAME = ""  # base name of PDF (without.pdf)

PROMPT = '''You're a compliance documentation expert. Generate 5-10 question-answer pairs following these rules:
            1. Each question must start with "QX: " (e.g., "Q1: What is the requirement for XYZ?")
            2. Each answer must start with "A X: " (e.g., "A1: The requirement is ABC.")
            3. Each answer must include a "Text Snippet:" with the exact 1-5 sentences from the context.
            4. Focus on technical specifications, requirements, and compliance.'''


# Chunking function
def chunk_text(text, chunk_size=1000):
    """
    chunk text

    Args:
    text(str): parsed document text
    size(int): chunk size (in words)

    Returns:
    list of chunks
    """
    words = re.split(r'\s+', text)
    chunks = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i in range(0, len(words), chunk_size):
        chunk = {
            "id": len(chunks) + 1,  # sequential ID
            "text": " ".join(words[i:i+chunk_size]),
            "timestamp": timestamp
        }
        chunks.append(chunk)

    return chunks


# Guidance function
@guidance(dedent=False)
def qa_generator(model, chunk):
    """
    Used MS Guidance package to synthetically generate QA Pairs
    Args:
    model: A Guidance model object used to build and execute the prompt
    chunk (str): A chunk of text (usually from a parsed PDF) to be used as the document excerpt for QA generation

    Returns:
    guidance.program: A Guidance program object that, when executed with the model, returns the generated QA pairs
    """
    with system():
        model += PROMPT

    with user():
        model += f"DOCUMENT EXCERPT:\n{chunk}\n\nGenerate 5-10 QA pairs"

    with assistant():
        model += gen('response', temperature=0.2)

    return model


# Main method
def main():
    """
    main method:
    1. initialized openrouted deepseek model
    2. parses pdf
    3. creates chunks according to chunk_text function
    4. Generates QA pairs using openrouter model and one chunk of text

    Saves LLM response for corresponding chunks in .txt
    """
    # Initializing OpenRouter model
    OPENROUTER_API_KEY = API_KEY
    encoding = tiktoken.get_encoding("cl100k_base")

    llm = models.OpenAI(
        model="deepseek/deepseek-r1:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        tokenizer=encoding,
        timeout=30
    )


    # Load and extract text from the PDF
    reader = PdfReader(f"{PDF_NAME}.pdf")
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = chunk_text(text)
    print(f'Created {len(chunks)} chunks')

    # Save chunk
    with open(f"chunks_{PDF_NAME}.json", "w") as chunk_file:
        json.dump(chunks, chunk_file, indent=2)

    # Rate limit variables
    requests_made = 0
    start_time = time.time()

    with open(f"output_{PDF_NAME}.txt", 'w') as txt_file:
        txt_file.write(f"LLM Output - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for i, chunk in enumerate(chunks):
            try:
                print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
                elapsed_time = time.time() - start_time

                # Enforce rate limiting: 10 requests per 10 seconds
                if requests_made >= 10:
                    wait_time = max(0, 10 - elapsed_time)
                    print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    requests_made = 0  # Reset counter
                    start_time = time.time()  # Reset timer

                # Generate QA pairs
                program = qa_generator(chunk=chunk['text'])
                result = program(llm)
                response = result["response"].strip()

                # Write LLM output to file
                txt_file.write(f"\n=== Chunk {i+1} ===\n")
                txt_file.write(response + "\n")

                # Update request counter
                requests_made += 1

                print(f"Chunk {i+1} processed.")

            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")

    print(f"\nChunks saved to chunks_{PDF_NAME}.json")
    print(f"LLM output saved to output_{PDF_NAME}.txt")


if __name__ == "__main__":
    # using multiprocessing to create multiple threads for processing different chunks at the same time
    multiprocessing.freeze_support()
    main()
