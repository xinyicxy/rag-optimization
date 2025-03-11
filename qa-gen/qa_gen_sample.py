import guidance
from PyPDF2 import PdfReader
import re
import json
from guidance import gen, models, user, assistant, system
import tiktoken
import multiprocessing
import time
from datetime import datetime


def main():
    # initializing open router model 
    OPENROUTER_API_KEY = "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"
    encoding = tiktoken.get_encoding("cl100k_base")

    llm = models.OpenAI(
        model="deepseek/deepseek-r1:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        tokenizer=encoding,
        timeout=30
    )

    # chunking
    def chunk_text(text, chunk_size=1000):
        words = re.split(r'\s+', text)
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    # load rfp in 
    # ROOCHI NOTE: this part takes less than a second no problem here
    reader = PdfReader("rfp-example-1.pdf") #NOTE: put pdf in here
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = chunk_text(text)
    print(f'Created {len(chunks)} chunks')

    # GUIDANCE FUNCTION ############
    # ROOCHI NOTE: DEDENT=FALSE IS NEEDED
    # ROOCHI NOTE: MODIFY SYSTEM PROMPT - prompt was chatpgted
    @guidance(dedent=False)
    def qa_generator(model, chunk):
        with system():
            model += '''You're a compliance documentation expert. Generate 5-10 question-answer pairs following these rules:
            1. Each question must start with "QX: " (e.g., "Q1: What is the requirement for XYZ?")
            2. Each answer must start with "A X: " (e.g., "A1: The requirement is ABC.")
            3. Each answer must include a "Text Snippet:" with the exact 1-5 sentences from the context
            4. Focus on technical specifications, requirements, and compliance.'''

        with user():
            model += f"DOCUMENT EXERPT:\n{chunk}\n\nGenerate 5-10 QA pairs"

        with assistant():
            model += gen('response', temperature=0.2)

        return model

    # Process all chunks
    final_qas = []
    for i, chunk in enumerate(chunks):
        try:
            print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
            start_time = time.time()

            program = qa_generator(chunk=chunk)
            result = program(llm)
            response = result["response"].strip()

            # ROOCHI NOTE: DEBUG
            print(f"\nRaw Response (Chunk {i+1}):\n{response}\n")


            # ROOCHI NOTE: STRING MATCHING DONE W CHATGPT AND IS INCORRECT!
            pairs = re.findall(r'\*\*Q\d+: (.*?)\*\*\s*A\d+: (.*?)\s*\*\*Text Snippet:\*\* (.*?)$', response, re.DOTALL | re.MULTILINE)

            if not pairs:
                print(f"⚠️ No QA pairs found in chunk {i+1}. Logging raw response")
                with open(f"failed_chunk_{i+1}.txt", "w") as f:
                    f.write(response)

            for question, answer, text_snippet in pairs:
                final_qas.append({
                    "question": question.strip() + ("?" if not question.strip().endswith("?") else ""),
                    "answer": answer.strip(),
                    "context": text_snippet.strip(),
                    "chunk": chunk
                })

            # ROOCHI NOTE: this is for rate limit stuff - sleep timer may not be necessary
            if i < len(chunks) - 1:
                time.sleep(1.5)

            print(f"Chunk {i+1} processed in {time.time()-start_time:.1f}s, extracted {len(pairs)} QA pairs")

        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")

    # json dump
    # ROOCHI NOTE: MODIFY TO SAVE TO JSON AS WE GO INSTEAD OF ALL AT THE END
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"qa_pairs_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(final_qas, f, indent=2)

    print(f"\nSuccessfully saved {len(final_qas)} QA pairs to {filename}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()