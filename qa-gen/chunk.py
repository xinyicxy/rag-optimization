
from PyPDF2 import PdfReader
import re


def chunk_text(text, chunk_size=1000):
    words = re.split(r'\s+', text)
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words),
                                                             chunk_size)]
    return chunks


# example: extract from pdf
reader = PdfReader("rfp-example-1.pdf")
text = " ".join([page.extract_text() for page in reader.pages])

# chunking test
print('start chunking')
chunks = chunk_text(text)
print('complete')
print(len(chunks))