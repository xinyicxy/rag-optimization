# Extracts and stores all contexts in a PDF
# Creates dictionary for context labels

import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
import re
import random


with open('morehopqa_final.json', 'r') as file:
    morehop_full = json.load(file)

with open('morehopqa_final_150samples.json', 'r') as file:
    morehop_150 = json.load(file)


# Extract contexts
def clean_contexts(text):
    return re.sub(r'<br>', ' ', text)  # remove extra <br>s


random.seed(12345)

paragraphs_full = []
dict_full = []

for id in morehop_full:
    contexts = id.get("context", [])
    for item in contexts:
        context = item[1]
        context = [sentence.strip() for sentence in context] # remove extra whitespaces
        context_cat = " ".join(context)
        # context_cat = clean_contexts(context_cat)
        paragraphs_full.append(context_cat)

        label = item[0]
        dict_full.append({"label": label, "context": context})

paragraphs_full = list(set(paragraphs_full))  # remove duplicates
random.shuffle(paragraphs_full)  # shuffle chunks so they aren't in order


paragraphs_150 = []
dict_150 = []

for id in morehop_150:
    contexts = id.get("context", [])
    for item in contexts:
        context = item[1]
        context = [sentence.strip() for sentence in context]
        context_cat = " ".join(context)
        # context_cat = clean_contexts(context_cat)
        paragraphs_150.append(context_cat)

        label = item[0]
        dict_150.append({"label": label, "context": context})

paragraphs_150 = list(set(paragraphs_150))
random.shuffle(paragraphs_150)


# Save to PDF
def create_pdf(output_filename, contexts):
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    style = styles['Normal']
    for context_string in contexts:
        paragraph = Paragraph(context_string, style)
        story.append(paragraph)
        space = Paragraph("PARAGRAPH BREAK", style)  # Specify paragraph break
        story.append(space)
    story.pop()  # Remove the last paragraph break
    doc.build(story)


# Create PDF of contexts
create_pdf("morehop_contexts_full.pdf", paragraphs_full)
create_pdf("morehop_contexts_150.pdf", paragraphs_150)

# Create context dictionary mapping context labels to context
with open("context_dict_full.json", "w") as json_file:
    json.dump(dict_full, json_file, indent=4)

with open("context_dict_150.json", "w") as json_file:
    json.dump(dict_150, json_file, indent=4)

