import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
import re


with open('morehopqa_final.json', 'r') as file:
    morehop_full = json.load(file)

with open('morehopqa_final_150samples.json', 'r') as file:
    morehop_150 = json.load(file)


# Extract contexts
def remove_br(text):
    return re.sub(r'<br>', '', text)


paragraphs_full = []
for id in morehop_full:
    contexts = id.get("context", [])
    for item in contexts:
        context = item[1]
        context_cat = "".join(context)
        context_cat = remove_br(context_cat)
        paragraphs_full.append(context_cat)

paragraphs_150 = []
for id in morehop_150:
    contexts = id.get("context", [])
    for item in contexts:
        context = item[1]
        context_cat = "".join(context)
        context_cat = remove_br(context_cat)
        paragraphs_150.append(context_cat)


# Save to PDF
def create_pdf(output_filename, contexts):
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    style = styles['Normal']

    for context_string in contexts:
        paragraph = Paragraph(context_string, style)
        story.append(paragraph)

        # Add a space between contexts
        space = Paragraph("<br/><br/>", style)  # Two <br/> for extra space
        story.append(space)

    doc.build(story)


create_pdf("morehop_contexts_full.pdf", paragraphs_full)
create_pdf("morehop_contexts_150.pdf", paragraphs_150)
