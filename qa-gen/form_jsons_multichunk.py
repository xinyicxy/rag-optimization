import json
import re
import os


def load_files(base_name):
    """Load QnA output and chunk data files."""
    output_file = f"data/infra/{base_name}-multichunk-output.txt"
    chunks_file = f"data/infra/{base_name}-chunks.json"

    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = f.readlines()

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # print(output_data)
    # print(chunks_data)
    return output_data, chunks_data


def extract_qna(output_data):
    qna_list = []
    chunk_numbers = []

    question, answer = None, None
    excerpt_1, excerpt_2 = None, None

    for line in output_data:
        chunk_match = re.match(r'=== Chunk Pair (\d+) & (\d+) ===', line)

        if chunk_match:
            chunk_numbers = [int(chunk_match.group(1)), int(chunk_match.group(2))]
            # print(chunk_numbers)

        q_match = re.search(r'Q\d+:\s*(.+)', line)
        a_match = re.search(r'A\d+:\s*(.+)', line)
        excerpt_1_match = re.search(r'\*\*Text Snippet \(Excerpt 1\):\*\*\s*"(.+?)"', line)
        excerpt_2_match = re.search(r'\*\*Text Snippet \(Excerpt 2\):\*\*\s*"(.+?)"', line)

        if q_match:
            question = q_match.group(1).strip('*').strip()
            # print(question)

        elif a_match:
            answer = a_match.group(1).strip('*').strip()
            # Remove "Excerpt 1" and "Excerpt 2" references but keep section numbers
            answer = re.sub(r'Excerpt 1\s*\(|Excerpt 2\s*\(', '(', answer)
            #print(answer)

        elif excerpt_1_match:
            excerpt_1 = excerpt_1_match.group(1).strip('*').strip()
            print(excerpt_1)

        elif excerpt_2_match:
            excerpt_2 = excerpt_2_match.group(1).strip('*').strip()
            # print(excerpt_2)

        if question and answer and excerpt_1 and excerpt_2:
            qna_list.append({
                "chunks": chunk_numbers,
                "question": question,
                "answer": answer,
                "context": segment_snippet(excerpt_1) + segment_snippet(excerpt_2)
            })
            question, answer, excerpt_1, excerpt_2 = None, None, None, None  # Reset

    return qna_list


def segment_snippet(snippet):
    """
    Segments the snippet into a list of parts by splitting on '...' or '[...]',
    removing the ellipses from the output.
    """
    segments = re.split(r'\.\.\.|\[\.\.\.\]', snippet)
    return [segment.strip() for segment in segments if segment.strip()]


def process_qna(base_name):
    output_data, chunks_data = load_files(base_name)
    qna_list = extract_qna(output_data)

    output_dir = "expanded_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{base_name}-multichunk-expanded.json")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(qna_list, f, indent=4)

    print(f"Expanded QnA saved to {output_filename}")


# Run the script for multiple QnA files
if __name__ == "__main__":
    process_qna("infra-1")
