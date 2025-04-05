# Extract questions into csv/excel format, run this while in the multihop-data folder
import json
import pandas as pd


def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The data from the JSON file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {file_path}")
        return None


file_path = 'data/morehopqa_final.json'
data = read_json_file(file_path)


# Extract the questions and answers
res = []

for item in data:
    # Add the new questions
    qns = item["question"]
    ans = item["answer"]
    answer_type = item["answer_type"]
    reasoning_type = item["reasoning_type"]
    id = item["_id"]
    toAdd = {"Question": qns, "Answer": ans, "Answer Type": answer_type,
             "Reasoning Type": reasoning_type, "id": id}
    res.append(toAdd)

    # Add the old questions
    qns = item["previous_question"]
    ans = item["previous_answer"]
    answer_type = item["previous_answer_type"]
    reasoning_type = "Original"
    toAdd = {"Question": qns, "Answer": ans, "Answer Type": answer_type,
             "Reasoning Type": reasoning_type, "id": id}
    res.append(toAdd)


# Convert to csv
df = pd.DataFrame.from_dict(res)
df = df.drop_duplicates(subset=["Question", "Answer"])
df.to_csv("multihopqa_pairs.csv", index=False)
