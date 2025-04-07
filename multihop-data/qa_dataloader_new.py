# Extract questions into csv
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


file_path = 'morehopqa_120.json'
data = read_json_file(file_path)

res = []

for item in data:
    # Add the original questions
    qns = item["previous_question"]
    answer_type = item["previous_answer_type"]

    if answer_type == 'date':  # convert dates to yyyy-mm-dd
        date = pd.to_datetime(item['previous_answer'], errors='coerce')
        if not pd.isna(date):  # keep original answer if date format not recognized
            item['previous_answer'] = date.strftime('%Y-%m-%d')

    toAdd = {"prompt": qns}
    res.append(toAdd)


# Convert to csv
df = pd.DataFrame.from_dict(res)
df.to_csv("multihop120_questions.csv", index=False)

# batch_size = 15
# batch = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
# # Write each batch to a separate CSV
# for i, chunk in enumerate(batch, start=1):
#     chunk.to_csv(f"multihop150_questions{i}.csv", index=False)
