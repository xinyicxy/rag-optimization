import os
import json
import pandas as pd
import re

def parse_metrics_folder(folder_path, output_csv_path="combined_metrics.csv"):
    records = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)

            # Updated pattern: metrics_k5_typesentences_size10.json
            match = re.match(r"metrics_k(\d+)_type(\w+)_size(\d+)\.json", filename)
            if not match:
                print(f"Skipping unrecognized file name format: {filename}")
                continue

            k, chunk_type, chunk_size = match.groups()

            with open(filepath, "r") as f:
                data = json.load(f)

            row = {
                "chunk_type": chunk_type,
                "chunk_size": int(chunk_size),
                "k": int(k),
            }

            # Add manually_edited metrics
            if "manually_edited" in data:
                for edit_flag, metrics in data["manually_edited"].items():
                    for metric, value in metrics.items():
                        col_name = f"manually_edited_{edit_flag}_{metric}"
                        row[col_name] = value

            # Add chunk_length metrics
            if "chunks_length" in data:
                for length, metrics in data["chunks_length"].items():
                    for metric, value in metrics.items():
                        col_name = f"chunks_length_{length}_{metric}"
                        row[col_name] = value

            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved combined metrics to {output_csv_path}")


if __name__ == "__main__":
    parse_metrics_folder(folder_path = "./outputs/", output_csv_path="special_combined_metrics.csv")