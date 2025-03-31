import os
import json
import re
import pandas as pd


def extract_metrics_from_files(directory):
    """ function to extract all of the metrics data"""
    data = []

    for filename in os.listdir(directory):
        if filename.startswith("metrics_") and filename.endswith(".json"):

            # get hyperparams from exp name
            match = re.search(r'k(\d+)_type(\w+)_size(\d+)', filename)
            if not match:
                print(f"Skipping file {filename}: Pattern not found")
                continue

            k, type_, size = match.groups()

            # read and save json
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                try:
                    metrics = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {filename}")
                    continue

            # store the flattened metrics
            row = {"k": int(k), "type": type_, "size": int(size), **metrics}
            data.append(row)


    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # example usage
    directory = "./outputs_r1/"  # REPLACE
    df_metrics = extract_metrics_from_files(directory)
    check = df_metrics.at[0, 'RFP_type']
    print(check)
    # print(df_metrics.head())