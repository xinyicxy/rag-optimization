import os
import json
import re


def extract_rfp_info(filename):
    """Extracts RFP_id and RFP type from the filename."""
    # match = re.match(r"(infra|natsec)-(\d+)-expanded\.json", filename)
    match = re.match(r"(infra|natsec)[-_](\d+)[-_]?(multichunk)?[-_]?expanded\.json", filename)
    if match:
        rfp_type = match.group(1)
        rfp_id = f"{rfp_type}_{match.group(2)}"
        print(rfp_id, rfp_type)
        return rfp_id, rfp_type

    return None, None


def combine_json_files(directory, output_filename):
    combined_data = []
    question_id = 1

    for filename in os.listdir(directory):
        if filename.endswith("-expanded.json"):
            file_path = os.path.join(directory, filename)
            rfp_id, rfp_type = extract_rfp_info(filename)     
            if not rfp_id or not rfp_type:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                for entry in data:
                    combined_entry = {
                        "question_id": question_id,
                        "question": entry.get("question", ""),
                        "answer": entry.get("answer", ""),
                        "snippet": entry.get("snippet", None),  # Some entries may not have snippets
                        "context": entry.get("context", []),
                        "chunks": entry.get("chunks", [entry.get("chunk")]) if "chunk" in entry else entry.get("chunks", []),
                        "RFP_id": rfp_id,
                        "RFP_type": rfp_type,
                        "manually_edited": False
                    }
                    combined_data.append(combined_entry)
                    question_id += 1

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4)

    print(f"Final combined JSON saved to {output_filename}")


if __name__ == "__main__":
    combine_json_files("expanded_outputs", "final_combined.json")
