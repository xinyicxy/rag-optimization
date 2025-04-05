import json
import pandas as pd
import argparse

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
from ragas.llms import LangchainLLMWrapper
from langchain_openai import OpenAI
from credentials import OPENAI_KEY

# arg parse!
parser = argparse.ArgumentParser(description="Process MoreHop queries with OpenAI and ChromaDB.")
parser.add_argument("chunk_type", type=str, choices=["characters", "words", "sentences", "paragraphs", "pages"],
                    default="words", help="Chunking method to use.")
parser.add_argument("chunk_size", type=int, default=1000,
                    help="Size of each chunk.")
parser.add_argument("top_k", type=int, default=2, help="K chunks retrieved during search.")

args = parser.parse_args()
CHUNK_TYPE = args.chunk_type
CHUNK_SIZE = args.chunk_size
TOP_K = args.top_k

# setting api key
OPENAI_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_KEY  # Set for RAGAS

"""
# configuring faster model for eval
gpt3_llm = LangchainLLMWrapper(model_name=OpenAI(model_name="gpt-3.5-turbo", temperature=0))
answer_relevancy.llm = gpt3_llm
faithfulness.llm = gpt3_llm
context_recall.llm = gpt3_llm
context_precision.llm = gpt3_llm
answer_correctness.llm = gpt3_llm
"""

# load experiment data
exp_filename = f"outputs/morehop_exp_output_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.json"
with open(exp_filename) as f:
    data = json.load(f)

# convert to pandas
df = pd.DataFrame(data['results'])

# add in metadata
# df['chunks_length'] = df['metadata'].apply(lambda x: len(x['chunks']))
df['answer_type'] = df['metadata'].apply(lambda x: x['answer_type'])


def compute_metrics(row):
    # lots of type fixes here -> making sure they are list of strings
    retrieved_contexts = row['retrieved_context']
    ground_truths = [entry[1] for entry in row['ground_truth_context']]

    if isinstance(retrieved_contexts, str):
        retrieved_contexts = [retrieved_contexts]  # putting single string in list
    elif not isinstance(retrieved_contexts, list):
        raise TypeError(f"Unexpected type for retrieved_context: {type(retrieved_contexts)}")

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif not isinstance(ground_truths, list):
        raise TypeError(f"Unexpected type for ground_truth_context: {type(ground_truths)}")

    # constructing evaluation data set
    dataset = EvaluationDataset.from_list([
        {
            "user_input": row["question"],
            "response": row["llm_response"],
            "retrieved_contexts": retrieved_contexts,
            "reference": row['ground_truth_answer'],
            "ground_truths": ground_truths  # TODO: this one may not be needed
        }
    ])

    # actaully compute the metrics -> roughly 1 to 4 seconds / example
    scores = evaluate(
        dataset,
        metrics=[
            context_recall
        ]
    )

    return pd.Series({
        "context_recall": scores["context_recall"]
    })


df = pd.concat([df, df.apply(compute_metrics, axis=1)], axis=1)
print(df.head())


# computing EM
def exact_match(row):
    return int(row['llm_response'].strip().lower() == row['ground_truth_answer'].strip().lower())


df['EM'] = df.apply(exact_match, axis=1)
df['F1'] = df.apply(compute_f1, axis=1)

# define grouping dimensions
grouping_dimensions = [
    ('full_dataset', None),
    ('answer_type', 'answer_type')
    #('chunks_length', 'chunks_length')
]

# compute metrics for each group
report = {}

for group_name, group_key in grouping_dimensions:
    if group_key:
        groups = df.groupby(group_key)
    else:
        # Full dataset
        groups = [('full_dataset', df)]

    group_results = {}
    for name, group in groups:
        # Ensure list-type columns are converted to numeric values before computing mean
        for col in ['context_recall', 'EM']:
            group[col] = group[col].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)

        # Compute average metrics
        metrics_avg = group[[
            'context_recall', 'EM'
        ]].mean().to_dict()

        # Compute latency metrics
        latency_avg = group['timing'].apply(pd.Series).mean().to_dict()

        group_results[name] = {
            **metrics_avg,
            **{'avg_'+k: v for k, v in latency_avg.items()},
            'sample_size': len(group)
        }

    report[group_name] = group_results

# save report
metrics_filename = f"outputs/morehop_metrics_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.json"
with open(metrics_filename, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Metrics report generated at morehop_metrics_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.json")
