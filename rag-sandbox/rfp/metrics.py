import json
import pandas as pd
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
import argparse
from credentials import OPENAI_KEY

# setting api key
os.environ["OPENAI_API_KEY"] = OPENAI_KEY  # Set for RAGAS

"""
# TODO? configuring faster model for eval
gpt3_llm = LangchainLLMWrapper(model_name=OpenAI(model_name="gpt-3.5-turbo", temperature=0))
answer_relevancy.llm = gpt3_llm
faithfulness.llm = gpt3_llm
context_recall.llm = gpt3_llm
context_precision.llm = gpt3_llm
answer_correctness.llm = gpt3_llm
"""

# arg parse!
parser = argparse.ArgumentParser(
    description="Process RFP queries with OpenAI and ChromaDB.")
parser.add_argument("chunk_type", type=str,
                    help="Type of chunking (e.g., 'words' or 'sentences').")
parser.add_argument("chunk_size", type=int,
                    help="Size of chunks for document processing.")
parser.add_argument("top_k", type=int,
                    help="K chunks retrieved during search.")

args = parser.parse_args()
CHUNK_TYPE = args.chunk_type
CHUNK_SIZE = args.chunk_size
TOP_K = args.top_k

# load experiment data TODO: change the filename
exp_filename = f"outputs/exp_output_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.json"
with open(exp_filename) as f:
    data = json.load(f)

# convert to pandas
df = pd.DataFrame(data['results'])

# add in metadata
df['chunks_length'] = df['metadata'].apply(lambda x: len(x['chunks']))
df['manually_edited'] = df['metadata'].apply(lambda x: x['manually_edited'])
df['RFP_id'] = df['metadata'].apply(lambda x: x['RFP_id'])
df['RFP_type'] = df['metadata'].apply(lambda x: x['RFP_type'])
df['question_id'] = df['metadata'].apply(lambda x: x['question_id'])


# hack for now?
def is_negative_rejection(response):
    # "Insufficient information to answer question based on given context"
    return "insufficient information" in response.lower()


# filtering the negative rejection questions and computing stats!
# filter on question id TODO replace 400 with correct quesiton id threshold
negative_rejection_df = df[df['question_id'] >= 400]
total_negative_rejections = int(
    negative_rejection_df['llm_response'].apply(is_negative_rejection).sum())
total_negative_questions = len(negative_rejection_df)
negative_rejection_percentage = (
    total_negative_rejections / total_negative_questions * 100) if total_negative_questions > 0 else 0


def compute_metrics(row):
    # lots of type fixes here -> making sure they are list of strings
    retrieved_contexts = row['retrieved_context']
    ground_truths = row['ground_truth_context']

    if isinstance(retrieved_contexts, str):
        # putting single string in list
        retrieved_contexts = [retrieved_contexts]
    elif not isinstance(retrieved_contexts, list):
        raise TypeError(
            f"Unexpected type for retrieved_context: {type(retrieved_contexts)}")

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif not isinstance(ground_truths, list):
        raise TypeError(
            f"Unexpected type for ground_truth_context: {type(ground_truths)}")

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
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
            answer_correctness
        ]
    )

    return pd.Series({
        "answer_relevancy": scores["answer_relevancy"],
        "faithfulness": scores["faithfulness"],
        "context_recall": scores["context_recall"],
        "context_precision": scores["context_precision"],
        "answer_correctness": scores["answer_correctness"]
    })


df = pd.concat([df, df.apply(compute_metrics, axis=1)], axis=1)
print(df.head())


# computing EM and F1
def exact_match(row):
    return int(row['llm_response'].strip().lower() == row['ground_truth_answer'].strip().lower())


def compute_f1(row):
    pred_tokens = set(row['llm_response'].lower().split())
    true_tokens = set(row['ground_truth_answer'].lower().split())

    if len(true_tokens) == 0:
        return 0.0

    precision = len(pred_tokens & true_tokens) / \
        len(pred_tokens) if pred_tokens else 0
    recall = len(pred_tokens & true_tokens) / len(true_tokens)
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0
    return f1


df['EM'] = df.apply(exact_match, axis=1)
df['F1'] = df.apply(compute_f1, axis=1)

# define grouping dimensions
grouping_dimensions = [
    ('full_dataset', None),
    ('RFP_id', 'RFP_id'),
    ('RFP_type', 'RFP_type'),
    ('chunks_length', 'chunks_length'),
    ('manually_edited', 'manually_edited')
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
        for col in ['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision', 'answer_correctness', 'EM', 'F1']:
            group[col] = group[col].apply(lambda x: sum(
                x) / len(x) if isinstance(x, list) else x)

        # Compute average metrics
        metrics_avg = group[[
            'answer_relevancy', 'faithfulness', 'context_recall',
            'context_precision', 'answer_correctness', 'EM', 'F1'
        ]].mean().to_dict()

        # Compute latency metrics
        latency_avg = group['timing'].apply(pd.Series).mean().to_dict()

        group_results[name] = {
            **metrics_avg,
            **{'avg_'+k: v for k, v in latency_avg.items()},
            'sample_size': len(group)
        }

    report[group_name] = group_results

# adding negative rejection stuff
report.update({
    "total_negative_questions": total_negative_questions,
    "total_negative_rejections": int(total_negative_rejections),
    "negative_rejection_percentage": negative_rejection_percentage
})

# save report
metrics_filename = f"outputs/metrics_k{TOP_K}_type{CHUNK_TYPE}_size{CHUNK_SIZE}.json"
with open(metrics_filename, 'w') as f:
    json.dump(report, f, indent=2)

print("Metrics report generated at metrics_report.json")
