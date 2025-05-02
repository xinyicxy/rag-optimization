"""parsonsgpt metrics - this was never used"""
import json
import pandas as pd
from ragas.metrics import (
    answer_relevancy,
    answer_correctness
)
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
from ragas.llms import LangchainLLMWrapper
from langchain_openai import OpenAI
import csv

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
##################################################### need to concatenate all batches into 1 csv
with open('results_rfp_test.csv') as f:
    results = list(csv.DictReader(f))
with open('test_rfp_groundtruths.csv') as f:  ##################### write script to get ground truth ans + RFP type
    ground_truth_ans = list(csv.DictReader(f))


df = pd.DataFrame(results)
groundtruth_df = pd.DataFrame(ground_truth_ans)
# Put question, response, and ground truth together
df['ground_truth'] = groundtruth_df['ans']
# add in metadata
df['RFP_type'] = groundtruth_df['RFP_type']


def compute_metrics(row):
    # lots of type fixes here -> making sure they are list of strings
    # retrieved_contexts = row['retrieved_context']
    # ground_truths = [entry[1] for entry in row['ground_truth_context']]

    # if isinstance(retrieved_contexts, str):
    #     retrieved_contexts = [retrieved_contexts]  # putting single string in list
    # elif not isinstance(retrieved_contexts, list):
    #     raise TypeError(f"Unexpected type for retrieved_context: {type(retrieved_contexts)}")

    # if isinstance(ground_truths, str):
    #     ground_truths = [ground_truths]
    # elif not isinstance(ground_truths, list):
    #     raise TypeError(f"Unexpected type for ground_truth_context: {type(ground_truths)}")

    # constructing evaluation data set
    dataset = EvaluationDataset.from_list([
        {
            "user_input": row["Prompt"],
            "response": row["Response"],
            # "retrieved_contexts": retrieved_contexts,
            "reference": row['ground_truth'],
            # "ground_truths": ground_truths  # TODO: this one may not be needed
        }
    ])

    # actaully compute the metrics -> roughly 1 to 4 seconds / example
    scores = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            # faithfulness,
            # context_recall,
            # context_precision,
            answer_correctness
        ]
    )

    return pd.Series({
        "answer_relevancy": scores["answer_relevancy"],
        # "faithfulness": scores["faithfulness"],
        # "context_recall": scores["context_recall"],
        # "context_precision": scores["context_precision"],
        "answer_correctness": scores["answer_correctness"]
    })


df = pd.concat([df, df.apply(compute_metrics, axis=1)], axis=1)
print(df.head())


# computing EM and F1
def exact_match(row):
    return int(row['Response'].strip().lower() == row['ground_truth'].strip().lower())


def compute_f1(row):
    pred_tokens = set(row['Response'].lower().split())
    true_tokens = set(row['ground_truth'].lower().split())

    if len(true_tokens) == 0:
        return 0.0

    precision = len(pred_tokens & true_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(pred_tokens & true_tokens) / len(true_tokens)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


df['EM'] = df.apply(exact_match, axis=1)
df['F1'] = df.apply(compute_f1, axis=1)

# define grouping dimensions
grouping_dimensions = [
    ('full_dataset', None),
    ('RFP_type', 'RFP_type')
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
        for col in ['answer_relevancy', 'answer_correctness', 'EM', 'F1']:
            group[col] = group[col].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)

        # Compute average metrics
        metrics_avg = group[[
            'answer_relevancy', 'answer_correctness', 'EM', 'F1'
        ]].mean().to_dict()

        # Compute latency metrics
        #latency_avg = group['timing'].apply(pd.Series).mean().to_dict()

        group_results[name] = {
            **metrics_avg,
            #**{'avg_'+k: v for k, v in latency_avg.items()},
            'sample_size': len(group)
        }

    report[group_name] = group_results

# save report
with open('metrics_report_rfp.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Metrics report generated at metrics_report_rfp.json")
