
import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import os
import re
from chunking import load_pdf
from glob import glob


def plot_metric_for_natsec(metrics_folder, metric_name):
    records = []

    for filename in os.listdir(metrics_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(metrics_folder, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Determine retrieval type
            if 'reranking_filtering' in filename:
                retrieval_type = 'reranking_filtering'
            elif 'reranking' in filename:
                retrieval_type = 'reranking'
            elif 'filtering' in filename:
                retrieval_type = 'filtering'
            else:
                retrieval_type = 'regular'

            # Determine chunking type
            if 'typewords_size50' in filename:
                chunking = 'words_50'
            elif 'typesentences_size10' in filename:
                chunking = 'sentences_10'
            elif 'typeparagraphs_size1' in filename:
                chunking = 'paragraphs_1'
            else:
                raise ValueError(f"Unknown chunking type in filename: {filename}")

            # Extract metric for RFP_type 'natsec'
            rfp_metrics = data.get('RFP_type', {}).get('natsec', {})
            value = rfp_metrics.get(metric_name)

            if value is not None:
                records.append({
                    'chunking': chunking,
                    'retrieval': retrieval_type,
                    'value': value
                })

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x='chunking',
        y='value',
        hue='retrieval',
        palette='Set2'
    )

    plt.title(f"{metric_name.replace('_', ' ').title()} (Natsec only) by Chunking and Retrieval Method")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.xlabel("Chunking Type")
    plt.legend(title="Retrieval")
    plt.tight_layout()
    plt.show()


def plot_metric_by_chunks_and_manual_edit(metrics_folder, metric_name):
    records_chunks = []
    records_manual = []

    for filename in os.listdir(metrics_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(metrics_folder, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Determine retrieval type
            if 'reranking_filtering' in filename:
                retrieval_type = 'reranking_filtering'
            elif 'reranking' in filename:
                retrieval_type = 'reranking'
            elif 'filtering' in filename:
                retrieval_type = 'filtering'
            else:
                retrieval_type = 'regular'

            # Determine chunking type
            if 'typewords_size50' in filename:
                chunking = 'words_50'
            elif 'typesentences_size10' in filename:
                chunking = 'sentences_10'
            elif 'typeparagraphs_size1' in filename:
                chunking = 'paragraphs_1'
            else:
                raise ValueError(f"Unknown chunking type in filename: {filename}")

            # Collect from chunks_length 1 and 2
            chunks_data = data.get('chunks_length', {})
            for chunk_len_key in ['1', '2']:
                metric_value = chunks_data.get(chunk_len_key, {}).get(metric_name)
                if metric_value is not None:
                    records_chunks.append({
                        'chunk_length': int(chunk_len_key),
                        'chunking': chunking,
                        'retrieval': retrieval_type,
                        'value': metric_value
                    })

            # Collect from manually_edited true/false
            manual_data = data.get('manually_edited', {})
            for manual_flag in ['true', 'false']:
                metric_value = manual_data.get(manual_flag, {}).get(metric_name)
                if metric_value is not None:
                    records_manual.append({
                        'manually_edited': manual_flag,
                        'chunking': chunking,
                        'retrieval': retrieval_type,
                        'value': metric_value
                    })

    # Plotting chunks_length
    df_chunks = pd.DataFrame(records_chunks)
    for cl in [1, 2]:
        plt.figure(figsize=(10, 6))
        subset = df_chunks[df_chunks['chunk_length'] == cl]
        sns.barplot(
            data=subset,
            x='chunking',
            y='value',
            hue='retrieval',
            palette='Set2'
        )
        plt.title(f"{metric_name.replace('_', ' ').title()} for Chunk Length {cl}")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.xlabel("Chunking Type")
        plt.legend(title="Retrieval")
        plt.tight_layout()
        plt.show()

    # Plotting manually_edited
    df_manual = pd.DataFrame(records_manual)
    for flag in ['true', 'false']:
        plt.figure(figsize=(10, 6))
        subset = df_manual[df_manual['manually_edited'] == flag]
        sns.barplot(
            data=subset,
            x='chunking',
            y='value',
            hue='retrieval',
            palette='Set1'
        )
        flag_title = "Manually Edited" if flag == "true" else "Not Manually Edited"
        plt.title(f"{metric_name.replace('_', ' ').title()} for {flag_title}")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.xlabel("Chunking Type")
        plt.legend(title="Retrieval")
        plt.tight_layout()
        plt.show()


def plot_metric_full_dataset(metrics_folder, metric_name):
    records = []

    for filename in os.listdir(metrics_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(metrics_folder, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Determine retrieval type
            if 'reranking_filtering' in filename:
                retrieval_type = 'reranking_filtering'
            elif 'reranking' in filename:
                retrieval_type = 'reranking'
            elif 'filtering' in filename:
                retrieval_type = 'filtering'
            else:
                retrieval_type = 'regular'

            # Determine chunking type
            if 'typewords_size50' in filename:
                chunking = 'words_50'
            elif 'typesentences_size10' in filename:
                chunking = 'sentences_10'
            elif 'typeparagraphs_size1' in filename:
                chunking = 'paragraphs_1'
            else:
                raise ValueError(f"Unknown chunking type in filename: {filename}")

            # Get metric from full_dataset
            full_data = data.get('full_dataset', {}).get('full_dataset', {})
            metric_value = full_data.get(metric_name)
            if metric_value is not None:
                records.append({
                    'chunking': chunking,
                    'retrieval': retrieval_type,
                    'value': metric_value
                })

    # Create DataFrame and plot
    df = pd.DataFrame(records)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x='chunking',
        y='value',
        hue='retrieval',
        palette='Set3'
    )
    plt.title(f"{metric_name.replace('_', ' ').title()} Across Full Dataset")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.xlabel("Chunking Type")
    plt.legend(title="Retrieval")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    METRICS_DIR = "./outputs/"
    DOCUMENTS_DIR = "./documents/"
    METRIC = "answer_correctness"

    METRICS_TO_PLOT = [
      "answer_relevancy",
      "faithfulness",
      "context_recall",
      "context_precision",
      "answer_correctness",
      "EM",
      "F1",
      "avg_retrieve_context",
      "avg_llm_response",
      "avg_total",
      "sample_size"]

    CSV = "metric_plot_data.csv"

    #plot_metric_for_natsec("./outputs/rerankvreg/", metric_name = "answer_correctness")
    #plot_metric_for_natsec("./outputs/rerankvreg/", metric_name = "faithfulness")
    #plot_metric_for_natsec("./outputs/rerankvreg/", metric_name = "answer_relevancy")
    #plot_metric_for_natsec("./outputs/rerankvreg/", metric_name =  "context_recall")

    plot_metric_by_chunks_and_manual_edit("./outputs/rerankvreg/", metric_name = "answer_correctness")
    plot_metric_by_chunks_and_manual_edit("./outputs/rerankvreg/", metric_name = "faithfulness")
    plot_metric_by_chunks_and_manual_edit("./outputs/rerankvreg/", metric_name = "answer_relevancy")
    plot_metric_by_chunks_and_manual_edit("./outputs/rerankvreg/", metric_name = "context_recall")

    #plot_metric_full_dataset(metrics_folder = "./outputs/rerankvreg/", metric_name = "avg_total")
    #plot_metric_full_dataset("./outputs/rerankvreg/", metric_name = "answer_correctness")
    #plot_metric_full_dataset("./outputs/rerankvreg/", metric_name = "faithfulness")
    #plot_metric_full_dataset("./outputs/rerankvreg/", metric_name = "answer_relevancy")
    #plot_metric_full_dataset("./outputs/rerankvreg/", metric_name = "context_recall")