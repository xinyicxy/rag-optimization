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


def plot_metric_boxplot_by_doc(csv_path, metric_col):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Check necessary columns
    if 'color' not in df.columns:
        raise ValueError("Column 'color' not found in the CSV.")
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in the CSV.")

    # Mapping from color code to filename label
    color_values = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    file_names = [
        "Infra 1", "Infra 2", "Infra 3", "Infra 4",
        "Natsec 1", "Natsec 2", "Natsec 3", "Natsec 4",
        "Natsec 5", "Natsec 6", "Natsec 7", "Natsec 8"
    ]
    label_map = dict(zip(color_values, file_names))

    # Map color to document labels
    df = df[df['color'].isin(label_map.keys())].copy()
    df['doc_label'] = df['color'].map(label_map)

    # Plot
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='doc_label', y=metric_col, palette="tab20")

    # Labels and title
    plt.xlabel("Document")
    plt.ylabel(metric_col.replace("_", " ").title())
    plt.title(f"Distribution of {metric_col.replace('_', ' ').title()} by Document")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sample_size_barplot(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Required columns
    if 'color' not in df.columns or 'sample_size' not in df.columns:
        raise ValueError("CSV must contain 'color' and 'sample_size' columns.")

    # Map color to labels
    color_values = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    file_names = [
        "Infra 1", "Infra 2", "Infra 3", "Infra 4",
        "Natsec 1", "Natsec 2", "Natsec 3", "Natsec 4",
        "Natsec 5", "Natsec 6", "Natsec 7", "Natsec 8"
    ]
    label_map = dict(zip(color_values, file_names))

    # Filter and map
    df = df[df['color'].isin(label_map.keys())].copy()
    df['doc_label'] = df['color'].map(label_map)

    # Take the first sample size for each document
    sample_sizes = df.groupby('doc_label', sort=False)['sample_size'].first().reset_index()

    # Plot
    plt.figure(figsize=(14, 6))
    sns.barplot(data=sample_sizes, x='doc_label', y='sample_size', palette="Blues_d")

    # Labels and title
    plt.xlabel("Document")
    plt.ylabel("Number of Questions (Sample Size)")
    plt.title("Sample Size per Document")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def boxplot_metric_by_chunk_type(csv_path, metric):
    df = pd.read_csv(csv_path)

    if 'chunk_type' not in df.columns or metric not in df.columns:
        raise ValueError("Ensure the dataframe contains 'chunk_type' and the specified metric.")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='chunk_type', y=metric, palette='Set2')
    plt.xlabel("Chunk Type")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} by Chunk Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def boxplot_metric_by_chunk_size_per_type(csv_path, metric):
    df = pd.read_csv(csv_path)

    if 'chunk_type' not in df.columns or 'chunk_size' not in df.columns or metric not in df.columns:
        raise ValueError("CSV must include 'chunk_type', 'chunk_size', and the specified metric.")

    unique_chunk_types = df['chunk_type'].unique()
    n = len(unique_chunk_types)

    # Setup subplot grid
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(6 * n, 6), sharey=True)

    if n == 1:
        axes = [axes]  # Ensure it's iterable

    for ax, chunk_type in zip(axes, unique_chunk_types):
        subset = df[df['chunk_type'] == chunk_type]
        sns.boxplot(data=subset, x='chunk_size', y=metric, ax=ax, palette='Set3')
        ax.set_title(f"{chunk_type.title()} Chunks")
        ax.set_xlabel("Chunk Size")
        ax.set_ylabel(metric.replace("_", " ").title())

    plt.tight_layout()
    # plt.title(f"{metric.replace('_', ' ').title()} by Chunk Size and Type")
    plt.show()


def boxplot_metric_by_k_per_chunk_type(csv_path, metric):
    df = pd.read_csv(csv_path)

    if 'chunk_type' not in df.columns or 'k' not in df.columns or metric not in df.columns:
        raise ValueError("CSV must include 'chunk_type', 'k', and the specified metric.")

    unique_chunk_types = df['chunk_type'].unique()
    n = len(unique_chunk_types)

    # Setup subplot grid
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(6 * n, 6), sharey=True)

    if n == 1:
        axes = [axes]  # Ensure it's iterable

    for ax, chunk_type in zip(axes, unique_chunk_types):
        subset = df[df['chunk_type'] == chunk_type]
        sns.boxplot(data=subset, x='k', y=metric, ax=ax, palette='Set2')
        ax.set_title(f"{chunk_type.title()} Chunks")
        ax.set_xlabel("k Value")
        ax.set_ylabel(metric.replace("_", " ").title())

    # Adjust layout to fit suptitle
    plt.subplots_adjust(top=0.85)
    fig.suptitle(f"{metric.replace('_', ' ').title()} by Chunk Type and k", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.82])  # avoid cutting off labels and suptitle
    plt.show()


def extract_insufficient_responses_to_csv(directory_path, output_csv_path):
    records = []

    # Match files starting with "exp_output" and ending with ".json"
    json_files = glob(os.path.join(directory_path, "exp_output*.json"))

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        hyperparams = data.get("hyperparameters", {})

        for result in data.get("results", []):
            llm_response = result.get("llm_response", "").lower()
            if "insufficient information" in llm_response:
                metadata = result.get("metadata", {})
                records.append({
                    "file_name": os.path.basename(file_path),
                    "k": hyperparams.get("k"),
                    "chunk_type": hyperparams.get("chunk_type"),
                    "chunk_size": hyperparams.get("chunk_size"),
                    "question_id": metadata.get("question_id"),
                    "RFP_id": metadata.get("RFP_id")
                })

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} insufficient responses to {output_csv_path}")


def plot_rfp_id_count(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Count the occurrences of each RFP_id
    rfp_counts = df['RFP_id'].value_counts().reset_index()
    rfp_counts.columns = ['RFP_id', 'Count']

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='RFP_id', y='Count', data=rfp_counts, palette='viridis')

    plt.title('Total Number of Responses per RFP ID')
    plt.xlabel('RFP ID')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_question_id_count(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Filter out question IDs >= 400
    df = df[df['question_id'] < 400]

    # Count the occurrences of each question_id
    question_counts = df['question_id'].value_counts().reset_index()
    question_counts.columns = ['question_id', 'Count']

    # Sort in descending order and select top 15
    top_questions = question_counts.sort_values(by='Count', ascending=False).head(15)

    # Create the bar plot in descending order
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='question_id',
        y='Count',
        data=top_questions,
        order=top_questions['question_id'],
        palette='plasma'
    )

    plt.title('Top 15 Questions (ID < 400) with Most "Insufficient Information" Responses')
    plt.xlabel('Question ID')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_chunk_type_boxplot(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Count the number of occurrences for each chunk_type
    chunk_counts = df.groupby('chunk_type').size().reset_index(name='Count')

    # Create a box plot based on the counts of rows per chunk_type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='chunk_type', y='Count', data=chunk_counts, palette='coolwarm')

    plt.title('Distribution of Row Counts per Chunk Type')
    plt.xlabel('Chunk Type')
    plt.ylabel('Row Count per Chunk Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def insufficient_answer_by_experiment(reg_csv_path, reranking_csv_path):
    # Load data
    df_reg = pd.read_csv(reg_csv_path)
    df_reranking = pd.read_csv(reranking_csv_path)

    # Add 'technique' column
    def label_technique(df):
        def get_technique(file_name):
            name = file_name.lower()
            if 'reranking_filtering' in name:
                return 'reranking_filtering'
            elif 'filtering' in name:
                return 'filtering'
            elif 'reranking' in name:
                return 'reranking'
            else:
                return 'regular'
        df['technique'] = df['file_name'].apply(get_technique)
        return df

    df_reg['technique'] = 'regular'
    df_reranking = label_technique(df_reranking)

    # Combine
    df_all = pd.concat([df_reg, df_reranking], ignore_index=True)

    # Create 'experiment' column
    df_all['experiment'] = df_all['chunk_type'].astype(str) + "_" + df_all['chunk_size'].astype(str)

    # Filter to insufficient responses
    df_insufficient = df_all
    #df_all[df_all['llm_response'].str.contains("insufficient information", case=False, na=False)]

    # Group by experiment and technique
    counts = df_insufficient.groupby(['experiment', 'technique']).size().reset_index(name='count')
    counts.loc[counts['technique'] == 'regular', 'count'] = counts.loc[counts['technique'] == 'regular', 'count'] / 4

    # Pivot for plotting
    plot_df = counts.pivot(index='experiment', columns='technique', values='count').fillna(0).astype(int).reset_index()

    # Melt for seaborn
    plot_df_melted = plot_df.melt(id_vars='experiment',
                                  value_vars=['regular', 'reranking', 'filtering', 'reranking_filtering'],
                                  var_name='Technique', value_name='Count')

    # Order experiments
    experiment_order = ['sentences_10', 'words_50', 'paragraphs_1']
    plot_df_melted['experiment'] = pd.Categorical(plot_df_melted['experiment'],
                                                  categories=experiment_order,
                                                  ordered=True)
    plot_df_melted = plot_df_melted.sort_values('experiment')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df_melted, x='experiment', y='Count', hue='Technique')
    plt.title('Insufficient Info Responses by Technique & Chunking Strategy')
    plt.xlabel('Chunking Strategy')
    plt.ylabel('Number of "Insufficient Info" Responses')
    plt.legend(title='Technique')
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
    SPECIAL_CSV = "special_combined_metrics.csv"

    # plot_metric_boxplot_by_doc(csv_path=CSV, metric_col="answer_correctness")

    # boxplot_metric_by_chunk_size_per_type(csv_path=CSV, metric="answer_correctness")
   
    # plot_sample_size_barplot(csv_path=CSV)
    # extract_insufficient_responses_to_csv("./outputs/reranking/", "insufficient_reranking_data.csv")

    # plot_rfp_id_count("insufficient_reranking_data.csv")
    # plot_question_id_count("insufficient_reg_data.csv")
    # plot_chunk_type_boxplot("insufficient_reg_data.csv")

    # insufficient_answer_by_experiment(reg_csv_path = "insufficient_reg_data.csv", reranking_csv_path="insufficient_reranking_data.csv")

    # boxplot_metric_by_k_per_chunk_type(csv_path=CSV, metric="answer_correctness")

    # plot_question_id_count("insufficient_reg_data.csv")

    METRICS_MAN_EDITED = ["manually_edited_true_" + item for item in METRICS_TO_PLOT]
    METRICS_MULTICHUNK = ["chunks_length_2_" + item for item in METRICS_TO_PLOT]

    #boxplot_metric_by_chunk_size_per_type(csv_path=SPECIAL_CSV, metric=METRICS_MAN_EDITED[0])
    boxplot_metric_by_k_per_chunk_type(csv_path=SPECIAL_CSV, metric=METRICS_MAN_EDITED[4])

    #boxplot_metric_by_chunk_size_per_type(csv_path=SPECIAL_CSV, metric=METRICS_MULTICHUNK[0])
    #boxplot_metric_by_k_per_chunk_type(csv_path=SPECIAL_CSV, metric=METRICS_MULTICHUNK[0])
