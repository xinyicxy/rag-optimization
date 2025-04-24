import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


def plot_smoothed_metric(df, metric_name, num_bins=100):
    """
    Plots a smoothed line with confidence band and raw data points for a given metric
    vs. proportion of document retrieved (x).

    Parameters:
        df (pd.DataFrame): DataFrame with at least 'x' and the specified metric column
        metric_name (str): Name of the metric column to plot (e.g., 'faithfulness')
        num_bins (int): Number of bins to smooth over (default = 50)
    """

    # Clip x to [0, 1]
    df = df.copy()

    # Drop rows with missing metric values
    df = df.dropna(subset=["x", metric_name])

    # Bin x into evenly spaced intervals
    df["x_bin"] = pd.cut(df["x"], bins=np.linspace(0, 1.1, num_bins + 1), include_lowest=True)
    df["x_bin"] = pd.qcut(df["x"], q=num_bins, duplicates='drop')

    # Aggregate stats by bin
    binned = df.groupby("x_bin", observed=True).agg(
        x_mean=("x", "mean"),
        metric_mean=(metric_name, "mean"),
        metric_std=(metric_name, "std"),
        count=(metric_name, "count")
    ).dropna()

    # Compute 95% confidence intervals
    binned["ci95"] = 1.96 * (binned["metric_std"] / np.sqrt(binned["count"]))

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot individual raw points
    plt.scatter(df["x"], df[metric_name], alpha=0.5, s=10, color="gray", label="Raw Data")

    # Smoothed line
    plt.plot(binned["x_mean"], binned["metric_mean"], linewidth=2, color="navy", label="Mean")

    # Confidence interval band
    plt.fill_between(
        binned["x_mean"],
        binned["metric_mean"] - binned["ci95"],
        binned["metric_mean"] + binned["ci95"],
        color="skyblue",
        alpha=0.4,
        label="95% CI"
    )

    # Formatting
    plt.ylim(0, 1)
    plt.title(f"{metric_name.replace('_', ' ').title()} vs. Proportion of Document Retrieved")
    plt.xlabel("Proportion of Document Retrieved")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_relationship(df, x_metric, y_metric, alpha=0.5, figsize=(8, 6)):
    df = df.copy()
    df = df.dropna(subset=[x_metric, y_metric])

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_metric, y=y_metric, alpha=alpha, s=20, edgecolor=None)

    # Optional trend line (comment out if you don’t want it)
    sns.regplot(data=df, x=x_metric, y=y_metric, scatter=False, color='darkblue', ci=None, line_kws={"linewidth": 2})

    plt.title(f"{y_metric.replace('_', ' ').title()} vs. {x_metric.replace('_', ' ').title()}")
    plt.xlabel(x_metric.replace('_', ' ').title())
    plt.ylabel(y_metric.replace('_', ' ').title())
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def extract_responses_to_csv(directory, output_csv="compiled_llm_responses.csv"):
    records = []

    for filename in os.listdir(directory):
        if filename.startswith("exp_output") and filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                results = data.get("results", [])
                for r in results:
                    record = {
                        "file_name": filename,
                        "question_id": r["metadata"]["question_id"],
                        "llm_response": r["llm_response"],
                        "ground_truth_answer": r["ground_truth_answer"]
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV with {len(df)} records to {output_csv}")
    return df


def plot_response_vs_groundtruth_wordcounts(csv_file):
    df = pd.read_csv(csv_file)

    # Filter out "insufficient information" LLM responses (case-insensitive)
    df_filtered = df[~df["llm_response"].str.contains("insufficient information", case=False, na=False)]

    # Word counts
    df_filtered["llm_word_count"] = df_filtered["llm_response"].str.split().str.len()
    df_filtered["gt_word_count"] = df_filtered["ground_truth_answer"].str.split().str.len()

    # Melt into long format for boxplot
    melted = df_filtered.melt(
        value_vars=["llm_word_count", "gt_word_count"],
        var_name="type",
        value_name="word_count"
    )

    # Cleaner labels
    melted["type"] = melted["type"].replace({
        "llm_word_count": "LLM Response",
        "gt_word_count": "Ground Truth"
    })

    # Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=melted, x="type", y="word_count", palette="pastel")
    plt.title("Word Count Comparison (LLM vs. Ground Truth)")
    plt.ylabel("Word Count")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


def cluster_llm_vs_ground_truth(csv_file, model_name="all-MiniLM-L6-v2", n_clusters=2):
    df = pd.read_csv(csv_file)
    df_filtered = df[~df["llm_response"].str.contains("insufficient information", case=False, na=False)].dropna()

    # Load sentence embedding model
    model = SentenceTransformer(model_name)

    # Compute embeddings
    llm_embeddings = model.encode(df_filtered["llm_response"].tolist(), show_progress_bar=True)
    gt_embeddings = model.encode(df_filtered["ground_truth_answer"].tolist(), show_progress_bar=True)

    # Compute cosine similarity between each LLM and its GT
    similarities = [cosine_similarity([llm], [gt])[0][0] for llm, gt in zip(llm_embeddings, gt_embeddings)]
    df_filtered["similarity"] = similarities

    # Optional: Visualize distribution of similarity scores
    plt.figure(figsize=(8, 5))
    sns.histplot(df_filtered["similarity"], bins=30, kde=True, color="skyblue")
    plt.title("Cosine Similarity Between LLM Responses and Ground Truth Answers")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Optional: Cluster the combined embeddings
    combined = [*llm_embeddings, *gt_embeddings]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)

    labels = ["LLM"] * len(llm_embeddings) + ["GT"] * len(gt_embeddings)
    df_plot = pd.DataFrame(reduced, columns=["x", "y"])
    df_plot["type"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x="x", y="y", hue="type", palette=["salmon", "dodgerblue"], alpha=0.6)
    plt.title("2D PCA Projection of LLM vs Ground Truth Answer Embeddings")
    plt.legend(title="Type")
    plt.tight_layout()
    plt.show()

    return df_filtered[["llm_response", "ground_truth_answer", "similarity"]]


if __name__ == "__main__":

    METRICS = [
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
     
    df = pd.read_csv("metric_plot_data.csv")
    
    # extract_responses_to_csv(directory = "./outputs", output_csv="compiled_llm_responses.csv")
    # plot_response_vs_groundtruth_wordcounts("compiled_llm_responses.csv")
    cluster_llm_vs_ground_truth("compiled_llm_responses.csv")
