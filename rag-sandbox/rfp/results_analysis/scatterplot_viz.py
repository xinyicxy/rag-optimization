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


def get_plot_arrays(metrics_dir, document_dir, metric_names):
    """
    metrics_dir (str): path to metrics outputs
    document_dir (str): path to documents
    metric_names (list[str]): list of metrics to extract

    returns:
        x (list): x-values (k/n)
        y_dict (dict): mapping metric_name -> list of y-values
        color (list): color index per document
    """

    x = []
    color = []
    k_list = []
    chunk_type_list = []
    chunk_size_list = []
    y_dict = {metric: [] for metric in metric_names}

    for filename in os.listdir(metrics_dir):
        if filename.startswith("metrics_") and filename.endswith(".json"):

            # get hyperparams from exp name
            match = re.search(r'k(\d+)_type(\w+)_size(\d+)', filename)
            if not match:
                print(f"Skipping file {filename}: Pattern not found")
                continue

            k, chunk_type, chunk_size = match.groups()
            k = int(k)
            chunk_size = int(chunk_size)

            # load in metrics
            filepath = os.path.join(metrics_dir, filename)
            try:
                with open(filepath, "r") as f:
                    metrics_file = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {filename}")
                continue

            # loop through RFP documents
            for i, rfp_path in enumerate(os.listdir(document_dir)):
                if rfp_path.endswith(".pdf"):
                    full_rfp_path = os.path.join(document_dir, rfp_path)

                    try:
                        chunks_ = load_pdf(pdf_path=full_rfp_path,
                                           chunk_type=chunk_type,
                                           chunk_size=chunk_size)
                    except Exception as e:
                        print(f"Error loading PDF {rfp_path}: {e}")
                        continue

                    n = len(chunks_)
                    rfp_id = re.sub(r'[-]|\.pdf$', '_', rfp_path)[:-1]

                    if rfp_id not in metrics_file["RFP_id"]:
                        print(f"Skipping {rfp_id} — not found in metrics file.")
                        continue

                    metric_entry = metrics_file["RFP_id"][rfp_id]

                    # append common x and color
                    x_val = (1, round(k/n, 3))
                    x.append(min(x_val))
                    color.append(i)
                    k_list.append(k)
                    chunk_type_list.append(chunk_type)
                    chunk_size_list.append(chunk_size)

                    # append each metric
                    for metric in metric_names:
                        metric_val = metric_entry.get(metric, None)
                        if metric_val is None:
                            print(f"{metric} missing for {rfp_id} in {filename}")
                        y_dict[metric].append(metric_val)

    return x, y_dict, color, k_list, chunk_type_list, chunk_size_list


def _get_plot_arrays_single(metrics_dir, document_dir, metric_name):
    """
    metrics_dir (str): path to metrics outputs
    document_dir(str): path to documents

    metrics_name(str): one of {"answer_relevancy",
      "faithfulness",
      "context_recall",
      "context_precision",
      "answer_correctness",
      "EM",
      "F1",
      "avg_retrieve_context", ** latency metric
      "avg_llm_response", ** latency metric
      "avg_total", ** latency metric
      "sample_size"}

    returns:
    x (list): x-values (k/n)
    y_dict (dict): mapping metric_name -> list of y-values
    color (list): color index per document
    """

    x = []
    y = []
    color = []

    for filename in os.listdir(metrics_dir):
        if filename.startswith("metrics_") and filename.endswith(".json"):

            # get hyperparams from exp name
            match = re.search(r'k(\d+)_type(\w+)_size(\d+)', filename)
            if not match:
                print(f"Skipping file {filename}: Pattern not found")
                continue

            k, chunk_type, chunk_size = match.groups()
            k = int(k)
            chunk_size = int(chunk_size)

            # load in da metrics
            filepath = os.path.join(metrics_dir, filename)
            with open(filepath, "r") as f:
                try:
                    metrics_file = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {filename}")
                    continue

            # looping through the rfp boi
            for i, rfp_path in enumerate(os.listdir(document_dir)):
                if rfp_path.endswith(".pdf"):
                    full_rfp_path = os.path.join(document_dir, rfp_path)
                    # print(full_rfp_path)
                    chunks_ = load_pdf(pdf_path=full_rfp_path,
                                       chunk_type=chunk_type,
                                       chunk_size=chunk_size)
                    n = len(chunks_)

                    # get the metric value
                    rfp_id = re.sub(r'[-]|\.pdf$', '_', rfp_path)[:-1]
                    # print(rfp_id)
                    metric_val = metrics_file["RFP_id"][rfp_id][metric_name]

                    # append x,y, color
                    x_val = (1, round(k/n, 3))
                    x.append(min(x_val))
                    y.append(metric_val)
                    color.append(i)

    return x, y, color


def save_plot_arrays_to_csv(metrics_dir, document_dir, metric_names,
                            output_path):
    """
    calculate plot values and save to one csv
    """
    x, y_dict, color, k_list, chunk_type_list, chunk_size_list = get_plot_arrays(metrics_dir, document_dir, metric_names)

    # create dataframe
    data = {
        "x": x,
        "color": color,
        "k": k_list,
        "chunk_type": chunk_type_list,
        "chunk_size": chunk_size_list
    }

    for metric in metric_names:
        data[metric] = y_dict[metric]

    df = pd.DataFrame(data)

    # save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def _plot_scatter(x, y, color):

    doc_ids = np.array(color)
    num_docs = 12
    unique_docs = np.unique(doc_ids)
    cmap = cm.get_cmap('tab20', num_docs)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(num_docs+1)-0.5,
                                ncolors=num_docs)

    file_names = ["Infra 1", "Infra 2", "Infra 3", "Infra 4",
                  "Natsec 1", "Natsec 2", "Natsec 3", "Natsec 4",
                  "Natsec 5", "Natsec 6", "Natsec 7", "Natsec 8",
                  "Natsec 9", "Natsec 10", "Natsec 11", "Natsec 12"]

    doc_id_to_label = dict(zip(unique_docs, file_names))

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(unique_docs)}
    color_indices = [doc_id_to_idx[doc] for doc in doc_ids]

    scatter = plt.scatter(x, y, c=color_indices, cmap=cmap, norm=norm)


    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=doc_id_to_label[doc],
                          markerfacecolor=cmap(idx), markersize=8)
            for doc, idx in doc_id_to_idx.items()]
    plt.legend(handles=handles, title="Document ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('PROPORTION OF DOCUMENT RETURNED IN RETRIEVAL K/N')
    plt.ylabel('METRIC NAME')
    plt.title('MEANINGFUL TITLE HERE')
    plt.tight_layout()
    plt.show()


def plot_scatter_by_rfp(csv_path):
    df = pd.read_csv(csv_path)

    x = df["x"].values
    doc_ids = df["color"].values
    num_docs = 12
    unique_docs = np.unique(doc_ids)
    metric_names = [col for col in df.columns if col not in ["x", "color", "k", "chunk_type", "chunk_size"]]

    num_docs = 12

    # custom color palette
    warm_colors = plt.cm.Oranges(np.linspace(0.5, 1, 4))  # Infra (0–3)
    cool_colors = plt.cm.PuBuGn(np.linspace(0.4, 0.9, num_docs - 4))  # Natsec (4–11)
    full_colors = np.vstack([warm_colors, cool_colors])
    cmap = mcolors.ListedColormap(full_colors)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(num_docs+1)-0.5, ncolors=num_docs)

    # document labels
    file_names = [
        "Infra 1", "Infra 2", "Infra 3", "Infra 4",
        "Natsec 1", "Natsec 2", "Natsec 3", "Natsec 4",
        "Natsec 5", "Natsec 6", "Natsec 7", "Natsec 8"
    ]

    # map doc_id -> file_name
    doc_id_to_label = dict(zip(unique_docs, file_names))

    # ,ap each document ID to its index in the color scale
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(unique_docs)}
    color_indices = [doc_id_to_idx[doc] for doc in doc_ids]

    for metric in metric_names:
        y = df[metric].values
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(x, y, c=color_indices, cmap=cmap, norm=norm, s=60, alpha=0.8)

        # legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                        label=doc_id_to_label[doc],
                        markerfacecolor=cmap(idx), markersize=8)
        for doc, idx in doc_id_to_idx.items()]

        ax.legend(handles=handles, title="Document", bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_xlabel('Proportion of Document Retrieved as Context (k/n)', fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs Context Size', fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_scatter_by_chunk_type(csv_path):
    # FIX this later
    df = pd.read_csv(csv_path)

    chunk_types = df["chunk_type"].unique()
    colors = plt.cm.get_cmap("tab10", len(chunk_types))

    metric_names = [col for col in df.columns if col not in ["x", "color", "k", "chunk_type", "chunk_size"]]

    for metric in metric_names:
        plt.figure(figsize=(10, 6))

        for i, chunk in enumerate(chunk_types):
            subset = df[df['chunk_type'] == chunk]
            plt.scatter(subset['x'], subset[metric], label=chunk, color=colors(i))

        plt.xlabel('Proportion of Document Retrieved as Context (k/n)')
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f'{metric.replace("_", " ").title()} vs Context Size')
        plt.legend(title='Chunk Type')
        plt.tight_layout()
        plt.show()


def plot_with_density(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Determine metrics to plot
    metric_names = [col for col in df.columns if col not in ["x", "color", "k", "chunk_type", "chunk_size"]]

    for metric in metric_names:
        plt.figure(figsize=(10, 6))

        # Scatter plot + regression line with confidence interval
        sns.regplot(data=df, x='x', y=metric, scatter=True, ci=95, scatter_kws={"s": 30, "alpha": 0.5}, line_kws={"color": "red"})

        plt.xlabel('Proportion of Document Retrieved as Context (k/n)')
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f'{metric.replace("_", " ").title()} vs Context Size with Density Estimation')
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


    """
    save_plot_arrays_to_csv(metrics_dir=METRICS_DIR,
                            document_dir=DOCUMENTS_DIR,
                            metric_names=METRICS_TO_PLOT,
                            output_path="metric_plot_data.csv")
    """

    #plot_scatter_by_rfp("metric_plot_data.csv")
    #plot_scatter_by_chunk_type("metric_plot_data.csv")
    #plot_with_density("metric_plot_data.csv")

    x, y_dict, color, k_list, chunk_type_list, chunk_size_list = get_plot_arrays(metrics_dir=METRICS_DIR, document_dir=DOCUMENTS_DIR, metric_names=METRICS_TO_PLOT)
