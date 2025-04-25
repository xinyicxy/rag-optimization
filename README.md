# 🔍 RAG Optimization Project

This repository contains our work on optimizing Retrieval-Augmented Generation (RAG) pipelines for our corporate capstone project, including Synthetic QA pair generation and end-to-end RAG experimentation pipelines (including query reformulation and query expansion).

## 📁 Repository Structure
### qa-gen/
Scripts and outputs related to synthetic QA pair generation using an LLM.

Extracts QA pairs from documents

Combines them into a final dataset for use in RAG models

### multihop-data/
Contains data and preprocessing scripts for the MoreHop benchmark dataset.
(https://github.com/Alab-NII/morehopqa)

Scripts to format it appropriately for downstream RAG tasks

### rag-sandbox/
Houses end-to-end RAG pipelines for two methods:

morehop/: RAG pipeline for evaluating multi-hop reasoning

rfp/: RAG pipeline for public Request for Proposal (RFP) data

## 📌 Goals
1. Preprocess existing benchmark data and synthetically generate new data to evaluate multihop reasoning and RAG performance on RFP data.
2. Optimize indexing methods (chunk size, chunk type, search type, top k at retrieval)
3. Optimize SOTA retrieval strategies (query reformulation, re-ranking)

