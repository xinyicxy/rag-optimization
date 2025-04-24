#!/bin/bash

# fixed hyperparameter
python metrics_reranking.py words 50 15 reranking
python metrics_reranking.py sentences 10 15 reranking
python metrics_reranking.py paragraphs 1 15 reranking
