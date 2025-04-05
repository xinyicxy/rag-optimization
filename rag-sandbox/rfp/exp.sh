#!/bin/bash

# fixed hyperparameter
CHUNK_TYPE="words"

python metrics.py "$CHUNK_TYPE" 500 5
python metrics.py "$CHUNK_TYPE" 500 10
python metrics.py "$CHUNK_TYPE" 50 15
python metrics.py "$CHUNK_TYPE" 300 15
python metrics.py "$CHUNK_TYPE" 500 15
