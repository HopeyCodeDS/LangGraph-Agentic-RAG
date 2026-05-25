#!/bin/bash

cd "$(dirname "$0")"
echo "Welcome to the LangGraph RAG with grading"

source .venv/Scripts/activate

python -m src.main