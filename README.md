# RAG-powered Chatbot

## Overview
This project involves creating a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on a provided PDF document. The chatbot uses a combination of a pre-trained language model and a custom query-response dataset to generate accurate and relevant answers.

## Files
- `extract_text.py`: Extracts text from the provided PDF document.
- `create_dataset.py`: Generates a dataset of query-response pairs from the extracted text.
- `chatbot.py`: Contains the implementation of the RAG-powered chatbot using the custom dataset and a fallback mechanism using a pre-trained language model.
- `evaluate.py`: Evaluates the chatbot's performance using standard evaluation metrics.
- `query_response_dataset.csv`: The dataset used for training and evaluating the chatbot.