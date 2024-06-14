import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import torch
import streamlit as st

# Load the dataset
df = pd.read_csv("query_response_dataset.csv")

# Combine responses into a single text for TF-IDF vectorization
responses = df['response'].tolist()
queries = df['query'].tolist()

# Use TfidfVectorizer to convert the sentences to TF-IDF vectors
vectorizer = TfidfVectorizer().fit(responses)
response_vectors = vectorizer.transform(responses)

# Function to find the most relevant response for a given query
def find_relevant_response(query, top_n=3):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, response_vectors).flatten()
    most_similar_response_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [responses[i] for i in most_similar_response_indices]

# Load the pre-trained BERT model for question answering
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

def generate_answer(question, passages):
    combined_passage = " ".join(passages)
    inputs = tokenizer(question, combined_passage, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
    return answer

# Create the Streamlit UI
st.title("RAG-powered Chatbot for Athina AI")

query = st.text_input("Enter your question:")
if query:
    # Check if the query exists in the dataset
    if query in queries:
        response_index = queries.index(query)
        response = responses[response_index]
        st.write("**Answer:**")
        st.write(response)
    else:
        # If the query is not found, use the model-based approach
        relevant_passages = find_relevant_response(query, top_n=3)
        if relevant_passages:
            answer = generate_answer(query, relevant_passages)
            st.header("**Answer:**")
            st.write(answer)
        else:
            st.write("Sorry, I couldn't find an answer to your question.")

st.markdown("### Sample Questions:")
sample_queries = [
    "What is the policy renewal process?",
    "What are the exclusions of the policy?",
    "How can I cancel my policy?",
    "What is the coverage for theft under the policy?",
    "What is the procedure for reporting an accident?",
    "What are the steps to follow if my car is stolen?",
    "What is the process for transferring the policy to a new owner?",
    "How do I update my address on the policy?",
    "What is the policy period and how is it determined?",
    "What happens if I miss a premium payment?",
    "What is the policy’s territorial coverage?",
    "What is the process for renewing the policy online?",
    "Can I make changes to my policy online?",
    "What is the process for getting a duplicate policy document?",
    "How is the premium amount calculated?"

]
for sample_query in sample_queries:
    st.write(f"- {sample_query}")