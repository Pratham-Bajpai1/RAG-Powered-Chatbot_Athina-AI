import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load the extracted text from the file
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    pdf_text = f.read()

# Split the text into sentences
sentences = pdf_text.split('. ')

# Create a list of potential queries
queries = [
    "What is covered under the motor insurance policy?",
    "How can I make a claim?",
    "What is the policy renewal process?",
    "What documents are required to make a claim?",
    "What are the exclusions of the policy?",
    "What is the contact information for customer support?",
    "What is the process for adding a driver to the policy?",
    "How can I cancel my policy?",
    "What is the process for changing my vehicle details?",
    "Are there any discounts available for safe drivers?",
    "What is the coverage for theft under the policy?",
    "What is the procedure for reporting an accident?",
    "What are the steps to follow if my car is stolen?",
    "What is the process for transferring the policy to a new owner?",
    "How do I update my address on the policy?",
    "What is the policy period and how is it determined?",
    "Are there any additional covers available?",
    "What is the process for lodging a complaint?",
    "What is the waiting period for certain covers?",
    "What happens if I miss a premium payment?",
    "What is the policy’s territorial coverage?",
    "What is the process for renewing the policy online?",
    "Can I make changes to my policy online?",
    "What is the process for getting a duplicate policy document?",
    "How is the premium amount calculated?",
    "What is the no-claim bonus?",
    "What should I do if I lose my policy document?",
    "What is the coverage for natural disasters?",
    "How do I add a new vehicle to my existing policy?",
    "What is the grace period for policy renewal?"
]

# Use TfidfVectorizer to convert the sentences to TF-IDF vectors
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)

# Function to find the most relevant sentence for a given query
def find_relevant_sentence(query):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, sentence_vectors).flatten()
    most_similar_sentence_index = cosine_similarities.argmax()
    return sentences[most_similar_sentence_index]

# Create query-response pairs
query_response_pairs = []
for query in queries:
    response = find_relevant_sentence(query)
    query_response_pairs.append((query, response))

# Create a DataFrame and save to CSV
df = pd.DataFrame(query_response_pairs, columns=["query", "response"])
df.to_csv("query_response_dataset.csv", index=False)
# Display the dataset
print(df.head())
print("Dataset creation completed.")