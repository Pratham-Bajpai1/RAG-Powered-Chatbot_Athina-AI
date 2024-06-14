import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from chatbot import find_relevant_response, generate_answer, queries, responses

# Example test set
test_queries = [
    "What is the policy renewal process?",
    "What are the exclusions of the policy?",
    "How can I cancel my policy?",
    "What is the coverage for theft under the policy?",
    "What are the steps to follow if my car is stolen?",
]

# these will be manually filled based on the actual dataset or known correct answers
expected_responses = [
    "You can see the details we have on your car insurance details. These changes may mean we need to increase or reduce the premium, or in some cases cancel your policy. Page 37How the policy works How the policy worksBefore renewal.",
    "Your cover to drive the hire car is restricted to the limits on use and exclusions shown on your certificate of motor insurance, and in the terms of your policy.",
    "This applies whether we cancel the policy or you cancel it. If we need to cancel the policy. We can cancel the policy at any time if we have a valid reason",
    "We’ll only pay what is reasonable and necessary for these expenses. How much am I covered for? We’ll provide cover up to the amount shown in ‘What your cover includes’ on page 8",
    "The arbitrator’s decision will be final, and whoever doesn’t win will have to pay all costs and expenses.We’re here to support you when accidents happen, so we’ve put together some useful steps for you to follow to help make the process smoothe",
]

# Evaluate the chatbot using the test set
def evaluate_chatbot(test_queries, expected_responses):
    predicted_responses = []
    for query in test_queries:
        # Check if the query exists in the dataset
        if query in queries:
            response_index = queries.index(query)
            response = responses[response_index]
        else:
            # If the query is not found, use the model-based approach
            relevant_passages = find_relevant_response(query, top_n=3)
            response = generate_answer(query, relevant_passages)
        predicted_responses.append(response)

    # Using TF-IDF and cosine similarity for evaluation
    vectorizer = TfidfVectorizer().fit(expected_responses + predicted_responses)
    expected_vectors = vectorizer.transform(expected_responses)
    predicted_vectors = vectorizer.transform(predicted_responses)

    cosine_similarities = cosine_similarity(expected_vectors, predicted_vectors)
    similarity_threshold = 0.7

    matches = (cosine_similarities.diagonal() > similarity_threshold).astype(int)
    accuracy = matches.mean()

    print(f"Accuracy: {accuracy}")
    print(f"Cosine Similarities: {cosine_similarities.diagonal()}")

    #we can print a detailed report
    print("Detailed Evaluation Report:")
    for i, (expected, predicted, similarity) in enumerate(zip(expected_responses, predicted_responses, cosine_similarities.diagonal())):
        print(f"\nQuery {i+1}: {test_queries[i]}")
        print(f"Expected Response: {expected}")
        print(f"Predicted Response: {predicted}")
        print(f"Cosine Similarity: {similarity}")

# Run the evaluation
evaluate_chatbot(test_queries, expected_responses)
