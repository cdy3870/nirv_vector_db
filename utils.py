import numpy as np
import random 
import os
from sentence_transformers import SentenceTransformer
import torch

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = st.secrets["api_key"]

params = {"sim_mets": ["l2", "cosine", "dot_product"],
     "search_algo": ["exhaustive", "beam", "hierarchical"],
     "index_algo": ["flat", "hierarchical_navigable_small_world_graph", "annoy", "faiss", "ball_tree", "kd_tree"]}

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

def is_relevant(query, data_point):
    k = random.randint(0, 1)
    return k

def ndcg_score(relevance, K):
  """Calculates the Normalized Discounted Cumulative Gain (NDCG) at rank K.

  Args:
      relevance: A list of relevance scores.
      K: The rank position to evaluate.

  Returns:
      The NDCG@K score.
  """  
  relevance = [len(relevance) - i - 1 for i in relevance]

  # print(relevance)
    
  # Calculate Discounted Cumulative Gain (DCG)
  DCG = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance[:K])])

  # Calculate Ideal Discounted Cumulative Gain (IDCG) by sorting ideal ranking
  ideal_relevance = sorted(relevance, reverse=True)
  IDCG = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:K])])

  # Avoid division by zero
  if IDCG == 0:
     return 0

  return DCG / IDCG 

def precision_at_k(actual_items, predicted_items, k):
    """Calculates precision at k.

    Args:
        actual_items: List of relevant items.
        predicted_items: List of predicted items.
        k: Cutoff position for the metric.

    Returns:
        Precision at k
    """

    predicted_top_k = predicted_items[:k]
    num_relevant_in_top_k = sum(item in actual_items for item in predicted_top_k)
    return num_relevant_in_top_k / k

def average_precision(actual_items, predicted_items):
    """Calculates the average precision (AP).

    Args:
        actual_items: List of relevant items.
        predicted_items: List of predicted items.

    Returns:
        Average precision
    """

    ap = 0.0
    num_relevant_found = 0
    for i, item in enumerate(predicted_items):
        if item in actual_items:
            num_relevant_found += 1
            # Note: i+1 as we start index from 0
            ap += precision_at_k(actual_items, predicted_items, i + 1)

    if num_relevant_found != 0:
        return ap / num_relevant_found
    else:
        return 0.0

def mean_average_precision(actual, predicted, k):
    """Calculates the mean average precision (MAP) at k.

    Args:
        actual: List of lists, where each inner list contains relevant items for a query.
        predicted: List of all predicted items (common across all queries).
        k: Cutoff position for the metric.

    Returns:
        Mean average precision
    """

    average_precisions = []
    for query_actual_items in actual:
        ap = average_precision(query_actual_items, predicted)
        average_precisions.append(ap)
    return sum(average_precisions) / len(average_precisions)

