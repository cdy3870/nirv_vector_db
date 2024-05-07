# Manages vector search queries and results in Pinecone

from pinecone import Pinecone
import utils
from FlagEmbedding import FlagReranker
import json
import random
from pprint import pprint
import indexer

# Setting use_fp16 to True speeds up computation with a slight performance degradation
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

class Searcher():
	def __init__(self, index_name):
		self.index = indexer.connect_to_db(index_name)

	def execute_query(self, q, k=10):
		xq = utils.embedding_model.encode(q).tolist()
		xc = self.index.query(vector=xq, top_k=k, include_metadata=True)
		return xc

	def get_query_results(self, k, test_queries):
		"""
		Queries the index with the test queries and prints the results.

		Args:
			index: The Pinecone index.
			k: The number of results to return.
			test_queries: The test queries.
			query_results: A dictionary to store the query results.

		Returns:
			The query results.
		"""

		query_results = {q:[] for q in test_queries}

		for q in test_queries:
			# print(f"Query: {q}")
			xc = self.execute_query(q, k=k)

			for result in xc['matches']:
				query_results[q].append(result['id'])
			#     print(f"{round(result['score'], 2)}: {result['id']}")
			# print("\n")

		return query_results


	def get_map(self, K, parsed_data, test_queries, query_results):
		"""
		Calculates the Mean Average Precision (MAP) at K.

		Args:
			K: The number of results to consider.
			test_queries: The test queries.
			query_results: The query results.

		Returns:
			The MAP at K.
		"""
		# Dummy representation of MAP, relevance determined randomly
		actual = []
		retrieved_results = list(query_results.values())
		test_samples = [t[0] for t in parsed_data]

		for query, results in zip(test_queries, retrieved_results):
			relevant_items = []
			for data_point in results:
				if utils.is_relevant(query, data_point):
					relevant_items.append(test_samples.index(data_point))  # Assume data points are unique
			actual.append(relevant_items)

		# Calculate MAP@10
		predicted = [i for i in range(len(test_samples))]
		map_10 = utils.mean_average_precision(actual, predicted, K)
		# print("Mean Average Precision @ 10:", map_10)

		return map_10


	def get_ndcg(self, parsed_data, query_results):
		"""
		Calculates the Normalized Discounted Cumulative Gain (NDCG) scores.

		Args:
			query_results: The query results.

		Returns:
			A dictionary containing the ranked results and NDCG scores, and the average NDCG score.
		"""
		# Dummy representation of NDCG, relevance ranking determined randomly
		results = {}
		ndcg_scores = []

		temp_data = {d[0]:d for d in parsed_data}

		for query, items in query_results.items():
			temp_dict = {}
			for item in items:
				temp_dict[item] = reranker.compute_score([query, item])
			sorted_results = dict(sorted(temp_dict.items(), key=lambda x: x[1], reverse=True))

			ground_truth_relevance = list(range(0, len(items)))
			random.shuffle(ground_truth_relevance)

			ndcg_value = utils.ndcg_score(ground_truth_relevance, K=len(items))
			ndcg_scores.append(ndcg_value)
			
			ranked_results = [{"id":k, "rr score": v, "experience": temp_data[k][2]["experience"]}
							  for i, (k, v) in enumerate(sorted_results.items())]
			
			results[id(query)] = {
				"Query": query,
				"Ranked Results": ranked_results,
				"NDCG Score": ndcg_value
			}

		average_ndcg = sum(ndcg_scores) / len(ndcg_scores)

		return results, average_ndcg


def main():
	"""
	The main function that queries the index, calculates MAP and NDCG scores, and saves the results to a JSON file.
	"""

	test_queries = ["i need a data analyst from seattle who knows python", 
					"visualization expert with frontend skills",
					"AI expert with data engineering experience",
					"Tech lead with business knowledge and communication skills",
					"data scientist with business expertise and visualization skills"]

	with open("dump.json", "rb") as f:
		data = f.read()

	parsed_data = indexer.parse_data(data)

	searcher = Searcher('beta-index')

	k = 10
	query_results = searcher.get_query_results(k, test_queries)

	K = 10
	map_val = searcher.get_map(K, parsed_data, test_queries, query_results)

	results, average_ndcg = searcher.get_ndcg(parsed_data, query_results)

	overall_results = {
		"MAP@10": map_val,
		"Average NDCG": average_ndcg
	}

	with open('reranked_results.json', 'w') as outfile:
		json.dump({'results': results, 'overall_results': overall_results}, outfile, indent=4)


if __name__ == "__main__":
	main()