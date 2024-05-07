import streamlit as st
import searcher
import pandas as pd

st.set_page_config(
	page_title="Resume Indexer Beta",
	layout="wide"
)

def main():

	if "searcher" not in st.session_state:
		s = searcher.Searcher("beta-index")
		st.session_state.pinecone_object = s
	else:
		s = st.session_state.pinecone_object

	test_queries = ["i need a data analyst from seattle who knows python", 
					"visualization expert with frontend skills",
					"AI expert with data engineering experience",
					"Tech lead with business knowledge and communication skills",
					"data scientist with business expertise and visualization skills"]

	col1, col2 = st.columns(2)

	with col1:
	    selected_query = st.selectbox('Select a query', test_queries)
	    query = st.text_input('Or enter a custom query')

	with col2:
	    k = st.slider('Number of results (k)', 1, 10, 5)  # default value is 5

	search_button = st.button('Search')

	if search_button:
	    if query:
	        xc = s.execute_query(query, k=k)
	    elif selected_query:
	        xc = s.execute_query(selected_query, k=k)

	    results = []
	    for i, result in enumerate(xc['matches'], start=1):
	        result_data = {
	            "ID": result['id'],
	            "Experience": result['metadata']['experience'],
	            "Highest Degree": result['metadata']['highest degree'],
	            "Location": result['metadata']['location'],
	            "University": result['metadata']['university'],
	            "Score": result['score']
	        }
	        results.append(result_data)

	    df = pd.DataFrame(results)
	    st.dataframe(df)


if __name__ == "__main__":
	main()