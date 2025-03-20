rounting_prompt = """You are an expert at routing a query to the appropriate data source.
Consider the following query: {}\n
Answer only with Yes or No. Dont add comments or titles. 
Does the query relevant to the legal world? 
Answer: """

RAG_prompt = """Here is the relevant information extracted from the legal documents: \n {} \n\n
Based on this relevant information and your knowledge, answer the following query.
Query: {}
Answer: """

QA_prompt = """Query: {}
Answer: """