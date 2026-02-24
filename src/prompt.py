prompt = ChatPromptTemplate.from_template("""
Answer the question using the context below:

Context:
{context}

Question:
{question}
""")