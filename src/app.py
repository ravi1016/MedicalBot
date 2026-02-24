from flask import Flask, render_template, jsonify, request
from helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
import os


app = Flask(__name__, template_folder="../templates", static_folder="../static")


load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HF_TOKEN = os.getenv("HF_TOKEN")


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using the context below:

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)



@app.route("/")
def index():
    return render_template("chat.html")



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke(msg)
    print("Response : ", response.content)
    return str(response.content)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)