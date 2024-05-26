from flask import Flask, render_template, request # type: ignore
from src.helper import download_hugging_face_embeddings
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Ollama
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone.vectorstores import PineconeVectorStore

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "medical-chatbot"
embeddings = download_hugging_face_embeddings()

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

llm = Ollama(model="llama3", temperature=0.8, num_predict=512)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings, text_key="text"
)

retriever = docsearch.as_retriever(search_kwargs={'k': 5})


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)


@app.route("/")
def index() -> str:
    """
    Render the chat.html template when the root URL is accessed.

    Returns:
        str: HTML content of the chat.html file.
    """
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    """
    A function that handles the chat functionality. 
    It retrieves a message from the request form, queries a response, 
    and returns the result as a string.
    """
    msg = request.form["msg"]
    input_message = msg
    print(f"Input Message: {input_message}")
    result = qa.invoke({"query": input_message})
    print(f"Response: {result['result']}")
    return str(result["result"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
