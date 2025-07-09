"""
RUN IN TERMINAL
pip install langchain langchain-community chromadb ibm-watsonx-ai gradio

"""

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams, EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA

import os

# ----- Config -----
MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"
EMBED_MODEL_ID = "ibm/slate-125m-english-rtrvr"
PROJECT_ID = "skills-network"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

# ----- LLM -----
def get_llm():
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    return WatsonxLLM(model_id=MODEL_ID, url=WATSONX_URL, project_id=PROJECT_ID, params=parameters)

# ----- Embedding Model -----
def get_embedding_model():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    return WatsonxEmbeddings(model_id=EMBED_MODEL_ID, url=WATSONX_URL, project_id=PROJECT_ID, params=embed_params)

# ----- Task 1: Load PDF -----
def task1_load_pdf():
    print("\n--- Task 1: Load PDF ---")
    loader = PyPDFLoader("paper.pdf")
    docs = loader.load()
    print(docs[0].page_content[:1000])

# ----- Task 2: Text Splitter (LaTeX) -----
def task2_text_split():
    print("\n--- Task 2: Text Splitter (LaTeX) ---")
    latex_text = """
    \\documentclass{article}
    \\begin{document}
    \\maketitle
    \\section{Introduction}
    Large language models (LLMs) are a type of machine learning model...
    \\subsection{History of LLMs}
    The earliest LLMs were developed in the 1980s and 1990s...
    \\subsection{Applications of LLMs}
    LLMs have many applications in the industry...
    \\end{document}
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    from langchain_core.documents import Document
    docs = [Document(page_content=latex_text)]
    splits = text_splitter.split_documents(docs)
    for i, s in enumerate(splits):
        print(f"Chunk {i+1}: {s.page_content[:150]}")

# ----- Task 3: Embedding -----
def task3_embed():
    print("\n--- Task 3: Embedding ---")
    embedding_model = get_embedding_model()
    embedding = embedding_model.embed_query("How are you?")
    print("First 5 embedding values:", embedding[:5])

# ----- Task 4: VectorDB and Similarity Search -----
def task4_vectordb():
    print("\n--- Task 4: ChromaDB + Search ---")
    loader = TextLoader("new-Policies.txt")
    docs = loader.load()
    embedding_model = get_embedding_model()
    db = Chroma.from_documents(docs, embedding_model)
    results = db.similarity_search("Smoking policy", k=5)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:200]}")

# ----- Task 5: Retriever -----
def task5_retriever():
    print("\n--- Task 5: Retriever for 'Email policy' ---")
    loader = TextLoader("new-Policies.txt")
    docs = loader.load()
    embedding_model = get_embedding_model()
    db = Chroma.from_documents(docs, embedding_model)
    retriever = db.as_retriever()
    results = retriever.get_relevant_documents("Email policy")
    for i, doc in enumerate(results[:2]):
        print(f"Doc {i+1}: {doc.page_content[:200]}")

# ----- Task 6: QA Bot with LangChain -----
def task6_qa():
    print("\n--- Task 6: QA Bot ---")
    llm = get_llm()
    loader = PyPDFLoader("paper.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedding_model = get_embedding_model()
    db = Chroma.from_documents(chunks, embedding_model)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    query = "What this paper is talking about?"
    answer = qa.invoke(query)
    print("Answer:", answer['result'])

# ----- Main -----
if __name__ == "__main__":
    task1_load_pdf()
    task2_text_split()
    task3_embed()
    task4_vectordb()
    task5_retriever()
    task6_qa()
