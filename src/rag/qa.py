import os
import dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


# Carrega variáveis de ambiente
dotenv.load_dotenv()

def load_vectorstore(persist_directory="vectorstore"):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb

def build_qa_chain():
    vectordb = load_vectorstore()
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def answer_question(question: str) -> str:
    qa = build_qa_chain()
    result = qa({"query": question})
    return result["result"]
