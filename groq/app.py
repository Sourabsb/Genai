import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings # Corrected import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

#load the Groq API key from the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# This block runs only once when the app starts
if "vector" not in st.session_state:
    # 1. Initialize embeddings, specifying the model to use
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/observability")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    
    # 2. Create the vector store using the embeddings
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq Demo")
# 3. Use the standard lowercase model name
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the question based on the context provided below. Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 4. Use a different variable for the text input to avoid overwriting the prompt template
prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt_input})
    print("Response time: ", time.process_time() - start)
    st.write(response["answer"])

    #with a streamlit expander
    with st.expander("Documents Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            # 5. Display the actual content, not the string "doc.page_content"
            st.write(doc.page_content)
            st.write("----------------------------------")