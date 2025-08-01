import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

load_dotenv()

## load the Groq Api Key
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Objectbox VectorstoreDB With Llama3 Demo")

# Initialize LLM and prompt once
if 'llm' not in st.session_state:
    st.session_state.llm = ChatGroq(groq_api_key=groq_api_key,
                                    model_name="llama3-8b-8192") # Corrected model name

if 'prompt' not in st.session_state:
    st.session_state.prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Questions:{input}
        """
    )

def vector_embedding():
    """
    Creates and stores vector embeddings in ObjectBox if they don't already exist.
    """
    if "vectors" not in st.session_state:
        with st.spinner("Creating embeddings... This may take a moment."):
            st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[0:20])
            
            # Create and persist the ObjectBox vector store
            st.session_state.vectors = ObjectBox.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings,
                embedding_dimensions=768,
                db_directory="objectbox_db" # Use a dedicated directory
            )
        st.success("ObjectBox Database is ready")

# --- Streamlit UI ---

# Button to create embeddings
if st.button("Create Document Embeddings"):
    vector_embedding()

input_prompt = st.text_input("Enter Your Question From Documents")

# Only proceed if the prompt is entered AND the vector store is ready
if input_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': input_prompt})
    end_time = time.process_time()

    print(f"Response time: {end_time - start_time}")
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
elif input_prompt:
    st.warning("Please create the document embeddings first by clicking the button above.")
