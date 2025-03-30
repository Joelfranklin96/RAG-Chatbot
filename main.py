from dotenv import load_dotenv

load_dotenv()
from typing import Set

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import os
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader

from consts import INDEX_NAME

import streamlit as st
from streamlit_chat import message

from backend.core import run_llm

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Add these imports
from PIL import Image
import requests
from io import BytesIO


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    
    # Initialize Pinecone with the new syntax
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if index exists and delete it
    if INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(INDEX_NAME)
    
    # Create a new serverless index
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Dimension for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    # Now use LangChain's PineconeVectorStore
    # Note: LangChain should handle the new Pinecone initialization internally
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)


# Custom CSS for dark theme and modern look
st.markdown(
    """
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #252526;
    }
    .stMessage {
        background-color: #2D2D2D;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Set page config at the very beginning


# Sidebar user information
with st.sidebar:
    st.title("Proof of Concept")
    # The above line of command sets the title as 'User Profile'.

    # You can replace these with actual user data
    user_name = "Joel Franklin"
    user_email = "joel.vijayakumar@emerjence.com"

    image_path = "profile_pic.jpg"
    profile_pic = Image.open(image_path)
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")
    pdf_docs = st.file_uploader("Upload your PDF Files. Click on the 'Submit & Process' Button", accept_multiple_files=True)
        # st.file_uploader() creates an interactive file uploader widget which allows users to select and upload files from their 
        # local machine. pdf_docs will be a list of uploaded PDF documents.
    if st.button("Submit & Process"):
        # The st.button("Submit & Process") command creates an interactive button. When a user clicks this button, it returns True.
        with st.spinner("Processing..."):
            # While the code block inside with st.spinner("Processing") is running, it displays a spinner with the message "Processing..".
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")
            # The above displays a success message. It adds a visible message element to the app's main layout.

st.header("RAG Chatbot for Emerjence")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Create two columns for a more modern layout
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_input("Prompt", placeholder="Enter your message here...")

with col2:
    if st.button("Submit", key="submit"):
        prompt = prompt or "Hello"  # Default message if input is empty

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        #sources = set(doc.metadata["source"] for doc in generated_response["context"])
        #formatted_response = (
        #    f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        #)

        formatted_response = (
            f"{generated_response['answer']}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append({"role":"human", "content": prompt})
        #st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append({"role":"assistant", "content":generated_response["answer"] })
        #st.session_state["chat_history"].append(("ai", generated_response["answer"]))

# Display chat history
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)

