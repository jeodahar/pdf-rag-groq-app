import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Set Streamlit Page Configuration
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")

st.title("📚 RAG PDF Assistant (Powered by Groq & FAISS)")
st.caption("Upload a PDF document, build a local Vector DB, and query it using open-source models.")

# Sidebar for Setup & API Key configuration
with st.sidebar:
    st.header("🔑 Configuration")
    
    # Retrieve key from environment variable (Streamlit Secrets / local env) or UI text input
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    # Model Selection from open-source models supported by Groq
    model_option = st.selectbox(
        "Select LLM Model:",
        options=[
            openai/gpt-oss-120b""
           
        ],
        index=0
    )

# Function to process PDF and return FAISS Vector Store
@st.cache_resource(show_spinner=False)
def create_vector_db(file_bytes, filename):
    # Save uploaded file into a temporary file path for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    # 1. Extract text from PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    os.remove(tmp_file_path)

    # 2. Chunk documents into token-friendly segments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # 3. Create Embeddings using open-source HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store in open-source FAISS Vector DB
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    if not groq_api_key:
        st.error("Please provide a Groq API Key in the sidebar or set the `GROQ_API_KEY` secret.")
    else:
        with st.spinner("Extracting PDF, tokenizing chunks, and building vector index..."):
            try:
                # Read bytes and process
                file_bytes = uploaded_file.read()
                vector_db = create_vector_db(file_bytes, uploaded_file.name)
                st.success("Vector Database created successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()

        # Chat Interface Initialization
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render past chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User Query Input
        if user_prompt := st.chat_input("Ask a question about your document..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching document & generating answer..."):
                    try:
                        # RAG Step 1: Retrieve top-3 relevant context chunks using modern invoke()
                        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                        retrieved_docs = retriever.invoke(user_prompt)
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                        # RAG Step 2: Query Groq LLM with context-augmented prompt
                        client = Groq(api_key=groq_api_key)
                        
                        system_prompt = (
                            "You are a helpful assistant. Use ONLY the provided context from the user's PDF "
                            "to answer the user's query. If you cannot answer using the context, state that clearly.\n\n"
                            f"Context:\n{context}"
                        )

                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            model=model_option,
                        )

                        response_text = chat_completion.choices[0].message.content
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
