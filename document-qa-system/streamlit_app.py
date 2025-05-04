import streamlit as st  # Add this import at top
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import tempfile

# ... keep existing imports and setup ...

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define your custom prompt template
CUSTOM_PROMPT = PromptTemplate(
    template="""
    Analyze the context and answer concisely:
    - Respond in 3-5 sentences max
    - If unsure, state "Not found in document"
    - Use bullet points when listing items

    Context: {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

def main():
    # Streamlit UI Configuration
    st.set_page_config(page_title="Document QA with Gemini", layout="wide")
    st.title("🦜🔗 Document QA System with Gemini")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load document
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create vector store
        db = FAISS.from_documents(texts, embeddings)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            ),
            chain_type="stuff",
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )
        
        # Query interface
        query = st.text_input("Ask about the document:")
        if query:
            with st.spinner("Analyzing..."):
                result = qa_chain.invoke({"query": query})
                st.subheader("Answer:")
                st.markdown(result['result'])

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        st.error("Set GOOGLE_API_KEY in .env file")
    else:
        main()
