import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from google.generativeai import configure
import google.generativeai as genai

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define structured prompt
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
    # Load document
    loader = PyPDFLoader(r"C:\Users\DCS\Desktop\document-qa-system\data\Sai_Resume.pdf")
    documents = loader.load()
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Required model
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create vector store
    db = FAISS.from_documents(texts, embeddings)
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",  # Updated model name
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
    ),
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}  # Pass the PromptTemplate instance
)
    
    # Interactive Q&A
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        result = qa_chain.invoke({"query": query})
        print(f"\nAnswer: {result['result']}")

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("ERROR: Set GOOGLE_API_KEY in .env file")
        exit()
    main()