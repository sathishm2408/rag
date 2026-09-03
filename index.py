from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

loader = PyPDFLoader(file_path=pdf_path)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = text_splitter.split_documents(doc)


# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large"
# )

# Gemini Developer API
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="documents_rag",
    url="http://localhost:6333"
)