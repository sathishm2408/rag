from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Gemini Developer API
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-2-preview"
# )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# qdrant_client = QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=60)  # 60s

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="documents_rag",
    url="http://localhost:6333",
    timeout=60,  # 60s
    # client=qdrant_client
)

user_query = input("Enter your query: ")

search_results = vector_db.similarity_search(query=user_query)


context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])


SYSTEM_PROMPT = f"""
 You are a helpfull AI Assistant who answeres user query based on the available context retrieved from a PDF file along with page_contents and page number.

 You should only ans the user based on the following context and navigate the user to open the right page number to know more.

 Context:
 {context}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        { "role": "system", "content":SYSTEM_PROMPT  },
        { "role": "user", "content":user_query  },
    ]
)

print(f"AI response: {response.choices[0].message.content}")