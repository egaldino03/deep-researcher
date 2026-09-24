from config import INDEX_NAME, OPENROUTER_API_KEY
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone_client import pc

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
