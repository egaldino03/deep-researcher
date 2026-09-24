import os
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import OPENROUTER_API_KEY


class EphemeralVectorStore:
    def __init__(self, collection_name: str = "ephemeral_research"):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.vector_store = Chroma(
            collection_name=collection_name, embedding_function=self.embeddings
        )

    def ingest_search_results(self, results: List[Dict[str, str]]):
        """Chunks content and loads LangChain Document objects into Chroma."""
        documents = []

        for res in results:
            content = res["content"]
            url = res["url"]

            chunks = [content[i : i + 500] for i in range(0, len(content), 400)]

            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    documents.append(
                        Document(page_content=chunk, metadata={"source": url})
                    )
        if documents:
            self.vector_store.add_documents(documents)

    def query(
        self, query_text: str, n_results: int = 6
    ) -> Tuple[List[str], List[dict]]:
        """Retrieve top relevant documents using LangChain similarity search."""
        try:
            results = self.vector_store.similarity_search(query=query_text, k=n_results)
        except Exception:
            return [], []

        chunks = [doc.page_content for doc in results]
        sources = [doc.metadata for doc in results]

        return chunks, sources
