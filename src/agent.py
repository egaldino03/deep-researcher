import os
from typing import List

from langchain_openrouter import ChatOpenRouter

from config import OPENROUTER_API_KEY
from models import ResearchReport, SubQueries
from search import SearchService
from vectorstore import EphemeralVectorStore


class DeepResearchAgent:
    def __init__(self, model_name: str = "deepseek/deepseek-v4.1-flash"):
        self.llm = ChatOpenRouter(
            model=model_name, temperature=0.2, api_key=OPENROUTER_API_KEY
        )
        self.search_service = SearchService()

    def _decompose_query(self, user_prompt: str) -> List[str]:
        structured_llm = self.llm.with_structured_output(SubQueries)
        prompt = f"""You are a research assistant. Break down the user's topic into 3 to 5 highly specific sub-queries to maximize information gathering. Topic: {user_prompt}"""
        response = structured_llm.invoke(prompt)
        return response.queries

    def _synthesize_report(
        self, user_prompt: str, context_chunks: List[str], sources: List[dict]
    ) -> ResearchReport:
        formatted_context = "\n\n".join(
            [
                f"Source ({s.get('source')}): {c}"
                for c, s in zip(context_chunks, sources)
            ]
        )

        structured_llm = self.llm.with_structured_output(ResearchReport)
        prompt = f"""You are a professional research analyst. Synthesize a comprehensive report using ONLY the provided context chunks. Include explicit citations. \n\n Topic: {user_prompt}\n\n Context:\n{formatted_context}"""
        return structured_llm.invoke(prompt)

    def run(self, user_prompt: str) -> ResearchReport:
        print(f"[*] Decomposing query...")
        sub_queries = self._decompose_query(user_prompt)

        print(f"[*] Fetching web data via Tavily for queries: {sub_queries}")
        raw_results = self.search_service.search_queries(sub_queries)

        print(f"[*] Initializing langchain-chroma vector store and embedding chunks...")
        vector_store = EphemeralVectorStore()
        vector_store.ingest_search_results(raw_results)

        print(f"[*] Retrieving relevant context...")
        chunks, sources = vector_store.query(user_prompt, n_results=6)

        flat_chunks = chunks if chunks else []
        flat_sources = sources if sources else []

        print(f"[*] Synthesizing final report...")
        return self._synthesize_report(user_prompt, flat_chunks, flat_sources)


if __name__ == "__main__":
    agent = DeepResearchAgent()
    report = agent.run(
        "What are the architectural patterns behind state-of-the-art coding agents in 2026?"
    )

    print("\n--- EXECUTIVE SUMMARY ---")
    print(report.executive_summary)
    print("\n--- KEY FINDINGS ---")
    for f in report.key_findings:
        print(f"* {f}")
    print("\n--- CITATIONS ---")
    for c in set(report.citations):
        print(f" - {c}")
