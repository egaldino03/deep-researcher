import os
from typing import Dict, List

from tavily import TavilyClient


class SearchService:
    def __init__(self):
        self.client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    def search_queries(
        self, queries: List[str], max_results_per_query: int = 3
    ) -> List[Dict[str, str]]:
        """Executes parallel or sequential Tavily searches and normalizes results."""
        all_results = []
        for q in queries:
            try:
                response = self.client.search(
                    query=q, max_results=max_results_per_query, search_depth="advanced"
                )
                for item in response.get("results", []):
                    all_results.append(
                        {"url": item.get("url"), "content": item.get("content")}
                    )
            except Exception as e:
                print(f"Error searching for {q}: {e}")
        return all_results
