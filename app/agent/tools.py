from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch


@tool
def search(query: str):
    """Searches the web and returns relevant answers"""
    tavily_search = TavilySearch(
        max_result=12, topic="general", search_depth="advanced"
    )
    return tavily_search.invoke({"query": query})
