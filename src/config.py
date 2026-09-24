import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY")
TAVILY_API_KEY: str | None = os.environ.get("TAVILY_API_KEY")
