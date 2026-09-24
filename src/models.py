from typing import List

from pydantic import BaseModel, Field


class SubQueries(BaseModel):
    queries: List[str] = Field(
        description="3 to 5 targeted sub-queries to thoroughly research the topic"
    )


class ResearchReport(BaseModel):
    executive_summary: str = Field(description="A brief summary of the findings.")
    key_findings: List[str] = Field(
        description="Bullet points of detailed technical findings"
    )
    citations: List[str] = Field(description="List of URLs used as soruces.")
