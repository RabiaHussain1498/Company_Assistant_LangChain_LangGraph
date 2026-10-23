"""Data shapes (Pydantic models) that the LLM is forced to answer with."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Intent(BaseModel):
    """What the user wants, extracted from their message."""

    intent: Literal["lookup_company", "ask_saved", "chitchat"] = Field(
        description=(
            "lookup_company: the user wants to learn about a company or website. "
            "ask_saved: the user asks about companies that were already saved. "
            "chitchat: general conversation."
        )
    )
    company: Optional[str] = Field(
        None, description="Company name, if it appears in the message"
    )
    url: Optional[str] = Field(
        None, description="Website URL, if it appears in the message"
    )