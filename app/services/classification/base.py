"""Abstract Base Classifier and Result Model."""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Encapsulates the decision and extracted metadata of message classification."""

    is_passenger_request: bool = Field(
        ...,
        description="True if the message is a genuine passenger seeking transportation",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the classification decision (0.0 to 1.0)",
    )
    origin: Optional[str] = Field(
        default=None,
        description="Detected origin / departure point",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Detected destination / arrival point",
    )
    seats: Optional[int] = Field(
        default=None,
        description="Detected number of passengers or requested seats",
    )
    order_type: Optional[str] = Field(
        default="passenger",
        description="Type of order: 'passenger' (yo'lovchi) or 'cargo' (pochta/yuk)",
    )
    time: Optional[str] = Field(
        default=None,
        description="Detected departure time if mentioned (e.g. 'Hozir', 'Kechga', 'Soat 5da')",
    )
    reason: str = Field(
        default="",
        description="Diagnostic explanation of why the message was or was not classified as a passenger request",
    )
    method: str = Field(
        default="rule_based",
        description="Classification engine used ('rule_based' or 'llm')",
    )


class BaseClassifier(ABC):
    """Abstract interface for all message classifiers."""

    @abstractmethod
    async def classify(self, raw_text: str) -> ClassificationResult:
        """Classify raw incoming message text and extract ride details."""
        raise NotImplementedError
