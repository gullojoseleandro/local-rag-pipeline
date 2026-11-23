"""Validated output contract for grounded answers."""

from pydantic import BaseModel, Field, model_validator


class RagResponse(BaseModel):
    texto: str = Field(min_length=1, description="Answer based only on retrieved context.")
    referencias: list[str] = Field(
        default_factory=list, description="Names of source documents supporting the answer."
    )

    @model_validator(mode="after")
    def check_abstention(self) -> "RagResponse":
        if self.texto.strip() == "No lo sé":
            if self.referencias:
                raise ValueError("An abstention cannot cite sources.")
        elif not self.referencias:
            raise ValueError("A factual answer must cite at least one retrieved source.")
        return self
