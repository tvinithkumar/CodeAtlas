from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMEnrichment:
    description: str
    tags: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    def to_profile_text(self) -> str:
        parts = [self.description]
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        if self.inputs:
            parts.append("Inputs: " + ", ".join(self.inputs))
        if self.outputs:
            parts.append("Outputs: " + ", ".join(self.outputs))
        if self.failure_modes:
            parts.append("Failure modes: " + ", ".join(self.failure_modes))
        return "\n".join(parts)

