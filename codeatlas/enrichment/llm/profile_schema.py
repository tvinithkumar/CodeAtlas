from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SymbolProfile:
    description: str = ""
    responsibilities: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    search_tags: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.description,
                self.responsibilities,
                self.inputs,
                self.outputs,
                self.side_effects,
                self.failure_modes,
                self.search_tags,
            ]
        )

    def to_profile_text(self) -> str:
        parts: list[str] = []
        if self.description:
            parts.append(self.description)
        if self.responsibilities:
            parts.append("Responsibilities: " + ", ".join(self.responsibilities))
        if self.search_tags:
            parts.append("Search tags: " + ", ".join(self.search_tags))
        if self.inputs:
            parts.append("Inputs: " + ", ".join(self.inputs))
        if self.outputs:
            parts.append("Outputs: " + ", ".join(self.outputs))
        if self.side_effects:
            parts.append("Side effects: " + ", ".join(self.side_effects))
        if self.failure_modes:
            parts.append("Failure modes: " + ", ".join(self.failure_modes))
        return "\n".join(parts)

    @property
    def tags(self) -> list[str]:
        return self.search_tags


PROFILE_SCHEMA_TEXT = """{
  "description": "concise explanation of what the symbol does",
  "responsibilities": ["main responsibilities"],
  "inputs": ["parameters, data dependencies, config keys"],
  "outputs": ["return values, emitted values, state changes"],
  "side_effects": ["logs, metrics, network calls, database writes"],
  "failure_modes": ["likely errors or edge cases"],
  "search_tags": ["terms developers might search for"]
}"""

