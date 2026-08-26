"""Quality-attribute advocate agent (CADENCE Stage 2)."""
from dataclasses import dataclass

from src.deliberation.knowledge_graph import supporting_tactics_for
from src.retrieval.records import ADRRecord


@dataclass(frozen=True)
class AgentPosition:
    quality_attribute: str
    round_number: int
    stance: str
    content: str


class QualityAttributeAgent:
    def __init__(self, quality_attribute: str, llm_client, knowledge_graph):
        self._quality_attribute = quality_attribute
        self._client = llm_client
        self._graph = knowledge_graph

    @property
    def quality_attribute(self) -> str:
        return self._quality_attribute

    def _system_prompt(self) -> str:
        tactics = supporting_tactics_for(self._graph, self._quality_attribute)
        lines = [
            f"You are the {self._quality_attribute} advocate in an architectural "
            "decision-making deliberation. Argue for the decision that best serves "
            f"{self._quality_attribute}, while acknowledging real trade-offs.",
            f"Tactics you can draw on for {self._quality_attribute}: " + ", ".join(tactics),
        ]
        return "\n".join(lines)

    def propose(self, context: str, precedents: list[ADRRecord]) -> AgentPosition:
        precedent_lines = "\n".join(f"- {p.title}" for p in precedents) or "(no precedents retrieved)"
        prompt = (
            f"Decision context:\n{context}\n\n"
            f"Precedent decisions from similar past projects:\n{precedent_lines}\n\n"
            f"Propose and justify a position from the {self._quality_attribute} "
            "perspective for this decision."
        )
        content = self._client.generate(prompt, system=self._system_prompt())
        return AgentPosition(self._quality_attribute, round_number=1, stance="propose", content=content)

    def critique(
        self, candidate_text: str, other_positions: list["AgentPosition"], round_number: int
    ) -> AgentPosition:
        others_lines = "\n".join(
            f"- [{p.quality_attribute}] {p.content}" for p in other_positions
        ) or "(no other positions yet)"
        prompt = (
            f"Current candidate decision:\n{candidate_text}\n\n"
            f"Other advocates' positions this round:\n{others_lines}\n\n"
            f"From the {self._quality_attribute} perspective, critique or refine "
            "this candidate, citing specific trade-offs where relevant."
        )
        content = self._client.generate(prompt, system=self._system_prompt())
        return AgentPosition(self._quality_attribute, round_number=round_number, stance="critique", content=content)
