"""Bounded-round multi-agent deliberation orchestrator (CADENCE Stage 2)."""
import re
from dataclasses import dataclass

from src.deliberation.agent import AgentPosition, QualityAttributeAgent
from src.retrieval.records import ADRRecord


class SynthesisParseError(RuntimeError):
    """Raised when the synthesizer's response doesn't contain a parseable
    CANDIDATE/RATIONALE pair — surfaced loudly rather than silently
    returning an empty converged candidate."""


@dataclass(frozen=True)
class DeliberationResult:
    converged_candidate: str
    rationale: str
    transcript: list[AgentPosition]
    # Total rounds run, including round 1 (propose) — e.g. max_rounds=1
    # means only the propose round ran, with zero critique rounds.
    rounds_run: int


_LABEL_LINE = re.compile(r"^\s*[*_\-\s]*(CANDIDATE|RATIONALE)[*_\s]*:[*_\s]*(.*)$", re.IGNORECASE)


def _parse_synthesis(text: str) -> tuple[str, str]:
    fields: dict[str, list[str]] = {"CANDIDATE": [], "RATIONALE": []}
    current: str | None = None

    for line in text.splitlines():
        match = _LABEL_LINE.match(line)
        if match:
            current = match.group(1).upper()
            remainder = match.group(2).strip()
            if remainder:
                fields[current].append(remainder)
        elif current is not None and line.strip():
            fields[current].append(line.strip())

    candidate = " ".join(fields["CANDIDATE"]).strip()
    rationale = " ".join(fields["RATIONALE"]).strip()

    if not candidate or not rationale:
        raise SynthesisParseError(
            f"Could not parse both CANDIDATE and RATIONALE from synthesizer output: {text!r}"
        )
    return candidate, rationale


class DeliberationOrchestrator:
    def __init__(self, agents: list[QualityAttributeAgent], synthesizer_client, max_rounds: int = 3):
        quality_attributes = [agent.quality_attribute for agent in agents]
        if len(quality_attributes) != len(set(quality_attributes)):
            raise ValueError(f"Agents must have distinct quality attributes, got: {quality_attributes}")

        self._agents = agents
        self._synthesizer = synthesizer_client
        self._max_rounds = max_rounds

    def deliberate(self, context: str, precedents: list[ADRRecord]) -> DeliberationResult:
        transcript: list[AgentPosition] = []

        round_1 = [agent.propose(context, precedents) for agent in self._agents]
        transcript.extend(round_1)
        latest = round_1

        for round_number in range(2, self._max_rounds + 1):
            candidate_text = "\n".join(p.content for p in latest)
            next_round = []
            for index, agent in enumerate(self._agents):
                others = [p for i, p in enumerate(latest) if i != index]
                next_round.append(agent.critique(candidate_text, others, round_number))
            transcript.extend(next_round)
            latest = next_round

        transcript_text = "\n".join(f"[{p.quality_attribute} r{p.round_number}] {p.content}" for p in transcript)
        synthesis_prompt = (
            f"Decision context:\n{context}\n\n"
            f"Full deliberation transcript:\n{transcript_text}\n\n"
            "Synthesize the deliberation into a final decision. Respond in exactly this format:\n"
            "CANDIDATE: <one or two sentence decision>\n"
            "RATIONALE: <one paragraph rationale>"
        )
        synthesis = self._synthesizer.generate(synthesis_prompt)
        candidate, rationale = _parse_synthesis(synthesis)

        return DeliberationResult(
            converged_candidate=candidate,
            rationale=rationale,
            transcript=transcript,
            rounds_run=self._max_rounds,
        )
