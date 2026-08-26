"""Shared CANDIDATE:/RATIONALE: text format parsing.

Deliberately duplicated (not imported) from
src/deliberation/orchestrator.py's private _parse_synthesis — see the
constraint-solver plan's Global Constraints for why.
"""
import re


class CandidateRationaleParseError(RuntimeError):
    """Raised when text doesn't contain a parseable CANDIDATE/RATIONALE pair."""


_LABEL_LINE = re.compile(r"^\s*[*_\-\s]*(CANDIDATE|RATIONALE)[*_\s]*:[*_\s]*(.*)$", re.IGNORECASE)


def parse_candidate_rationale(text: str) -> tuple[str, str]:
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
        raise CandidateRationaleParseError(
            f"Could not parse both CANDIDATE and RATIONALE from: {text!r}"
        )
    return candidate, rationale
