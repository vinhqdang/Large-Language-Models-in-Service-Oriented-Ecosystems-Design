# CADENCE: LLM-Assisted Architectural Decision-Making — Design Spec

*CADENCE = **C**ase-grounded **A**gentic **D**eliberation with **C**onstraint-verified **N**egotiation and **C**ritique **E**ngine (working title, placeholder — see §9).*

**Date:** 2026-08-25
**Target venue:** IEEE Transactions on Services Computing — Special Issue on "Large Language Models in Service-Oriented Ecosystems Design: Advances and Applications"
**Submission deadline:** 31 October 2026 (revision due 1 May 2027, final decision 1 June 2027)
**Author:** Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam (vinh.dq4@buv.edu.vn)
**Review policy:** Single-anonymous (confirmed via IEEE Computer Society author-resources page) — repo link and author identity may appear openly in the manuscript.

## 1. Problem and Contribution

The special issue calls for LLM-driven approaches to service-oriented architecture design, explicitly including "LLM-assisted design of service-based software architectural solutions" and assessment of impact on reliability/maintainability. This work targets **architectural decision-making**: given a set of requirements and quality-attribute constraints for a service-oriented system, produce a traceable, justified architectural decision (an ADR) that is grounded in precedent, debated across competing quality-attribute concerns, and formally checked for feasibility before being finalized.

**Contribution claim:** a new four-stage hybrid algorithm — retrieval-augmented case-based reasoning, knowledge-graph-grounded multi-agent deliberation, constraint-solver-verified feasibility with an LLM repair loop, and a separate self-critique finalization stage — with no existing system in the literature combining all four (see §2). This is a novel algorithmic contribution, not an extension of a prior conference paper.

## 2. Related Work and Novelty Positioning

| System | Retrieval from real ADR corpus | Multi-agent | Agent roles | Formal solver verification | Separate critique stage |
|---|---|---|---|---|---|
| Dhar et al., ICSA 2024 (arXiv:2403.01709) | No | No (single LLM) | — | No | No |
| Context Matters (arXiv:2604.03826) | Yes (750-repo corpus, context strategies) | No | — | No | No |
| MAAD (arXiv:2507.21382, forthcoming ACM TOSEM) | No | Yes (4 agents) | Functional pipeline roles: Analyst, Modeler, Designer, Evaluator | No (Evaluator is an LLM judge, not a solver) | No (evaluation folded into pipeline) |
| **CADENCE (this work)** | Yes (Buchgeher et al. corpus) | Yes | Quality-attribute *advocate* agents (performance, security, maintainability, scalability, cost/operability) in structured debate | **Yes** — weighted MaxSAT/SMT (Z3) feasibility check + repair loop | **Yes** — distinct final utility-scoring pass |

MAAD is the closest prior system and must be discussed carefully and fairly in the manuscript: the differentiation is (a) agents represent competing quality-attribute *concerns* in an adversarial/argumentative protocol rather than sequential functional pipeline roles, (b) feasibility is checked by a formal constraint solver producing an unsat core the agents can target for repair — not by a second LLM opinion, and (c) retrieval is grounded in a real, citable, mined ADR corpus rather than requirements-only generation.

General neuro-symbolic LLM+solver techniques exist (Logic-LM, LINC, MCP-Solver) but are not applied to architectural trade-off verification — cited as prior art for the *technique*, not the *application*.

**Baselines for evaluation** (§5): (a) Dhar-et-al.-style single-LLM zero-shot ADR generation, (b) Context-Matters-style retrieval-only (no deliberation, no solver), (c) MAAD-style multi-agent without solver verification (ablation), (d) human-authored ground-truth ADRs from the corpus.

## 3. Algorithm (4 Stages)

```
Requirements + QA constraints
        │
        ▼
[1] Retrieval (case-based reasoning)
    - Embed decision context
    - Vector-search top-k precedent ADRs from Buchgeher corpus
        │
        ▼
[2] Multi-agent deliberation (KG-grounded)
    - N agents, each anchored to one ISO/IEC 25010 quality attribute
      (performance, security, maintainability, scalability, cost/operability)
    - Agents propose/critique candidate decisions over a knowledge graph
      of architectural patterns, tactics, and known trade-offs
    - Bounded-round structured argumentation → converged candidate + rationale
        │
        ▼
[3] Constraint-solver verification + repair loop
    - Encode candidate's implied trade-off commitments as weighted
      MaxSAT/SMT (Z3)
    - If infeasible: return unsat core to agents for targeted repair
    - Iterate to a fixed cap; if still infeasible, degrade gracefully
      (best partial-feasibility candidate + explicit caveat)
        │
        ▼
[4] Self-critique / finalization
    - Separate LLM pass scores the converged, solver-verified decision
      against explicit per-attribute utility functions
    - Flags residual weaknesses
    - Emits final ADR with full provenance: precedents used, agent
      positions, constraints checked, repair iterations
```

## 4. Data and Knowledge Sources

- **ADR corpus:** Buchgeher, Schöberl, Geist, Dorninger, Haindl, Weinreich, "Using Architecture Decision Records in Open Source Projects — An MSR Study on GitHub," *IEEE Access*, vol. 11, pp. 63725–63740, 2023. DOI: [10.1109/ACCESS.2023.3287654](https://doi.org/10.1109/ACCESS.2023.3287654). 921 repositories with Markdown ADRs; scraped to ~5,262 individual ADRs (per DRAFT-ing paper, arXiv:2504.08207) usable in JSON form for retrieval indexing and as evaluation ground truth (held-out split).
- **Quality-attribute taxonomy:** ISO/IEC 25010 product quality model, used to define the fixed set of deliberation agents and the self-critique utility functions.
- **Knowledge graph:** built from architectural patterns/tactics literature (e.g., Bass/Clements/Kazman tactics catalog) linked to quality attributes and known trade-offs; construction method to be detailed in the implementation plan.

## 5. Evaluation Plan

- **Metrics (standard, for direct comparability):** BERTScore, BLEU, ROUGE-1, METEOR against held-out human-authored ADRs — matches the Context Matters paper's metric set.
- **Metrics (novel to this pipeline):** constraint-satisfaction rate (% of decisions solver-verified feasible) and repair-loop convergence (iterations to feasibility) — baselines without a solver cannot produce these, strengthening the novelty argument quantitatively.
- **Ablations:** remove each stage in turn (no retrieval / no multi-agent / no solver / no self-critique) to isolate each component's contribution.
- **Human/LLM-judge evaluation:** quality-attribute trade-off soundness scoring on a sample, to complement automatic metrics.

## 6. LLM Backend

- **Primary:** Gemini API, model `gemini-3.5-flash-lite`.
- **Secondary (reproducible/free backbone):** local open-weight model via CUDA (e.g., Qwen2.5-7B/14B-Instruct), run under conda env `py313`. Chosen over OpenRouter's free tier because the only current free general chat model (`dots-studio/dots-3-note-preview:free`) is scheduled for discontinuation 30 Sept 2026 — before the review cycle completes (revision due May 2027) — making it unsuitable as a load-bearing reproducibility backbone.
- **Optional tertiary comparison:** OpenRouter free-tier model, used only as a supplementary, clearly-labeled data point, not a primary result.
- API keys stored only in a local, gitignored `.env` (already covered by existing `.gitignore`); never committed.

## 7. Repository Layout

```
src/
  retrieval/       # embedding + vector index over ADR corpus
  deliberation/     # multi-agent QA-concern debate protocol, KG interface
  solver/           # Z3-based feasibility encoding + repair-loop driver
  critique/         # final self-critique / utility scoring
data/                # Buchgeher corpus (processed), knowledge graph
manuscript/          # LaTeX (IEEEtran, from local Computer_Society_LaTeX_template.zip)
tests/
docs/
.env                 # local only, gitignored
```

## 8. Manuscript Structure

Six sections: (1) Introduction, (2) Related Work, (3) Method (CADENCE algorithm), (4) Evaluation, (5) Discussion, (6) Conclusion. IEEEtran class from the local template zip, ≤14 pages per CFP. All references DOI-verified per standing research-project requirement.

## 9. Open Risks

- Constraint encoding (turning agent-debated trade-offs into a solvable MaxSAT/SMT instance) is the highest-risk engineering component and needs its own design pass during implementation planning.
- Corpus quality: Buchgeher corpus ADR adoption is sparse per-repo (~50% of repos have 1–5 ADRs) — retrieval index must handle this sparsity; may need dedup/quality filtering.
- Working title "CADENCE" is a placeholder; can be revisited once the paper's narrative is fully drafted.
