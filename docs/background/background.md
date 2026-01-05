# QSeaBattle -- Design Specification

## Purpose
This document is the **authoritative design specification** for the QSeaBattle project.
It defines the game mechanics, player models, algorithms, training regimes, and utilities
that together constitute the QSeaBattle system.

If any behavior in the codebase contradicts this specification, the code is considered incorrect.


## Scope and audience
This specification is intended for:
- Developers implementing or modifying the QSeaBattle codebase
- Researchers reviewing the algorithmic and information-theoretic aspects
- Future maintainers re-implementing components independently

This is **not** a user tutorial; it is a technical design contract.


## Structure of the specification

The document is organized into the following logical parts:

1. **Algorithms**
   - High-level, informative explanations of the classical assisted strategy,
     trainable Lin models, and hierarchical Pyr models.

2. **Conventions**
   - Global notation, terminology, and RFC-style contract language.

3. **Game infrastructure**
   - Game layout, environment, game execution, and tournaments.

4. **Players**
   - Abstract player interfaces and concrete player families
     (deterministic, assisted, neural).

5. **Shared randomness**
   - Classical shared-randomness resources and differentiable variants.

6. **Models**
   - Trainable assisted models (Lin and Pyr) with strict information constraints.

7. **Training**
   - Imitation learning, DRU/DIAL, and reinforcement learning regimes.

8. **Utilities**
   - Dataset generation, training helpers, evaluation, and weight transfer.

9. **Invariants**
   - One-sentence global truths that must always hold.


## Normative vs informative content

- Chapters describing components, interfaces, and constraints are **normative**.
- The Algorithms chapter is **informative** and provides intuition only.
- In case of conflict, **normative chapters take precedence**.

Normative language uses:
- **MUST / MUST NOT**
- **SHOULD / SHOULD NOT**
- **MAY**


## Versioning and stability

This specification evolves with the codebase.
Breaking changes to behavior MUST be accompanied by updates to this document.

A frozen PDF export may be tagged as:
- *QSeaBattle Specification vX.Y*


## How to read this document

- Start with **Algorithms** for intuition
- Read **Conventions** and **Invariants** carefully
- Then read the chapters relevant to the component you are working on


## Repository layout (informative)

```
QSeaBattle/
|-- docs/       # This specification (Markdown source)
|-- mkdocs.yml  # Documentation build configuration
|-- src/        # Implementation
|-- tests/      # Tests (should reference invariants)
`-- notebooks/  # Experiments and analysis
```


## Design principle (non-negotiable)

> Information flow is the primary design constraint.  
> Learning may approximate logic, but must never bypass it.