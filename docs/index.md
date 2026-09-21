# QSeaBattle

QSeaBattle is a simulation and machine-learning project for exploring classical, learned, and non-classical strategies in distributed guessing games.

The project grew out of a series of investigations into random access codes, nonlocal correlations, trainable assisted strategies, and the geometry behind quantum advantage.

<div class="grid cards" markdown>

-   :material-book-open-page-variant:{ .lg .middle } **Read The Story**

    ---

    The ideas behind QSeaBattle are explained in a series of articles on  
    **The Armchair Quantum Physicist**.

    [Read the articles →](https://armchairquantumphysicist.com/)

-   :fontawesome-brands-github:{ .lg .middle } **Explore The Code**

    ---

    Browse the source code, notebooks, tests, and simulations used in the project.

    [Open the GitHub repository →](https://github.com/robhendrik/QSeaBattle)

-   :material-file-document-multiple:{ .lg .middle } **Technical Documentation**

    ---

    Explore the algorithms, players, models, utilities, and generated API documentation.

    [Browse the documentation →](algorithms.md)

</div>

---

## The Article Series

The project is accompanied by five articles that develop the ideas step by step.

1. **[The Best Bit Alice Can Send](https://armchairquantumphysicist.com/2026/09/01/the-best-bit-alice-can-send/)**  
   Why the most informative message isn’t the winning one.

2. **[Beating the Odds in a Guessing Game — with a Single Quantum Box](https://armchairquantumphysicist.com/2026/09/01/beating-the-odds-in-a-guessing-game-with-a-single-quantum-box/)**  
   Alice sends Bob one bit and tells him nothing new — yet he guesses right more often than he should. A small quantum advantage, in the simplest game there is.

3. **[Quantum Mechanics Should Make This More Complicated. It Doesn’t.](https://medium.com/science-spectrum/quantum-mechanics-should-make-this-more-complicated-it-doesnt-9f59275ab189)**  
   A guessing game becomes geometrically more and more complicated — until we add quantum mechanics.

4. **Why Isn’t Nature More Quantum? (to be published)**  
   Quantum physics draws a boundary. Information theory approaches the same limit from a completely different direction.

5. **Is quantum mechanics the optimal balance between what we can know and what we can do?**  
   Quantum theory achieves in some sense an optimal balance of allowed states and dynamics.

---

## What You Will Find Here

The documentation covers:

- the game and tournament framework
- classical baseline strategies
- majority and assisted players
- PR-assisted strategies
- trainable assisted models
- linear and pyramidal architectures
- dataset generation and conversion
- imitation-learning utilities
- analysis and reference-performance tools

If you are mainly interested in the implementation, start with the **Algorithms** and **Players** sections.

If you want to understand the motivation and the physics behind the project, start with the article series above.

---

## Repository Structure

```text
QSeaBattle/
├─ docs/          Documentation source
├─ notebooks/     Tutorials and experiments
├─ presentation/  Presentation material
├─ src/           QSeaBattle source code
├─ tests/         Automated tests
└─ tools/         Documentation and maintenance tools