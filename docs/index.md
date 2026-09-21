# QSeaBattle

QSeaBattle is a simulation and machine-learning project for exploring classical, learned, and non-classical strategies in distributed guessing games.

The project grew out of a series of investigations into random access codes, nonlocal correlations, trainable assisted strategies, and the geometry behind quantum advantage.

<div class="grid cards" markdown>

-   **Read The Story**

    ---

    The ideas behind QSeaBattle are explained in a series of articles on  
    **The Armchair Quantum Physicist**.

    [Read the articles →](https://armchairquantumphysicist.com/)

-   **Explore The Code**

    ---

    Browse the source code, notebooks, tests, and simulations used in the project.

    [Open the GitHub repository →](https://github.com/robhendrik/QSeaBattle)

-   **Technical Documentation**

    ---

    Explore the algorithms, players, models, utilities, and generated API documentation.

    [Browse the documentation →](algorithms.md)

</div>

---
## Tutorials

The project provides tutorials that show how the game, tournaments and models can be used.

1. **[A quick start guide showing how to run the provided players in a tournament](https://robhendrik.github.io/QSeaBattle/html/Tutorial_QSeaBattle_QuickStartGuide.html)**

2. **[Demonstration for a neural-net player that is trained to imitate the majority algorithm](https://github.com/robhendrik/QSeaBattle/html/Tutorial_imitation_training_neural_net_models.html)**

3. **[Self training of a neural-net based player](https://github.com/robhendrik/QSeaBattle/html/Tutorial_DIAL_DRU_training_neural_net_models.html)**

4. **[Imitation training of a player using post-quantum resources with the 'linear' model](https://github.com/robhendrik/QSeaBattle/html/Tutorial_LinTrainableAssisted_Imitation.html)**

5. **[Imitation training of a player using post-quantum resources with the 'pyramid' model](https://github.com/robhendrik/QSeaBattle/html/Tutorial_PyrTrainableAssisted_Imitation.html)**

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

5. **Is quantum mechanics the optimal balance between what we can know and what we can do? (to be published)**  
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

- `docs/` — Documentation source
- `notebooks/` — Tutorials and experiments
- `presentation/` — Presentation material
- `src/` — QSeaBattle source code
- `tests/` — Automated tests
- `tools/` — Documentation and maintenance tools