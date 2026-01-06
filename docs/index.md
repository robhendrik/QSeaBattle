# QSeaBattle
*A quantum-inspired coordination game framework*

---

## What is QSeaBattle?

**QSeaBattle** is a research and experimentation framework for studying how **agents coordinate under severe communication constraints**.

Two cooperative players must make a correct decision about a hidden battlefield, but:

- One player sees **everything**
- The other sees **almost nothing**
- Only **a few classical bits** may be communicated

The framework explores when and how **shared randomness and quantum-inspired correlations** allow agents to outperform purely classical strategies — *without requiring quantum hardware*.

---

## The Core Challenge

At the heart of QSeaBattle is a simple but deep problem:

> **How much coordination is possible with limited communication?**

In each game:

- Player A observes an \( n \times n \) binary battlefield
- Player A sends only \( m \ll n^2 \) bits to Player B
- Player B sees a gun position and must decide whether to shoot
- Success means guessing the correct value at that position

This setup connects directly to foundational questions in:
- Information theory
- Quantum correlations
- Communication complexity
- Multi-agent learning

---

## Why QSeaBattle?

QSeaBattle sits at the intersection of **theory, simulation, and learning**.

### 🔬 Research
- Test information-theoretic limits such as *Information Causality*
- Study quantum-like advantages using classical simulations
- Explore Bell-type correlations and CHSH-style behaviors

### 🤖 Multi-Agent Learning
- Emergent communication under bandwidth constraints
- Imitation learning from optimal classical strategies
- Reinforcement learning with discrete communication
- Differentiable inter-agent communication (DIAL / DRU)

### 🧠 Neural + Symbolic Methods
- Combine known algorithms with trainable neural components
- Inject inductive bias via structured architectures
- Study when learning can rediscover known quantum-inspired protocols

---

## Architecture Overview

QSeaBattle is organized as a **progressive stack**, from simple baselines to advanced trainable agents.

1. **Core Game Infrastructure**  
   Game environment, evaluation, reproducibility, and logging

2. **Deterministic Baselines**  
   Hand-crafted classical strategies for verification and comparison

3. **Classical Neural Players**  
   Trainable agents without shared randomness

4. **Assisted (Quantum-Inspired) Players**  
   Classical simulation of quantum correlations using shared randomness

5. **Trainable Assisted Players**  
   Neural architectures combining learning with quantum-inspired structure

---

## What Makes QSeaBattle Different?

- No quantum hardware required  
- Exact reproducibility  
- Clean separation of theory and learning  
- Designed for experimentation and analysis  

QSeaBattle is built to support ablation studies, architecture comparisons, curriculum learning, and theoretical benchmarking.

---

## Getting Started

If you’re new, start here:

- **Quick Start** – run your first game
- **Game Mechanics** – understand the rules and parameters
- **Deterministic Players** – classical baselines
- **Neural Players** – learning under constraints
- **Assisted Players** – quantum-inspired coordination
- **Training Guides** – imitation and reinforcement learning
- **Theory** – information-theoretic foundations

Use the navigation on the left to dive in.

---

## Who Is This For?

- Researchers in **quantum information**
- ML practitioners working on **multi-agent systems**
- Students exploring **communication complexity**
- Anyone curious about the boundary between **classical and quantum coordination**

---

## Design Philosophy

- **Modular** — components are independent and composable  
- **Transparent** — no hidden state or magic  
- **Progressive** — complexity increases in layers  
- **Research-ready** — logging, reproducibility, and analysis built in  

---

## Ready to Explore?

Start with the one of the tutorials:

- [Tutorial 1 – Quick Start Guide](html/Tutorial_QSeaBattle_QuickStartGuide.html)
- [Tutorial 2 – Imitation training Neural Net Players](html/Tutorial_imitation_training_neural_net_models.html)
- [Tutorial 3 – Self learning Neural Net Players](html/Tutorial_DIAL_DRU_training_neural_net_models.html)
- [Tutorial 4 – Trainable model with (post)quantum resources](html/Tutorial_LinTrainableAssisted_Imitation.html)
- [Tutorial 5 – Alternative trainable model with (post)quantum resources](html/Tutorial_PyrTrainableAssisted_Imitation.html)

QSeaBattle is designed to reward curiosity — whether theoretical or practical.
