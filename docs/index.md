# Introduction to QSeaBattle

## What is QSeaBattle?

QSeaBattle is a collaborative quantum game framework that explores the boundaries between classical and quantum information processing. It provides a testbed for investigating how quantum-inspired correlations can enable coordination between agents with limited classical communication.

### The Core Challenge

In QSeaBattle, two players work together to solve a coordination problem:

- **Player A** observes a binary battlefield (an $n \times n$ grid of 0s and 1s)
- **Player A** can send only $m$ classical bits to **Player B** (where $m$ is much smaller than $n^2$)
- **Player B** sees only a gun position and must decide whether to shoot
- Success means correctly guessing the value at the gun position

This setup mirrors fundamental questions in quantum information theory: How much can two parties coordinate when restricted to limited classical communication?

## Why QSeaBattle?

### Scientific Motivation

QSeaBattle was developed to bridge several areas of research:

**1. Quantum Information Theory**

- Explores concepts like shared randomness and quantum correlations
- Investigates information-theoretic limits (Information Causality)
- Provides concrete implementations of abstract quantum protocols

**2. Multi-Agent Reinforcement Learning**

- Studies emergent communication in cooperative settings
- Tests architectures like DIAL (Differentiable Inter-Agent Learning)
- Examines the role of shared resources in coordination

**3. Neural-Symbolic Integration**

- Combines algorithmic knowledge (classical quantum protocols) with learning
- Uses imitation learning to bootstrap neural agents from known strategies
- Investigates how neural networks can discover or approximate quantum advantages

### Educational Value

QSeaBattle serves as an accessible introduction to:

- Quantum game theory without requiring quantum hardware
- Information-theoretic constraints in communication
- The relationship between correlation, communication, and computation
- Modern deep learning architectures for multi-agent systems

## The Game Mechanics

### Basic Setup

A single game of QSeaBattle proceeds as follows:

**1. Field Generation**

A battlefield of size $n \cdot n$ is created, where each cell independently has probability $p$ of containing an enemy (value 1)

**2. Gun Placement**

A gun position is randomly selected from the $n^2$ possible locations

**3. Player A Decision**

- Observes the entire battlefield
- Generates an $m$-bit communication message
- This message passes through a noisy channel with bit-flip probability $c$

**4. Player B Decision**

- Sees only the gun position
- Receives the (possibly noisy) communication
- Decides to shoot (1) or not shoot (0)

**5. Evaluation**

The decision is correct if:

- Shoot when there is an enemy at that position, or
- Do not shoot when that position is empty

### Game Parameters

The game is configured through a GameLayout object with the following parameters:

|Parameter        |Symbol|Description                  |Typical Values  |
|-----------------|------|-----------------------------|----------------|
|field_size       |$n$   |Dimension of battlefield     |4, 8, 16, 32, 64|
|comms_size       |$m$   |Number of communication bits |1 to 16         |
|enemy_probability|$p$   |Probability of enemy per cell|0.5             |
|channel_noise    |$c$   |Bit-flip probability         |0.0 to 0.3      |

### Performance Metrics

Player performance is measured by:

- **Win Rate**: Proportion of correct decisions across many games
- **Comparison to Baselines**: Simple random guessing, majority voting, classical protocols
- **Information-Theoretic Bounds**: Limits imposed by Information Causality given $m$ and $c$

## Architecture Overview

QSeaBattle is organized into several layers, from basic game infrastructure to advanced trainable agents.

### Layer 1: Core Game Infrastructure

The foundation provides the essential game mechanics:

- **GameLayout**: Configuration parameters
- **GameEnv**: Battlefield generation and evaluation
- **Players**: Base class for player implementations
- **PlayerA/PlayerB**: Individual player interfaces
- **Game**: Single game execution
- **Tournament**: Multiple game execution and logging
- **TournamentLog**: Data collection and analysis

Purpose: Ensures consistent game execution, reproducibility, and data collection across all player types.

### Layer 2: Deterministic Players

Hand-crafted strategies that serve as baselines:

- **SimplePlayers**: Direct communication of first $m$ cells
- **MajorityPlayers**: Segment-wise majority voting strategy

Purpose: Provide interpretable baselines and verify game mechanics are working correctly.

### Layer 3: Classical Neural Players

Trainable agents without shared randomness:

**NeuralNetPlayers**: Two independent neural networks

- Model A: field to communication logits
- Model B: gun and communication to shoot logit

Features:

- Can be trained by imitation (supervised learning on baseline strategies)
- Can be trained by reinforcement learning (self-play with policy gradients)
- Provides comparison point for quantum-inspired approaches

Purpose: Establish what purely classical learning can achieve within information-theoretic constraints.

### Layer 4: Assisted Players (Classical Quantum Simulation)

Deterministic algorithms that simulate quantum correlations using shared randomness:

**AssistedPlayers**: Classical implementation of quantum protocols

- **SharedRandomness**: Simulates entangled measurements
- **AssistedPlayerA**: Hierarchical measurement strategy
- **AssistedPlayerB**: Complementary measurement strategy

Key Concept: These players share access to pre-established random bits that are correlated in a specific way (controlled by parameter $p_{\mathrm{high}}$). This correlation mimics quantum entanglement and enables coordination beyond classical limits.

Purpose: Demonstrate the target behavior that trainable quantum-inspired agents should learn.

### Layer 5: Trainable Assisted Players

Neural implementations that combine learning with quantum-inspired structure:

**TrainableAssistedPlayers**: Neural networks with shared randomness layers

**Linear Architecture:**

- LinMeasurementLayerA/B: Learn measurement choices
- SharedRandomnessLayer: Fixed correlation (differentiable)
- LinCombineLayerA/B: Learn how to use outcomes
- Single shared resource per decision

**Pyramid Architecture:**

- PyrMeasurementLayerA/B: Multi-level measurement
- Multiple SharedRandomnessLayers (one per level)
- PyrCombineLayerA/B: Hierarchical combination
- $\log(n^2)$ shared resources per decision

**Training Approaches:**

1. Imitation Learning: Train on synthetic datasets generated by classical assisted algorithms
2. Differentiable Communication: Use DRU (Discretize/Regularize Unit) for end-to-end gradient flow
3. Reinforcement Learning: Policy gradient methods with shared randomness

Purpose: Enable neural networks to discover and improve quantum-inspired strategies through learning.

### Supporting Utilities

Several utility modules provide consistent implementations across the framework:

- **logit_utils**: Stable logit/probability conversions
- **dru_utils**: Differentiable communication (DIAL)
- **reference_performance_utils**: Analytic baselines and bounds
- **imitation_utils**: Synthetic dataset generation

## Information Flow in QSeaBattle

The key architectural pattern is how information flows between players.

### Classical Neural Players

Flow diagram:

```text
Field → Model A → Communication → Model B → Shoot Decision
        ↓                           ↑
      [Loss A]                   [Loss B]
```

Characteristics:

- Independent training possible
- Communication is a bottleneck ($m$ bits)
- Limited by classical information theory

### Trainable Assisted Players

Flow diagram:

```text
Field → Measurement A → Shared Randomness ← Measurement B ← Gun
              ↓              ↓    ↓              ↓
          Outcome A      Correlation      Outcome B
              ↓                                   ↓
        Combine A → Communication → Combine B → Shoot
              ↓                                   ↓
           [Loss A]                          [Loss B]
```

Characteristics:

- Measurements are learned (neural layers)
- Shared randomness is fixed (simulates physics)
- Combine operations are learned (neural layers)
- Key advantage: Correlation is outside the $m$-bit channel
- Can exceed classical limits if trained correctly

## Data Flow and Reproducibility

QSeaBattle ensures full reproducibility through careful data management.

### Tournament Execution

Pseudocode for tournament flow:

```python
for each game:
    env.reset()          # New field and gun
    players.reset()      # Clear player state
    
    field, gun = env.provide()
    comm = playerA.decide(field)
    comm_noisy = env.apply_channel_noise(comm)
    shoot = playerB.decide(gun, comm_noisy)
    reward = env.evaluate(shoot)
    
    log.update(field, gun, comm, shoot, reward)
    
    if players.has_log_probs:
        log.update_log_probs(playerA.get_log_prob(), 
                            playerB.get_log_prob())
    
    if players.has_prev:
        log.update_log_prev(playerA.get_prev())
```

### Logged Data

The TournamentLog captures everything needed for analysis and training:

**Inputs**: field, gun, comm (post-noise)

**Outputs**: shoot, reward

**Metadata**: game_id, tournament_id, timestamps

**Training Data** (optional):

- logprob_comm, logprob_shoot for policy gradients
- prev_measurements, prev_outcomes for assisted players
- sample_weight for curriculum learning

### Reproducibility Guarantees

- All randomness derives from a single global seed
- Logged games can be exactly replayed
- Models can be serialized and restored
- Training datasets are deterministically generated

## Use Cases

### Research Applications

**1. Quantum Advantage Without Quantum Hardware**

- Investigate when quantum-inspired correlations provide benefits
- Test predictions from quantum information theory
- Explore the classical-quantum boundary

**2. Multi-Agent Learning**

- Study emergent communication protocols
- Compare different coordination mechanisms
- Investigate transfer learning from classical to quantum-inspired agents

**3. Neural Architecture Design**

- Test structured vs unstructured neural networks
- Evaluate importance of inductive biases
- Study gradient flow through discrete operations

### Educational Applications

**1. Introduction to Quantum Games**

- Visualize quantum concepts without mathematics
- Interactive experiments with correlations
- Hands-on exploration of information limits

**2. Deep Learning Pedagogy**

- End-to-end differentiable systems
- Imitation and reinforcement learning
- Custom Keras layers and training loops

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QSeaBattle.git
cd QSeaBattle

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Quick Example

```python
from Q_Sea_Battle import GameLayout, GameEnv, SimplePlayers, Tournament

# Configure the game
layout = GameLayout(
    field_size=8,        # 8x8 battlefield
    comms_size=1,        # 1 bit of communication
    enemy_probability=0.5,
    channel_noise=0.0
)

# Create environment and players
env = GameEnv(layout)
players = SimplePlayers(layout)

# Run a tournament
tournament = Tournament(env, players, layout)
log = tournament.tournament()

# Analyze results
mean_reward, std_error = log.outcome()
print(f"Win rate: {mean_reward:.3f} ± {std_error:.3f}")
```

### Next Steps

- **Tutorial 1**: Understanding the game mechanics with deterministic players
- **Tutorial 2**: Training neural players by imitation learning
- **Tutorial 3**: Implementing and training assisted players
- **Tutorial 4**: Reinforcement learning and emergent strategies
- **Tutorial 5**: Analyzing results and comparing to theoretical bounds

## Design Philosophy

QSeaBattle is built on several key principles.

### Modularity

Each component is self-contained and can be tested independently:

- Game environment independent of players
- Player implementations share common interfaces
- Training utilities separated from game logic
- Easy to add new player types or game variants

### Transparency

All behavior is explicit and inspectable:

- No hidden state in game mechanics
- Full logging of all decisions and outcomes
- Deterministic reproducibility from seeds
- Clear separation of learned vs fixed components

### Gradual Complexity

The framework supports learning progressively:

- Start with simple deterministic baselines
- Add learning with classical neural networks
- Introduce quantum-inspired structure gradually
- Each layer builds on previous understanding

### Research-Ready

Designed for experimentation and publication:

- Clean API for custom player implementations
- Comprehensive logging for analysis
- Serialization for model sharing
- Utilities for theoretical comparisons

## Theoretical Foundations

QSeaBattle is grounded in several theoretical frameworks.

### Information Causality

This principle from quantum foundations states that the amount of information Bob can learn about Alice’s $N$-bit string, given $m$ bits of communication, is bounded by $m$. QSeaBattle provides a testbed for this bound.

In our setting: With $N = n^2$ and communication $m$, classical strategies cannot exceed certain win rates. Quantum-inspired strategies can approach but not violate these bounds.

### CHSH Inequality

The game mechanics are related to Bell-type inequalities. The correlation parameter $p_{\mathrm{high}}$ in shared randomness can be tuned to reproduce behaviors from:

- Classical local hidden variables: $p_{\mathrm{high}} = 0.5$
- Quantum mechanics: $p_{\mathrm{high}} \approx 0.85$
- Post-quantum (signaling) theories: $p_{\mathrm{high}} = 1.0$

### Communication Complexity

The constraint of $m$ bits relates to classical communication complexity. QSeaBattle explores:

- How much coordination is possible with limited communication
- The value of shared randomness or entanglement
- Tradeoffs between computation, communication, and correlation

## Conclusion

QSeaBattle provides a rich environment for exploring the interplay between:

- Classical and quantum information processing
- Hand-crafted algorithms and learned strategies
- Communication, correlation, and computation
- Theory and implementation

Whether you are interested in quantum foundations, multi-agent learning, or neural architecture design, QSeaBattle offers concrete problems, clean implementations, and connections to deep theoretical questions.

The framework is designed to be accessible to newcomers while providing enough depth for advanced research. We hope it serves as a bridge between different communities and inspires new insights into the nature of information, correlation, and intelligence.

Ready to explore? Head to the Quick Start Guide or dive into the Tutorials.