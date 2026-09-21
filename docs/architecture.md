# Architecture

QSeaBattle separates the **rules of the game**, the **strategy used by each player**, and the **machinery used to run repeated experiments**.

That separation is important because the project contains very different kinds of players:

- fixed classical strategies;
- PR-assisted strategies;
- neural-network players;
- trainable assisted models.

All of them should still be able to participate in the same game and the same tournament framework.

This page gives the big picture.

## The Big Picture

At the highest level, QSeaBattle has four main pieces:

1. **GameEnv** defines the rules and state of a single game.
2. **Players** decide how Player A and Player B act.
3. **Game** connects the environment and the players for one complete interaction.
4. **Tournament** repeats games and collects results.

A simplified view is:

```text
                ┌───────────────┐
                │    GameEnv    │
                │ rules + state │
                └───────┬───────┘
                        │
                ┌───────▼───────┐
                │      Game     │
                └──────┬─┬──────┘
                       │ │
              ┌────────┘ └────────┐
              │                   │
        ┌─────▼─────┐       ┌────▼──────┐
        │  Player A │       │  Player B │
        └───────────┘       └───────────┘

                Tournament
                    │
                    └── repeats Game
```

The key design idea is that the game framework should not need to know the internal details of a strategy.

A Simple player, a Majority player, and a neural-network player can therefore all be used through the same high-level gameplay flow.

## GameEnv: The Rules Of One Game

`GameEnv` represents the environment in which a single game is played.

Conceptually, it is responsible for things such as:

- generating or holding the battlefield;
- selecting the queried cell;
- exposing the correct observation to each player;
- enforcing the communication constraint;
- checking whether Player B's final answer is correct.

Player A sees the field.

Player B sees the query, represented by the gun position.

Player B does **not** see the field directly.

This separation is the central information constraint of the game.

The environment therefore defines what information is available, while the players decide what to do with that information.

## Players: The Strategy

The player classes contain the strategy.

Player A and Player B have different roles.

### Player A

Player A receives the field and produces the communication message.

Depending on the player family, this may involve:

- copying selected field bits;
- computing majority values;
- interacting with a shared resource;
- evaluating a neural network;
- executing one side of a trainable assisted model.

### Player B

Player B receives the query and the communication from Player A, then produces the final one-bit answer.

Again, the internal mechanism depends on the player family.

The important architectural constraint is always the same:

> Player B must never obtain the field through a hidden implementation path.

This means that even trainable players must respect the same information structure as hand-written strategies.

## Game: Bringing Environment And Players Together

`Game` connects a `GameEnv` with Player A and Player B.

A single game can be thought of as the following sequence:

1. the environment prepares a field and a query;
2. Player A receives the field;
3. Player A produces a communication message;
4. Player B receives the query and the communication;
5. Player B produces the final answer;
6. the environment evaluates whether that answer matches the queried field value.

This is deliberately a small interface.

The game object does not need to know whether the communication was produced by:

- a fixed rule;
- a majority calculation;
- a PR-assisted construction;
- a neural network.

That responsibility stays inside the player implementation.

## Tournament: Repeating The Experiment

A single game is usually not enough to evaluate a strategy.

`Tournament` repeatedly runs games and aggregates the results.

This is where different strategies can be compared statistically.

Typical tournament-level quantities include:

- number of games played;
- number of wins;
- win rate;
- performance under different game settings;
- performance under different resource assumptions.

This separation is useful because a player can be tested in exactly the same tournament machinery regardless of how sophisticated its internal strategy is.

## Classical Players And Neural Players

The game itself is naturally discrete.

Fields are made of bits.

Communication messages are bits.

The final answer is a bit.

Neural networks, however, do not naturally operate in exactly the same representation.

They usually produce continuous values such as **logits**.

That creates an important interface problem:

> How do we let a neural model participate in a game whose public interface is expressed in discrete bits?

QSeaBattle handles this by separating the gameplay representation from the model representation.

## Bits, Probabilities And Logits

A bit is simply:

```text
0 or 1
```

A probability represents uncertainty:

```text
0.0 ≤ p ≤ 1.0
```

A logit is an unrestricted real-valued number:

```text
-∞ < logit < +∞
```

For a binary decision, a logit can be converted to a probability using a sigmoid:

```text
probability = sigmoid(logit)
```

A large positive logit corresponds to a probability close to 1.

A large negative logit corresponds to a probability close to 0.

A logit near zero corresponds to a probability near 0.5.

During training, logits are useful because they provide a smooth signal for optimization.

During actual gameplay, however, the environment eventually needs a discrete decision.

## Gameplay Adapters: The Bridge Between Models And Games

The gameplay adapter is the bridge between those two worlds.

Conceptually:

```text
Neural model
    │
    │ logits / continuous outputs
    ▼
Gameplay adapter
    │
    │ game-compatible bits or decisions
    ▼
Game / GameEnv
```

The adapter allows neural-network players to participate in the same gameplay interface as fixed-rule players.

This is useful because it keeps responsibilities separate:

- the **model** can focus on learning;
- the **adapter** handles representation conversion;
- the **game** continues to operate on its normal interface.

The same idea works in the opposite direction as well: game observations can be converted into the tensor representation expected by a model.

This avoids spreading TensorFlow-specific or training-specific logic throughout the gameplay code.

## Why Keep Bits And Logits Separate?

It can be tempting to let every layer of the program work directly with floating-point values.

That would make some training code simpler, but it would blur an important distinction.

The game itself has discrete semantics.

A communication bit is not merely "a number close to zero or one"; it is the information actually sent from Player A to Player B.

Keeping the boundary explicit makes it easier to reason about:

- information flow;
- communication limits;
- reproducibility;
- testing;
- whether a model is accidentally using information it should not have.

This is especially important for assisted and trainable-assisted strategies.

## Shared Resources

Some player families use a shared resource in addition to the normal communication channel.

The shared resource is conceptually separate from both the players and the environment.

It provides correlated outcomes to Player A and Player B, but it must not create an ordinary signaling path from the field to Player B.

The players decide how to use those correlations.

The game still enforces the same visible communication constraint.

This keeps assisted strategies compatible with the same game and tournament framework as classical strategies.

## Trainable Assisted Models

The trainable Linear and Pyramid families add another layer.

Their internal computation may be learned, but their architecture is constrained so that the original game structure remains intact.

This means the model is not simply given all inputs and asked to predict the answer.

Instead:

- Player A-side components receive only Player A-side information;
- Player B-side components receive only Player B-side information;
- communication passes through the allowed message;
- shared resources enter only through the permitted interfaces.

This makes the architecture itself part of the experiment.

The model is free to learn *how* to use the allowed information, but not to bypass the rules of the game.

## Training Versus Gameplay

Training and gameplay are related, but they are not the same process.

During training:

- models operate on batches;
- outputs may remain as logits;
- differentiable approximations may be used;
- losses are computed;
- weights are updated.

During gameplay:

- one concrete field and query are evaluated;
- communication has game-level meaning;
- the final output must be interpretable as a game decision;
- no optimizer is involved.

The adapter layer and the player interfaces help keep these two contexts compatible without merging them into one abstraction.

## How The Main Pieces Relate

A useful mental model is:

```text
                 ┌───────────────────────┐
                 │      Tournament       │
                 │ repeats + aggregates  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │         Game          │
                 │ orchestrates one run  │
                 └───────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      ┌───────────────┐             ┌───────────────┐
      │    Player A   │             │    Player B   │
      └───────┬───────┘             └───────┬───────┘
              │                             │
              │       communication         │
              └──────────────►──────────────┘
              │                             │
              ▼                             ▼
      fixed logic / model           fixed logic / model
              │                             │
              └──────────┬──────────────────┘
                         ▼
                  gameplay adapters
                         │
                         ▼
                    GameEnv
                 rules + evaluation
```

Not every player family uses every internal component, but the high-level contract remains consistent.

## Why This Structure Matters

The architecture is designed to make three things possible at once.

### Comparability

Different strategies can be evaluated under the same game rules.

### Extensibility

New player families can be added without redesigning the environment or tournament framework.

### Safety Of Information Flow

The software structure helps preserve the distinction between what Player A knows, what Player B knows, and what may be communicated between them.

That is essential when comparing classical, assisted, and learned strategies.

## Where To Go Next

If you are new to the project, the following order is useful:

1. read **Built-In Algorithms** for the strategy intuition;
2. try the **Quick Start tutorial** to run players in a tournament;
3. inspect the **Players** documentation for concrete implementations;
4. move to the neural-network tutorials;
5. explore the **Linear** and **Pyramidal** model documentation for assisted trainable strategies.

The generated API pages contain the implementation-level detail, while this page is intended to explain how the main pieces fit together.
