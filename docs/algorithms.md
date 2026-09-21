# Built-In Algorithms

QSeaBattle includes several player strategies that all solve the same basic problem in different ways.

Player A sees the complete battlefield. Player B sees only the cell that is being queried. Player A may send a limited message to Player B, and Player B must use that message to guess the value of the requested cell.

The algorithms differ in how they compress information, how they use shared resources, and whether parts of the strategy are fixed or trainable.

## The Same Game, Different Strategies

At a high level, every strategy follows the same information flow:

1. Player A receives the field.
2. Player A computes a communication message.
3. Player B receives the queried position and the communication message.
4. Player B produces a one-bit answer.

The interesting part is how much useful information can be packed into the allowed communication.

The built-in algorithms can be viewed as a progression from simple classical baselines to more structured assisted strategies.

| Strategy | Main idea | Communication | Shared resource |
| --- | --- | --- | --- |
| Simple | Send selected field bits directly | Configurable | None |
| Majority | Summarize groups of cells by their majority value | Configurable | None |
| PR-assisted | Use non-signaling correlations to improve how information is combined | Limited | PR-style resource |
| Linear | Use assisted correlations in a parallel, linear-style construction | Configurable | PR-style resource |
| Pyramid | Recursively compress information through several levels | Typically one bit | PR-style resource |
| Trainable variants | Learn mappings while preserving the same information constraints | Depends on family | Depends on family |

## Simple Strategy

The Simple strategy is the most direct classical baseline.

Player A and Player B agree in advance on a fixed set of field positions. Player A sends the values of those selected positions directly.

If Player B is asked about one of the communicated positions, the answer is known exactly. If the queried position lies outside the covered set, Player B has no direct information about it and must fall back to a fixed guess.

This makes the trade-off easy to understand:

- more communication means more cells can be covered directly;
- uncovered cells remain essentially unknown;
- the strategy does not try to compress structure in the field.

The Simple player is useful as a reference because it shows what can be achieved by straightforward communication without any additional processing.

## Majority Strategy

The Majority strategy uses the same communication budget differently.

Instead of sending individual cells, Player A divides the flattened field into several segments. For each segment, Player A sends one bit indicating the majority value in that segment.

Player B determines which segment contains the queried position and uses the majority bit of that segment as the answer.

This sacrifices certainty about individual cells but spreads information over a larger part of the field.

The idea is simple:

- direct communication is accurate but narrow;
- majority communication is approximate but broad.

This makes Majority a useful classical compression baseline. It is often more effective than direct coverage when one communication bit must summarize several cells.

## Shared Resources

Some QSeaBattle strategies use an additional shared resource.

A shared resource gives Player A and Player B access to correlated outcomes that are established without allowing direct signaling between them. In QSeaBattle, PR-style resources are used as a model of stronger-than-classical correlations.

The important point is that the shared resource does **not** give Player B direct access to the field.

Player B can still depend on the field only through:

- the message sent by Player A;
- the agreed shared resource correlations.

The shared resource changes how efficiently the limited communication can be used, but it does not add an ordinary communication channel.

## PR-Assisted Strategies

PR-assisted players use the shared resource together with the ordinary communication channel.

The basic pattern is:

1. Player A computes one or more inputs to the shared resource from the field.
2. Player B computes corresponding inputs from the queried position.
3. The shared resource produces correlated outcomes.
4. Player A combines those outcomes with the field to form the message.
5. Player B combines the received message, the query, and the corresponding shared-resource outcomes to form the final answer.

The exact construction depends on the algorithm family.

The common goal is to exploit correlations that are stronger than what purely classical shared randomness can provide.

## Linear Strategy

The Linear family uses several assisted correlations in a relatively direct, parallel-style construction.

Conceptually, Player A computes measurements from the field, combines those measurements with shared-resource outcomes, and turns the result into the communication message.

Player B performs the matching operation on the query side and combines:

- the queried position;
- the received communication;
- the corresponding shared-resource outcomes.

This family is useful because it keeps the structure fairly transparent. The strategy is built from repeated measurement-and-combine operations rather than from a deep recursive reduction.

In the trainable version, the same information flow is preserved while some of the fixed mappings are replaced by neural-network components.

## Pyramid Strategy

The Pyramid family uses a different idea: recursive compression.

Instead of processing the whole field in one step, the strategy repeatedly reduces the active representation.

At each level:

1. neighbouring pieces of information are paired;
2. a shared-resource interaction is applied;
3. the active representation becomes smaller;
4. the next level repeats the process.

The number of active elements is halved at each stage until only the final information needed for the one-bit communication remains.

Player B performs a matching reduction on the query side.

This creates a tree-like or pyramidal information flow, which is where the strategy gets its name.

The Pyramid strategy is especially interesting because the final communication can remain very small even though the original field may contain many cells.

## Linear Versus Pyramid

The Linear and Pyramid families use the same basic ingredients but organize them differently.

The Linear construction is closer to a flat or parallel computation. The Pyramid construction is hierarchical.

A useful way to think about the difference is:

- **Linear:** combine many assisted pieces of information directly;
- **Pyramid:** repeatedly compress information through successive levels.

Neither description by itself implies that one family is always superior. They represent different structural choices for how information is processed before the final answer is produced.

## Trainable Assisted Models

QSeaBattle also contains trainable versions of the assisted strategies.

These models are not unrestricted neural networks. Their architecture is deliberately constrained so that the rules of the game remain intact.

In particular:

- Player A may use the field, but Player B may not;
- Player B may use the query and the received communication;
- shared resources may only enter through the allowed assisted interfaces;
- there is no hidden path that bypasses the communication constraint.

This makes it possible to train parts of a strategy while still preserving the information structure of the original game.

The trainable Linear and Pyramid models are therefore best understood as learned implementations of the same constrained information flow, rather than as generic end-to-end predictors.

## Expected And Sampled Shared Resources

The shared-resource implementation can be used in two broad ways.

### Expected Mode

The resource is replaced by its expected behaviour.

This is deterministic and is useful when a differentiable signal is needed during training.

### Sample Mode

Actual resource outcomes are sampled.

This is stochastic and more closely resembles repeated gameplay with explicit random outcomes.

Both modes represent the same underlying resource model, but they are useful for different purposes.

## When To Use Which Strategy

If you are exploring QSeaBattle for the first time, a useful progression is:

1. start with **Simple** to understand the communication constraint;
2. compare it with **Majority** to see the effect of classical compression;
3. move to **PR-assisted** players to introduce stronger correlations;
4. compare **Linear** and **Pyramid** constructions;
5. use the **trainable assisted** models to study whether the same strategy structure can be learned from data.

The tutorials on the project homepage follow this progression from fixed players to neural and assisted models.

## Technical Constraints

A few implementation constraints are worth keeping in mind.

- The field is represented as a flattened binary vector.
- The query is represented as a one-hot position.
- The communication size is configurable for most strategies.
- The Pyramid family typically uses a single communication bit.
- The Pyramid construction requires the number of field cells to support repeated halving through its levels.
- Player B never receives the field directly.

For exact tensor shapes, layer interfaces, and class-level details, see the generated documentation for the corresponding player and model classes.

## Related Documentation

For implementation details, see:

- **Players** for the concrete Player A and Player B classes;
- **Shared Resources** for the PR-assisted resource implementation;
- **Linear Models** for the trainable linear family;
- **Pyramidal Models** for the recursive trainable family;
- **Dataset Utilities** for the imitation-training data pipelines;
- **Tutorials** on the homepage for worked examples.
