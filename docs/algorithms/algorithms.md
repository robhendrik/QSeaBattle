# QSeaBattle Algorithms

## Overview

QSeaBattle is a collaborative quantum game involving two players (Player A and Player B) who work together to make optimal decisions on an $n \times n$ battlefield. Player A observes the field and sends a limited communication message to Player B, who must decide whether to shoot at a specific location.

-----

## 1. Simple Player Algorithm

The Simple Player algorithm represents the most basic deterministic strategy for QSeaBattle.

### Simple Player A

**Input:** Field array (flattened $n^2$ cells with values in ${0, 1}$)

**Output:** Communication vector of length $m$ (where $m =$ `comms_size`)

**Algorithm:**

1. Flatten the $n \times n$ field into a 1D array of length $n^2$
2. Extract the values of the **first** $m$ cells
3. Return these $m$ values as the communication vector

**Pseudocode:**

```
function SimplePlayerA.decide(field):
    comm = field[0:m]  // Extract first m cells
    return comm
```

### Simple Player B

**Input:**

- Gun array (one-hot encoded, length $n^2$)
- Communication vector (length $m$)

**Output:** Shoot decision (0 or 1)

**Algorithm:**

Determine the gun index (position where gun = 1)
**If** the gun points at one of the first $m$ cells:
- Return the corresponding communication bit from that position
**Else** (gun points beyond the first $m$ cells):
- Ignore communication
- Shoot with probability `enemy_probability`, don’t shoot with probability $1 -$ `enemy_probability`

**Pseudocode:**

```
function SimplePlayerB.decide(gun, comm):
    gun_index = argmax(gun)
    
    if gun_index < m:
        return comm[gun_index]
    else:
        return sample_bernoulli(enemy_probability)
```

**Performance:** The Simple Player achieves better-than-random performance on the first $m$ cells but reverts to random guessing elsewhere.

-----

## 2. Majority Player Algorithm

The Majority Player algorithm improves upon the Simple Player by encoding aggregate information about field segments.

### Setup

The field is partitioned into $m$ **equal-sized segments**:

- Field size: $n^2$ cells
- Segment size: $L = n^2 / m$
- Requirement: $m$ must divide $n^2$

### Majority Player A

**Input:** Field array (length $n^2$)

**Output:** Communication vector (length $m$)

**Algorithm:**

1. Divide the flattened field into $m$ contiguous segments of size $L$
2. For each segment $j$ (where $j = 0, 1, \ldots, m-1$):
- Count the number of ones: $k_j = \sum_{i \in \text{segment}_j} \text{field}[i]$
- Set $\text{comm}[j] = 1$ if $k_j \geq L/2$ (majority are ones)
- Set $\text{comm}[j] = 0$ otherwise

**Pseudocode:**

```
function MajorityPlayerA.decide(field):
    L = n^2 / m
    comm = empty array of length m
    
    for j from 0 to m-1:
        segment = field[j*L : (j+1)*L]
        count_ones = sum(segment)
        
        if count_ones >= L/2:
            comm[j] = 1
        else:
            comm[j] = 0
    
    return comm
```

### Majority Player B

**Input:**

- Gun array (one-hot, length $n^2$)
- Communication vector (length $m$)

**Output:** Shoot decision (0 or 1)

**Algorithm:**

1. Determine the gun index
2. Identify which segment contains the gun: $\text{segment}_{\text{gun}} = \lfloor \text{gun index} / L \rfloor$
3. Return the majority bit for that segment: $\text{comm}[\text{segment}_{\text{gun}}]$

**Pseudocode:**

```
function MajorityPlayerB.decide(gun, comm):
    gun_index = argmax(gun)
    L = n^2 / m
    segment_index = floor(gun_index / L)
    
    return comm[segment_index]
```

**Performance:** The Majority Player provides more informative communication than the Simple Player, as each bit encodes statistical information about $L$ cells rather than a single cell.

-----

## 3. Linear Trainable Assisted Algorithm

The Linear Trainable Assisted algorithm uses neural networks combined with shared randomness to achieve quantum-like coordination.

### Architecture Overview

The algorithm uses a **single shared randomness resource** per decision and consists of four trainable components:

1. **LinMeasurementLayerA**: Maps field to measurement choices
2. **LinMeasurementLayerB**: Maps gun to measurement choices
3. **LinCombineLayerA**: Compresses outcomes to communication
4. **LinCombineLayerB**: Combines outcomes and communication to shoot decision

### Key Concept: Shared Randomness

A shared randomness layer produces **correlated outcomes** for Player A and Player B based on their measurement choices. The correlation is controlled by parameter $p_{\text{high}} \in [0, 1]$.

**Correlation Rule:**

Let $m_A$ and $m_B$ be measurements in ${0, 1}$, and $o_A$ be A’s outcome.

For B’s outcome $o_B$:

- If $(m_A, m_B) \in {(0,0), (0,1), (1,0)}$: $o_B = o_A$ with probability $p_{\text{high}}$
- If $(m_A, m_B) = (1,1)$: $o_B = o_A$ with probability $1 - p_{\text{high}}$

### Training Target (Imitation Learning)

The model is trained to reproduce a specific parity-based strategy:

#### Model A Training

**Step 1: Measurement Target**

- For each cell $i$: $\text{meas target}[i] = \text{field}[i]$
- Interpretation: Measure “type 1” where field = 1, “type 0” where field = 0

**Step 2: Shared Randomness**

- Apply shared randomness as **first measurement**
- Obtain outcomes $o_A$ of length $n^2$

**Step 3: Communication Target**

- Compute parity: $p_A = \bigoplus_{i=0}^{n^2-1} o_A[i]$ (XOR of all outcomes)
- Communication target: $\text{comm target} = p_A$ (for $m=1$)

**Loss:** Binary cross-entropy between predicted and target at each stage

#### Model B Training

**Step 1: Measurement Target**

- For each cell $i$: $\text{meas target}[i] = \text{gun}[i]$
- Interpretation: Measure “type 1” only at gun position

**Step 2: Shared Randomness**

- Apply shared randomness as **second measurement**
- Use A’s previous measurement and outcome
- Obtain outcomes $o_B$ of length $n^2$

**Step 3: Shoot Target**

- Compute B’s parity: $p_B = \bigoplus_{i=0}^{n^2-1} o_B[i]$
- Compute communication parity: $p_C = \text{comm}$ (for $m=1$)
- Shoot target: $\text{shoot_target} = p_B \oplus p_C$

**Loss:** Binary cross-entropy at each stage

### Algorithm Flow

#### Player A Decision Process

**Input:** Field array (length $n^2$)

**Output:** Communication bits (length $m$)

**Steps:**

1. **Measure:** $\text{meas probs}_A = \text{LinMeasurementLayerA}(\text{field})$
2. **Binarize:** $\text{meas bits}_A = (\text{meas probs}_A \geq 0.5)$
3. **Shared Randomness (First):**
- $\text{outcomes}_A = \text{SharedRandomnessLayer}(\text{meas bits}_A, \text{first}=\text{True})$
- Store $\text{meas bits}_A$ and $\text{outcomes}_A$ for Player B
1. **Combine:** $\text{comm logits} = \text{LinCombineLayerA}(\text{outcomes}_A)$
2. **Return:** $\text{comm} = (\sigma(\text{comm logits}) \geq 0.5)$

#### Player B Decision Process

**Input:**

- Gun array (length $n^2$)
- Communication (length $m$)
- Previous measurements from A
- Previous outcomes from A

**Output:** Shoot decision (0 or 1)

**Steps:**

1. **Measure:** $\text{meas probs}_B = \text{LinMeasurementLayerB}(\text{gun})$
2. **Binarize:** $\text{meas bits}_B = (\text{meas probs}_B \geq 0.5)$
3. **Shared Randomness (Second):**
- $\text{outcomes}_B = \text{SharedRandomnessLayer}(\text{meas bits}_B, \text{prev meas}_A, \text{prev out}_A)$
1. **Combine:** $\text{shoot logit} = \text{LinCombineLayerB}(\text{outcomes}_B, \text{comm})$
2. **Return:** $\text{shoot} = (\sigma(\text{shoot logit}) \geq 0.5)$

### Pseudocode

```python
# Model A Forward Pass
function LinTrainableAssistedModelA(field):
    # Step 1: Measurement
    meas_probs = LinMeasurementLayerA(field)
    meas_bits = (meas_probs >= 0.5).astype(int)
    
    # Step 2: Shared Randomness (First Measurement)
    outcomes = SharedRandomnessLayer(
        current_measurement=meas_bits,
        first_measurement=True
    )
    
    # Step 3: Combine to Communication
    comm_logits = LinCombineLayerA(outcomes)
    
    # Store for Player B
    store(meas_bits, outcomes)
    
    return comm_logits, meas_bits, outcomes

# Model B Forward Pass
function LinTrainableAssistedModelB(gun, comm, prev_meas, prev_out):
    # Step 1: Measurement
    meas_probs = LinMeasurementLayerB(gun)
    meas_bits = (meas_probs >= 0.5).astype(int)
    
    # Step 2: Shared Randomness (Second Measurement)
    outcomes = SharedRandomnessLayer(
        current_measurement=meas_bits,
        previous_measurement=prev_meas,
        previous_outcome=prev_out,
        first_measurement=False
    )
    
    # Step 3: Combine with Communication
    shoot_logit = LinCombineLayerB(outcomes, comm)
    
    return shoot_logit
```

**Performance:** When trained correctly, this algorithm can exceed classical information-theoretic bounds through quantum-like correlations.

-----

## 4. Pyramid Trainable Assisted Algorithm

The Pyramid algorithm extends the Linear approach by using **multiple layers** of shared randomness in a hierarchical structure.

### Architecture Overview

The Pyramid algorithm processes the field through $K = \log_2(n^2)$ iterations, halving the active length at each level.

**Key Properties:**

- Field size must be a power of 2: $n^2 = 2^K$
- Each iteration uses a separate shared randomness resource
- State is progressively compressed from $n^2$ down to 1

### Iteration Structure

At pyramid level $\ell$ (where $\ell = 0, 1, \ldots, K-1$):

- **Active length:** $L_\ell = n^2 / 2^\ell$
- **Measurement output length:** $L_\ell / 2$
- **Next state length:** $L_\ell / 2$

### Training Target (Imitation Learning)

#### Model A - Iteration $\ell$

**Input State:** Current field state of length $L_\ell$

**Step 1: Measurement**

- Process pairs $(x_{2i}, x_{2i+1})$ for $i = 0, \ldots, L_\ell/2 - 1$
- Measurement: $m_i^{(A)} = x_{2i} \oplus x_{2i+1}$
- Target: measure 1 if pair differs, 0 if pair matches

**Step 2: Shared Randomness**

- First measurement on resource $\ell$
- Obtain outcomes $s_i$ for $i = 0, \ldots, L_\ell/2 - 1$

**Step 3: Next State**

- For each pair index $i$:
  - $x_i’ = x_{2i} \oplus s_i$
- New state length: $L_\ell / 2$

**Final Output (after all iterations):**

- Final state has length 1
- Communication bit: $\text{comm} = x_0’$ (the final remaining bit)

#### Model B - Iteration $\ell$

**Input State:**

- Current gun state $g$ of length $L_\ell$ (one-hot)
- Communication bit $c$

**Step 1: Measurement**

- Process gun pairs $(g_{2i}, g_{2i+1})$
- Measurement: $m_i^{(B)} = \neg g_{2i} \land g_{2i+1}$
- Target: measure 1 only if pair is $(0,1)$

**Step 2: Shared Randomness**

- Second measurement on resource $\ell$
- Use A’s measurement and outcome from same level
- Obtain outcomes $s_i$

**Step 3: Next State**

- New gun state: $g_i’ = g_{2i} \oplus g_{2i+1}$ (preserves one-hot property)
- If gun was at pair $(0,1)$ or $(1,0)$, outcome affects communication:
  - Let $j$ be the index where $g_j’ = 1$
  - Update communication: $c’ = c \oplus s_j$

**Final Output (after all iterations):**

- Shoot decision: $\text{shoot} = c’$ (the final communication bit)

### Algorithm Flow

#### Player A Full Process

**Input:** Field (length $n^2$)

**Output:** Communication bit (for $m=1$)

```python
function PyramidPlayerA(field):
    state = field
    measurements_list = []
    outcomes_list = []
    
    for level from 0 to K-1:
        # Pairwise XOR measurement
        L = length(state)
        meas = empty array of length L/2
        
        for i from 0 to L/2 - 1:
            meas[i] = state[2*i] XOR state[2*i + 1]
        
        # Shared randomness (first measurement)
        outcomes = SharedRandomness_level(meas, first=True)
        
        # Store for Player B
        measurements_list.append(meas)
        outcomes_list.append(outcomes)
        
        # Compute next state
        next_state = empty array of length L/2
        for i from 0 to L/2 - 1:
            next_state[i] = state[2*i] XOR outcomes[i]
        
        state = next_state
    
    # Final state has length 1
    comm = state[0]
    
    return comm, measurements_list, outcomes_list
```

#### Player B Full Process

**Input:**

- Gun (one-hot, length $n^2$)
- Communication bit
- Measurements list from A (length $K$)
- Outcomes list from A (length $K$)

**Output:** Shoot decision

```python
function PyramidPlayerB(gun, comm, meas_list_A, out_list_A):
    gun_state = gun
    comm_state = comm
    
    for level from 0 to K-1:
        # Pairwise measurement on gun
        L = length(gun_state)
        meas = empty array of length L/2
        active_pair_index = None
        
        for i from 0 to L/2 - 1:
            pair = (gun_state[2*i], gun_state[2*i + 1])
            
            if pair == (0, 1):
                meas[i] = 1
                active_pair_index = i
            else:
                meas[i] = 0
        
        # Shared randomness (second measurement)
        outcomes = SharedRandomness_level(
            meas,
            meas_list_A[level],
            out_list_A[level],
            first=False
        )
        
        # Update gun state
        next_gun = empty array of length L/2
        for i from 0 to L/2 - 1:
            next_gun[i] = gun_state[2*i] XOR gun_state[2*i + 1]
        
        # Update communication if gun was active
        if active_pair_index is not None:
            comm_state = comm_state XOR outcomes[active_pair_index]
        
        gun_state = next_gun
    
    # Final decision
    shoot = comm_state
    
    return shoot
```

### Pyramid Example (Field Size 4)

For a field of size $2 \times 2 = 4$ cells, $K = 2$ iterations:

**Level 0:**

- Input: 4 cells $[x_0, x_1, x_2, x_3]$
- A measures: $[m_0 = x_0 \oplus x_1, m_1 = x_2 \oplus x_3]$
- Outcomes: $[s_0, s_1]$
- Next state: $[x_0 \oplus s_0, x_2 \oplus s_1]$ (length 2)

**Level 1:**

- Input: 2 cells from level 0
- A measures: $[m_0 = \text{state}[0] \oplus \text{state}[1]]$
- Outcomes: $[s_0]$
- Next state: $[\text{state}[0] \oplus s_0]$ (length 1)
- Communication: $\text{comm} = \text{state}[0]$

**Performance:** The Pyramid algorithm provides more fine-grained control and can potentially achieve higher success rates than the Linear approach for larger fields.

-----

## Comparison Summary

|Algorithm       |Communication Strategy          |Trainable|Uses Shared Randomness       |Complexity     |
|----------------|--------------------------------|---------|-----------------------------|---------------|
|Simple          |First $m$ cells directly        |No       |No                           |O(1)           |
|Majority        |Segment-wise majority vote      |No       |No                           |O($n^2$)       |
|Linear Assisted |Single-layer parity with SR     |Yes      |Yes (1 resource)             |O($n^2$)       |
|Pyramid Assisted|Multi-layer hierarchical with SR|Yes      |Yes ($\log_2(n^2)$ resources)|O($n^2 \log n$)|

**Key Insights:**

1. **Simple Player:** Minimal strategy, provides baseline performance
2. **Majority Player:** Better information encoding without learning
3. **Linear Assisted:** Quantum-inspired coordination, single shared resource
4. **Pyramid Assisted:** Hierarchical quantum-inspired approach, multiple resources

All algorithms must respect the communication constraint: only $m$ classical bits can be transmitted from Player A to Player B.