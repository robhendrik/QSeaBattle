# **Simple (Subset-Revealing) Algorithm in Classical Random Access Codes (RACs)**

## Overview

In a **classical $n \to m$ random access code (RAC)**, Alice holds an $n$-bit string and is allowed to send an $m$-bit classical message to Bob. Bob is then asked to output the value of *any one* of Alice’s input bits, specified by a query index, and the performance of the protocol is measured by the **average success probability** over uniformly random inputs and queries.

Besides optimal strategies such as **majority encoding**, the RAC literature also considers **simple deterministic baseline strategies**. One of the most basic of these is the **subset-revealing (or identity-based) algorithm**, in which Alice directly communicates a fixed subset of her input bits, and Bob outputs the communicated bit whenever possible, otherwise guessing according to a fixed prior.

This algorithm is **strictly suboptimal** compared to majority encoding, but it is widely used in the literature as a **classical baseline** and as a reference point when proving optimality results or demonstrating quantum advantages.

---

## The Simple Algorithm – Encoding & Decoding

**Algorithm (Subset-revealing encoding, identity decoding):**

*Alice selects a fixed subset of $m$ positions from her $n$-bit input string and sends the corresponding $m$ bits directly to Bob.*  
*Bob, upon receiving the message and a query index $j$, outputs the communicated bit if $j$ lies in the revealed subset; otherwise, Bob outputs a default value or guesses according to a fixed prior probability.*

This strategy can be summarized as:

- **Encoding:** transmit raw input bits from predetermined positions
- **Decoding:** direct lookup if available, otherwise a fixed probabilistic guess

---

### Example (Binary case)

Let Alice’s input be a 5-bit string  
`x = 1 0 1 1 0`
Suppose Alice is allowed to send $m = 1$ bit and always transmits the **first bit** of her string. She sends:
`comm = 1`

- If Bob is asked for index 0, he outputs `1` and is correct.
- If Bob is asked for any other index, he has no information and must guess (e.g. with probability 1/2).

The average success probability of this strategy is therefore limited by the probability that the queried index lies in the revealed subset.

---

## Interpretation as a Classical RAC Baseline

This strategy corresponds to what is often referred to in the literature as a **trivial or identity-based classical RAC**:

> *Alice sends one of her input bits; Bob outputs that bit if it is queried, and otherwise guesses.*

Such strategies are conceptually simple and easy to analyze, making them natural **baseline protocols** when comparing different RAC constructions.

Importantly, the performance of this algorithm scales linearly with the fraction of revealed bits:
$$
P_{\text{succ}} = \frac{m}{n} + \left(1 - \frac{m}{n}\right) p_{\text{guess}},
$$
where $p_{\text{guess}}$ is the success probability of Bob’s default guess.

---

## Relation to Optimality Results in the Literature

### Comparison with Majority Encoding

In their study of classical and quantum RACs, **Tavakoli *et al.* (2015)** explicitly contrast *simple classical strategies* with majority encoding. While their focus is on identifying optimal strategies, they describe majority encoding as outperforming strategies in which Alice merely outputs part of her input:

> “Intuition strongly suggests that an optimal strategy is for Alice to use majority encoding…”
> — *Phys. Rev. Lett. 114, 170502 (2015)*

This statement appears in the context of comparing majority encoding against **simpler classical encodings**, such as sending individual symbols or fixed components of the input. These simpler encodings correspond precisely to subset-revealing strategies like the one described here.

---

### Classification in the Optimality Proof

The rigorous proof of optimality by **Ambainis, Kravchenko & Rai (2015)** analyzes the space of **all deterministic classical RAC strategies**. In doing so, the authors explicitly consider strategies in which Alice’s message is **directly identified with one of the input symbols** or otherwise depends on only a limited part of the input.

Within this classification framework, strategies that merely reveal a subset of bits are shown to be valid classical RACs but **provably suboptimal**:

- They achieve a success probability bounded by the fraction of information directly revealed.
- They are strictly dominated by majority-based strategies under uniform inputs.

This analysis appears in the sections where deterministic encodings are enumerated and compared before establishing the optimality of majority encoding.

---

## Conditions and Limitations

- The simple subset-revealing algorithm **does not exploit correlations** between input bits.
- Its performance depends directly on how often the queried index lies in the revealed subset.
- Under the standard RAC assumptions (uniform inputs and uniform query indices), this strategy **cannot achieve the maximal average success probability**.

As a result, it serves primarily as:
- a **didactic example**,
- a **baseline classical protocol**, or
- a **lower bound** when demonstrating the advantage of more sophisticated classical or quantum strategies.

---

## QSeaBattle Implementation

In QSeaBattle, this algorithm is implemented as a deterministic strategy in which:

- Alice transmits the first `m` bits of the flattened field.
- Bob outputs the communicated bit if the gun index lies within those `m` positions.
- Otherwise, Bob decides according to a fixed enemy probability.

This implementation directly realizes the subset-revealing classical RAC described above and is used as a **simple reference strategy** against which more advanced algorithms (such as majority encoding or trainable models) can be compared.

---

## Key References

* **Tavakoli *et al.***,  
  *“Quantum Random Access Codes using Single d-level Systems,”*  
  *Phys. Rev. Lett. 114, 170502 (2015).*  
  — Introduces majority encoding and contrasts it with simpler classical strategies such as sending individual input symbols. The discussion motivating majority encoding implicitly treats subset-revealing strategies as natural but suboptimal baselines.

* **Ambainis, Kravchenko & Rai**,  
  *“Quantum Advantages in (n, d)→1 Random Access Codes,”*  
  *arXiv:1510.03045 (2015).*  
  — Provides a complete characterization of optimal classical RAC strategies. Deterministic strategies that reveal only part of the input are included in the analysis and shown to be strictly suboptimal compared to majority-based encodings.
