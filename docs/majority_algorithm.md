# **Majority Algorithm in Classical Random Access Codes (RACs)**

## Overview

In a **classical $n \to 1$ random access code (RAC)**, Alice holds an $n$-bit string and wants Bob to retrieve *any one* of those $n$ bits with high probability, even though Alice is only allowed to send a single bit (a one-bit message) to Bob. The **“majority” algorithm** – a simple deterministic encoding/decoding strategy – has been conjectured and **proven to be an optimal classical strategy** for this task in terms of maximizing the average success probability. Below we summarize how this algorithm works and why it is optimal, along with conditions for its optimality and known limitations. We also indicate how this algorithm is implemented in QSeaBattle.


## The Majority Algorithm – Encoding & Decoding

**Algorithm (Majority-encoding, Identity-decoding):** *Alice counts the number of 0’s and 1’s in her $n$-bit string and sends a single bit indicating which value is in the majority*. (If there is a tie, she can choose either 0 or 1 consistently as a convention.) *Bob, upon receiving this one-bit message, simply interprets it as the bit value for **any** position he is asked to retrieve*. In other words, Bob outputs the **same bit he received** (this is “identity decoding”) as his guess for the requested index.

*   **Example:** If Alice’s string is `10110` (which has three `1`s and two `0`s), she will send “1” because 1 is the majority bit. Regardless of which index $j$ (1 through 5) Bob wants to know, Bob will answer “1”. In this example, Bob will be correct for any bit that was actually 1 (the three positions containing `1`), and wrong for any position that was 0. If we assume all input strings and query positions are uniformly random, Bob’s guess using the majority scheme is correct with probability $\frac{\text{count of majority bits in }x}{n}$. Here that’s $3/5 = 60%$ for this specific string. The **average** success probability over all 5-bit strings turns out to be about 68.75% for this strategy – and no other deterministic encoding can do better for 5 bits.

## Optimality and Theoretical Justification

For uniform random inputs and query positions (the standard RAC scenario measuring **average** success), the majority algorithm is provably optimal among all deterministic encodings. Intuitively, by sending the most frequent bit value, Alice maximizes the chance that a randomly chosen bit from her string matches the sent bit. Any other encoding would either send a minority value (reducing success on average) or encode some more complex function of the bits, which ends up being less correlated with a random individual bit than the majority vote is.

**Historical context:** Earlier works had already established the optimal success rate for small cases. In particular, it was known that no classical strategy can exceed 75% average accuracy for the 2-bit case, and the majority-vote strategy achieves this bound. This strongly suggested that using majority encoding is best for larger $n$ as well. Indeed, **Tavakoli *et al.* (2015)** conjectured that “Alice sends the most frequent symbol and Bob outputs that symbol” is the optimal classical strategy for general $n$. They calculated formulae for the expected success probability of this majority algorithm in general bases, noting it matched all known optimal values in binary cases.

**Proof of optimality:** The conjecture was later **proven rigorously by Ambainis, Kravchenko, and Rai (2015)**. They showed that for classical RACs with any alphabet size (including binary), **no other deterministic or even probabilistic strategy can surpass the majority-encoding scheme’s average success rate**. In fact, their theorem establishes that the majority vote (with identity decoding) is one of a family of optimal strategies, meaning if a strategy beats the majority’s performance, it would violate the derived optimality conditions. The proof characterizes all such optimal strategies and confirms that majority encoding is among them, thereby *“provid(ing) a firm basis”* for using majority as the benchmark for classical RAC performance. 

> **Theorem (Ambainis *et al.* 2015):** *Alice sending the most frequent letter in her input $x$ and Bob answering with that same letter for any query $j$ is an optimal deterministic strategy for the $n \to 1$ RAC (no classical strategy can achieve higher average success)*. *(They further found that this strategy is essentially unique up to symmetry – any optimal strategy must act like a “majority vote” possibly after relabeling symbols.)* 

## Conditions and Limitations

**Scope of optimality:** The majority algorithm’s optimality holds under the usual assumptions of RACs: Alice’s input bits are uniformly random, and Bob’s target index is uniformly random (so each bit is equally likely to be asked). The optimality is in the **average-case sense** – it maximizes the overall probability of correct decoding averaged over all inputs and target choices. Under these conditions, even allowing shared randomness or probabilistic encoding cannot surpass the performance of a deterministic majority vote strategy (by linearity of expectation, any randomized strategy is a mixture of deterministic ones, so at least one deterministic optimum exists). 

In the context of QSeaBattle the optimality holds holds for uniform inputs (enemy probability equal to 50%), a communication size of 1 bit ($m = 1$), and noiseless channels. 

## QSeaBattle implementation
The majority algorithm is implements as one of the deterministic strategies. In case Alice is allowed to communicate more than one but ($m > 1$), we implement the majority algorithm block-wise (the total input string is divided in blocks of equal size $L$. The total field size is $mL$). The expected probability that Bob is correct on a randomly chosen index, given that Alice sends the majority bit of the corresponding block of size $L$.

As performance reference we implement `expected_win_rate_majority(...)` to calculate the expected win rate for the block-wise application of the majority algorithm for a given field size, comms size, enemy probability and channel noise. The performance reference holds for the algorithm as implemented, but the proof that this is the optimal algorithm is subject to more stringent conditions as stated in the previous paragraph.

## Key References

*   **Tavakoli *et al.***, *"Quantum Random Access Codes using Single d-level Systems,"* *Phys. Rev. Lett. 114, 170502 (2015)* – Introduced high-level (d-ary) RACs and hypothesized that **“majority-encoding & identity-decoding”** is optimal classically. Provides intuition and examples for small $n$ cases. Quote: "However, intuition strongly suggests that an optimal strategy is for Alice to use majority encoding i.e. Alice counts the number of times each of the values {0, ..., d − 1} appears in her string x and outputs the d-level that occurs most frequently. In case of a tie, she can output either d-level. Bob does identity decoding
and thus outputs whatever he receives from Alice."

*   **Ambainis, Kravchenko & Rai (2015)**, *"Quantum Advantages in (n, d)7→1 Random Access
Codes"* *arXiv:1510.03045* – **Proof** that the majority algorithm indeed achieves the **maximum average success probability** for classical RACs. Characterizes all optimal strategies and confirms majority-vote is among them. This work settled the conjecture posed by Tavakoli *et al.* and solidified the majority algorithm as the benchmark for classical RAC performance. 
