***

### 1. No assistance (local / deterministic)

**Resource**: No pre-shared randomness or correlation.

**Strategy**:
Alice chooses $a = f(x)$, Bob chooses $b = g(y)$, for some deterministic functions $f,g$. (Equivalently, they may use local randomness, but since the goal is to maximize the winning probability, there is always an optimal deterministic strategy.)[3][1]

**Value**: The classical value $\omega_{\text{c}}(G)$, i.e., the maximum winning probability over all such local strategies.[1][2]

---

### 2. Shared classical randomness (SR)

**Resource**: A shared classical random variable $\lambda$ with distribution $P(\lambda)$.

**Strategy**:
Alice and Bob agree on functions $a = f(x,\lambda)$, $b = g(y,\lambda)$. During the game, they both see $\lambda$ and compute their answers based on their input and $\lambda$.[2][3]

**Value**: The classical value with shared randomness, $\omega_{\text{SR}}(G)$, which is the maximum winning probability over all such strategies.[3][2]

***

### 3. Entanglement assistance (EA)

**Resource**: A shared quantum state $\rho_{AB}$ on a bipartite Hilbert space $\mathcal{H}_A \otimes \mathcal{H}_B$.

**Strategy**:
Alice and Bob choose, for each $x$, a POVM $\{M_a^x\}_a$ on $\mathcal{H}_A$, and for each $y$, a POVM $\{N_b^y\}_b$ on $\mathcal{H}_B$.
On input $x$, Alice measures her system with $\{M_a^x\}_a$ and outputs outcome $a$; similarly, Bob outputs $b$ from $\{N_b^y\}_b$.[4][3]

**Value**: The entanglement-assisted value $\omega_{\text{EA}}(G)$, i.e., the supremum over all finite-dimensional states $\rho_{AB}$ and measurements.[4][3]

---

### 4. PR assistance (PR-assisted strategies)

**Resource**: A shared no-signalling box (PR-type box) that, on inputs $x,y$, produces outputs $a,b$ with conditional probabilities $P(a,b|x,y)$ satisfying the no-signalling conditions:
$
\sum_b P(a,b|x,y) = \sum_b P(a,b|x,y') \quad \text{and} \quad \sum_a P(a,b|x,y) = \sum_a P(a,b|x',y)
$
for all $x,x',y,y'$.[5][6]

**Strategy**:
-Alice and Bob use the box as a black box: on input $x,y$, they obtain $a,b$ from the box and send those as their answers.
The box may be parametrized by a quality factor $q$ that controls the CHSH value $S(q)$, so that:
$S(q) \le 2$: local box (can be simulated classically),
$2 < S(q) \le 2\sqrt{2}$: quantum box (can be simulated with entanglement),
- $2\sqrt{2} < S(q) \le 4$: post-quantum box (PR box at $S=4$).[6][5]

**Value**: The PR-assisted value $\omega_{\text{PR}}(G)$, i.e., the supremum over all no-signalling boxes (or over the family of boxes parametrized by $q$).[5][6]

---

### Suggested wording in your text

> In this work, we consider the following classes of strategies for the nonlocal game:
>
> - **No assistance**: Alice and Bob use only local randomness; their answers are deterministic functions $a = f(x)$, $b = g(y)$.
> - **Shared classical randomness (SR)**: Alice and Bob share a classical random variable $\lambda$ and choose answers $a = f(x,\lambda)$, $b = g(y,\lambda)$.
> - **Entanglement assistance (EA)**: Alice and Bob share a bipartite quantum state $\rho_{AB}$ and choose answers via local measurements on their respective subsystems.
> - **PR assistance (PREA)**: Alice and Bob share a no-signalling box with inputs $x,y$ and outputs $a,b$, whose correlations are parametrized by a quality factor $q$ that tunes the CHSH value from local to quantum to post-quantum regimes.
>
> For each class, the value of the game is the supremum of the winning probability (or expected payoff) over all strategies in that class.


Bronnen

[1] Extended non-local games and monogamy-of-entanglement ... https://royalsocietypublishing.org/rspa/article/472/2189/20160003/57103/Extended-non-local-games-and-monogamy-of

[2] Nonlocal Games — toqito 1.0.2 documentation https://toqito.readthedocs.io/en/1.0.2/nonlocal_games.html

[3] Lecture 6 - Nonlocal games and Tsirelson's theorem https://cs.uwaterloo.ca/~watrous/QIT-notes/QIT-notes.06.pdf

[4] Playing nonlocal games with phases of quantum matter https://link.aps.org/doi/10.1103/PhysRevB.107.045412

[5] When Are Popescu-Rohrlich Boxes and Random Access ... https://link.aps.org/doi/10.1103/PhysRevLett.113.100401

[6] When Are Popescu-Rohrlich Boxes and Random Access ... https://arxiv.org/abs/1307.7904
