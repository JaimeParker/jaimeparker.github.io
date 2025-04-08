---
title: "From Q-learning to Soft Actor Critic"
categories: tech
tags: [Reinforcement learning]
use_math: true
---

We shall introduce the foundation of value-base RL algos, Q-learning. Then from Q-learning to DQN, A2C, and finally to SAC.

## 1. Background: Q-Learning

### 1.1 What is a Q?

In reinforcement learning, an agent interacts with a Markov Decision Process (MDP), defined by:

- A set of states $\mathcal{S}$
- A set of actions $\mathcal{A}$
- Transition dynamics $P(s' \mid s, a)$
- A reward function $r(s, a)$
- A discount factor $\gamma \in [0, 1]$

<mark>**State value function $V^\pi(s)$**</mark>

The **state-value function** under a policy $\pi$ is defined as the expected return when starting in state $s$ and following $\pi$ thereafter:

$$
V^\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s \right]
$$

<mark>**Action value function $Q^\pi(s, a)$**</mark>

The **action-value function** under policy $\pi$ is the expected return starting from state $s$, taking action $a$, and then following policy $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s, a_0 = a \right]
$$

So compared to state value function, there is an action in this function.

<mark>**Relationship Between $Q^\pi(s, a)$ and $V^\pi(s)$**</mark>

The value of a state can be expressed as the expected value over actions drawn from the policy:

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ Q^\pi(s, a) \right]
$$

Thus, $Q$ quantifies **how good an action is** at a given state, while $V$ quantifies **how good the state is**, under a specific policy.

### 1.2 What is Q-Learning?

Q-learning is a **model-free**, **off-policy** algorithm that learns the optimal action-value function $Q^*(s, a)$, defined as:

$$
Q^*(s, a) = \max_{\pi} Q^\pi(s, a)
$$

That is, $Q^*(s, a)$ gives the expected return of taking action $a$ in state $s$ and thereafter following the **optimal policy** $\pi^*$.

#### 1.2.1 Bellman Optimality Equation for $Q^*$

Q-learning is based on the **Bellman optimality equation**:

$$
Q^*(s, a) = \mathbb{E}_{s'} \left[ r(s, a) + \gamma \max_{a'} Q^*(s', a') \right]
$$

This equation is recursive and serves as a fixed-point definition for the optimal Q-function.

**Bootstrapping and Backup**

Q-learning updates are **bootstrapped** — meaning they rely on the agent’s own current estimates to update future values. Unlike Monte Carlo methods, which wait for full returns at the end of an episode, Q-learning performs **online updates** using a one-step target:

$$
\text{TD target} = r + \gamma \max_{a'} Q(s', a')
$$

Here, $Q(s', a')$ is not the true value but the agent’s **current estimate**. This is the essence of bootstrapping — updating based on **estimated values rather than ground truth**.

Additionally, Q-learning implements a **backup operation** — information from the successor state $s'$ is **backed up** to improve the current estimate $Q(s, a)$:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{backed-up estimate}} - Q(s, a) \right]
$$

This is a **one-step backup**, where the immediate reward and the estimated value of the next state are used to update the current value.

These concepts are central to **temporal difference (TD) learning** and are what make Q-learning both **sample-efficient** and **incremental**.

#### 1.2.2 Q-learning Update Rule

At each step, the agent observes a transition $(s, a, r, s')$ and updates the Q-value:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

Where:

- $\alpha$ is the learning rate
- $\gamma$ is the discount factor
- The term in brackets is the **temporal difference (TD) error**

This is a form of **stochastic approximation** toward the fixed point of the Bellman optimality equation.

#### 1.2.3 Policy Iteration in Q-Learning

Q-learning can be interpreted as performing a form of **generalized policy iteration** (GPI), alternating between:

<mark>**Policy Evaluation (Approximate)**</mark>

Unlike classical policy evaluation (which assumes a fixed policy), Q-learning **bootstraps** the evaluation by updating $Q(s,a)$ based on one-step lookahead:

- Uses the **greedy action** in the next state to evaluate the current state-action pair.
- Does not wait for convergence — hence **approximate**.

<mark>**Policy Improvement (Implicit)**</mark>

At any point, we can extract a greedy policy from $Q$:

$$
\pi(s) = \arg\max_a Q(s, a)
$$

This defines the **greedy policy**, which becomes optimal as $Q(s,a)$ converges to $Q^*(s,a)$.

Although Q-learning does not explicitly represent a policy, it **implicitly improves the policy** by making $Q(s,a)$ closer to the optimal, and selecting better actions during planning or execution.

### 1.3 Theoretical Justification

Q-learning is a **stochastic approximation algorithm** that converges to the fixed point of the Bellman optimality operator:

Let the Bellman operator be:

$$
\mathcal{T}Q(s, a) = \mathbb{E}_{s'} \left[ r(s, a) + \gamma \max_{a'} Q(s', a') \right]
$$

This operator is a **$\gamma$-contraction** in the sup-norm:

$$
\| \mathcal{T}Q_1 - \mathcal{T}Q_2 \|_\infty \leq \gamma \| Q_1 - Q_2 \|_\infty
$$

Therefore, by Banach’s Fixed Point Theorem, repeated application of $\mathcal{T}$ converges to a unique fixed point: the optimal Q-function $Q^*$.

### 1.4 Training Q-Learning

At a high level, the Q-learning training loop consists of:

1. Initialize $Q(s,a)$ for all $s \in \mathcal{S}$, $a \in \mathcal{A}$
2. Repeat (for each episode or step):
   - Observe current state $s$
   - Choose action $a$ (e.g., using $\epsilon$-greedy policy)
   - Execute $a$ in the environment
   - Observe reward $r$ and next state $s'$
   - Update $Q(s, a)$ using the Bellman target
   - Update $s \leftarrow s'$
3. Continue until convergence

**Choose an action**

At state $s$, select action $a$ using an **exploration policy**:

- Commonly, $\epsilon$-greedy:

$$
\begin{cases}
\text{with probability } \epsilon & \text{choose a random } a \in \mathcal{A} \\
\text{with probability } 1 - \epsilon & a = \arg\max_{a'} Q(s, a')
\end{cases}
$$

**Update Q-Value**

Apply the Q-learning update rule:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

This moves the estimate $Q(s, a)$ toward a better approximation of $Q^*(s, a)$ using the one-step target:

$$
\text{TD target} = r + \gamma \max_{a'} Q(s', a')
$$

And it's also called TD target.

The TD error is:

$$
\delta = \text{TD target} - Q(s, a)
$$


### 1.5 Summary of Q-Learning

Q-learning is a foundational off-policy reinforcement learning algorithm that estimates the action-value function:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

- Learns the value of taking an action in a state.
- The policy is implicitly derived from the Q-function:  
  $$\pi(s) = \arg\max_a Q(s, a)$$
- Works well for discrete action spaces with small state/action counts.

Q-learning converges under certain conditions:

- The learning rate $\alpha_t(s, a)$ decays appropriately:
  $\sum_t \alpha_t(s,a) = \infty$
  $\sum_t \alpha_t^2(s,a) < \infty$
- Each state-action pair is visited infinitely often.
- The environment is a stationary MDP.

## 2. Deep Q-Networks (DQN)

DQN extends Q-learning using deep neural networks to approximate the Q-function in high-dimensional or continuous state spaces.

### 2.1 Motivation: Why Move Beyond Q-Learning

While Q-learning provides a theoretically grounded framework for value-based reinforcement learning, its **practical application in high-dimensional or continuous environments is limited** due to challenges in approximation, stability, and sample efficiency.

#### Limitations of Classical Q-Learning with Function Approximators:

1. **Instability with Nonlinear Function Approximators**:
   When combined with function approximators such as neural networks, Q-learning's bootstrapped updates can lead to divergence or oscillations. This is due to:
   - Correlations in sequential data
   - Non-stationary targets
   - Feedback loops between the Q-function and the target
2. **Sample Inefficiency**:
   Standard Q-learning updates use each transition exactly once and immediately discard it. This is inefficient, especially in environments where data collection is expensive.
3. **Poor Generalization Across Similar States**:
   Without proper architecture and training regularization, function approximators may overfit to individual transitions rather than generalizing well over the state space.
4. **Lack of Stabilization Mechanisms**:
   Classical Q-learning assumes exact updates and idealized convergence properties. In practice, approximators require architectural tools (e.g., target networks) to stabilize training.

These practical challenges necessitated the development of **Deep Q-Networks (DQN)** — a realization of Q-learning that integrates neural function approximation with a set of stabilization techniques.

### 2.2 Deep Q-Network (DQN): Core Idea

DQN replaces the Q-table with a **neural network** parameterized by $\theta$, approximating the Q-function:

$$
Q(s, a; \theta) \approx Q^*(s, a)
$$

This allows the agent to:

- Operate in high-dimensional state spaces (e.g., images)
- Learn generalized value functions
- Avoid the need for hand-crafted features or state abstractions

### 2.3 Architectural Contributions in DQN

While DQN is rooted in classical Q-learning, it introduces **key innovations** to ensure stable learning with deep networks.

- **Experience Replay**: stores transitions to break correlation and reuse data.
- **Target Network**: stabilizes learning by slowly updating a separate target Q-network.
- **Function Approximation**: Q is modeled with a deep neural network.

#### 2.3.1. **Neural Network Approximation**

A deep convolutional network processes visual input and outputs Q-values for all actions:

$$
s \rightarrow \text{CNN} \rightarrow [Q(s, a_1), \dots, Q(s, a_n)]
$$

This is essential for applying Q-learning to tasks like Atari games (raw pixels).

#### 2.3.2. **Experience Replay**

Rather than learning from sequentially correlated experiences, DQN stores transitions in a **replay buffer** $\mathcal{D}$:

- At each step, store $(s, a, r, s')$ in $\mathcal{D}$
- Sample mini-batches uniformly to perform stochastic gradient descent

This decorrelates data and improves sample efficiency.

#### 2.3.3. **Target Network**

To stabilize the bootstrapped target, DQN uses a **separate, delayed Q-network** $Q(s, a; \theta^{-})$ to compute the TD target:

$$
y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})
$$

- The parameters $\theta^{-}$ are updated to $\theta$ only every $K$ steps.
- This prevents oscillations and divergence caused by moving targets.

### 2.4 Comparison with Q-Learning

DQN is not a new algorithm; it is a **practical instantiation of Q-learning using function approximation**. It preserves the essential theoretical structure of Q-learning:

<mark>**Bellman Backup**</mark>

Both use the **Bellman optimality equation** to update the Q-function:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

In DQN, this becomes:

$$
\mathcal{L}(\theta) = \left( Q(s, a; \theta) - \left[ r + \gamma \max_{a'} Q(s', a'; \theta^-) \right] \right)^2
$$

Here, $\theta$ are the parameters of the Q-network.

<mark>**Bootstrapping**</mark>

Both perform **bootstrapped updates** — they use current estimates of $Q(s', a')$ to update $Q(s,a)$, without requiring full returns.

<mark>**Off-Policy Learning**</mark>

Both are **off-policy**: they learn the optimal Q-function independently of the behavior policy (e.g., $\epsilon$-greedy exploration).

<mark>**Implicit Greedy Policy Improvement**</mark>

Neither explicitly stores a policy. Instead, the action policy is derived from the current Q-function:

$$
\pi(s) = \arg\max_a Q(s, a)
$$

**Differences**

While DQN follows the theoretical framework of Q-learning, it introduces **practical modifications** to make Q-learning work with deep function approximation.

| Principle                     | Q-Learning (Abstract)                     | DQN (Implementation)                          |
| ----------------------------- | ----------------------------------------- | --------------------------------------------- |
| **Q-function representation** | Arbitrary (can be tabular or approximate) | Deep neural network $Q(s, a; \theta)$         |
| **TD target**                 | $r + \gamma \max_{a'} Q(s', a')$          | $r + \gamma \max_{a'} Q(s', a'; \theta^-)$    |
| **Bootstrapping**             | Yes                                       | Yes                                           |
| **Experience usage**          | One-step online update                    | Experience replay buffer $\mathcal{D}$        |
| **Target network**            | Optional in theory                        | Essential for stability                       |
| **Update method**             | Incremental update to table or function   | Mini-batch SGD using replayed transitions     |
| **Stability guarantees**      | Convergent under certain assumptions      | No convergence guarantees; empirically stable |
| **Exploration**               | $\epsilon$-greedy                         | $\epsilon$-greedy                             |

### 2.5 Policy Iteration in DQN

While DQN does not explicitly maintain or update a policy network, it still performs **generalized policy iteration (GPI)** — just like classical Q-learning. This occurs through two alternating processes: **policy evaluation** and **policy improvement**, both encoded in the Q-function updates.

#### 2.5.1 Policy Evaluation (via Bellman Backup)

DQN evaluates the current policy implicitly by minimizing the **temporal difference (TD) error**, using a bootstrapped one-step target:

$$
\mathcal{L}(\theta) = \left( Q(s, a; \theta) - \left[ r + \gamma \max_{a'} Q(s', a'; \theta^-) \right] \right)^2
$$

This approximates the expected return under the **greedy policy** defined by the current Q-function, using the **target network** to stabilize the value target.

#### 2.5.2 Policy Improvement (Implicit Greedy Update)

Though DQN does not explicitly represent a policy $\pi$, it **implicitly improves the policy** by defining action selection as:

$$
\pi(s) = \arg\max_{a} Q(s, a; \theta)
$$

As training progresses, the Q-values better approximate the optimal action values, and the greedy policy derived from them improves accordingly.

#### 2.5.3 Policy Iteration without Explicit Policies

DQN, like Q-learning, implements **generalized policy iteration** in an implicit form:

- The Q-network updates perform **approximate policy evaluation**
- The greedy action selection performs **policy improvement**

> The key insight is that in DQN, the **policy is embedded within the Q-function**. There is no need for an explicit policy network to perform improvement.

### 2.6 Training DQN

Initialization:

- Initialize Q-network parameters $\theta$
- Initialize target network parameters $\theta^{-} \leftarrow \theta$
- Initialize empty replay buffer $\mathcal{D}$

For each training step:

1. **Observe current state** $s$

2. **Select action** $a$ using $\epsilon$-greedy:
   
   $$
   a =
   \begin{cases}
   \text{random action} & \text{with probability } \epsilon \\
   \arg\max_{a'} Q(s, a'; \theta) & \text{otherwise}
   \end{cases}
   $$

3. **Execute action** $a$, observe reward $r$ and next state $s'$

4. **Store transition** $(s, a, r, s')$ in $\mathcal{D}$

5. **Sample mini-batch** from $\mathcal{D}$

6. **Compute TD target** using target network:
   
   $$
   y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})
   $$

7. **Update Q-network** by minimizing squared TD error:
   
   $$
   \mathcal{L}(\theta) = \left( Q(s, a; \theta) - y \right)^2
   $$

8. **Periodically update** target network:
   
   $$
   \theta^{-} \leftarrow \theta
   $$

Repeat until convergence.

### 2.7 Summary of DQN

DQN Loss:

$$
\mathcal{L}_{\text{DQN}} = \left( Q(s, a) - \left[ r + \gamma \max_{a'} Q_{\text{target}}(s', a') \right] \right)^2
$$

Limitations:

- Cannot handle continuous action spaces due to the $$\max_{a'} Q(s', a')$$ term being non-differentiable.
- Policy is implicit and deterministic.

| Aspect                | Q-Learning (General)              | DQN (Specialized Form)                             |
| --------------------- | --------------------------------- | -------------------------------------------------- |
| Algorithmic Structure | General value iteration with TD   | Same                                               |
| Representation        | Tabular or function approximator  | Deep neural networks                               |
| Update Form           | Online incremental or batch       | Batch via SGD + delayed targets                    |
| Usage Domain          | Low-dimensional discrete problems | High-dimensional problems (e.g. pixels)            |
| Practical Stability   | Assumed under conditions          | Requires architectural tricks (target net, replay) |
| Policy Derivation     | $\arg\max_a Q(s, a)$              | Same                                               |

### 2.8 Double DQN, A Little More About DQN

While DQN introduced critical stabilization mechanisms for deep reinforcement learning, it also inherited a significant issue from classical Q-learning: **overestimation bias** in Q-value targets.

#### 2.8.1 The Problem: Overestimation in Q-learning

In both tabular and deep Q-learning, the TD target uses:

$$
\max_{a'} Q(s', a')
$$

This maximum over estimated values introduces a positive bias when Q-values are noisy, especially during early training. The neural network in DQN may amplify this bias due to approximation error.

**Result**: The agent becomes overconfident in suboptimal actions, leading to unstable or suboptimal learning.

#### 2.8.2 The Idea of Double Q-Learning

**Double Q-learning** (Hasselt, 2010) was introduced to address this overestimation by **decoupling action selection from action evaluation**.

Instead of using the same Q-function to both **select** and **evaluate** the action (as in standard Q-learning), Double Q-learning splits the roles:

$$
Q_{\text{Double}}(s, a) \leftarrow r + \gamma Q(s', \arg\max_{a'} Q'(s', a'))
$$

Where:

- $Q$ is used to select the best next action
- $Q'$ is used to evaluate it

This prevents upward bias from the maximization over noisy values.

#### 2.8.3 Double DQN: The Extension to Deep Learning

**Double DQN** (Van Hasselt et al., 2016) applies this idea to DQN using two networks:

- The **online network** $Q(s, a; \theta)$ selects the next action via $\arg\max$
- The **target network** $Q(s, a; \theta^-)$ evaluates the action

The TD target becomes:

$$
y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
$$

**In words**:

> Use the **online network** to select the action with the highest Q-value at the next state, and use the **target network** to evaluate that action’s value.

This small change significantly reduces overestimation bias.

#### 2.8.4 Comparison: DQN vs Double DQN

| Component             | DQN                                        | Double DQN                                                   |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| TD Target             | $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ | $r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$ |
| Action Selection      | online network                             | online network                                               |
| Action Evaluation     | target network                             | target network                                               |
| Overestimation Bias   | High (especially early in training)        | Reduced via decoupling                                       |
| Convergence Stability | Sensitive to noise and high variance       | Empirically more stable                                      |

<mark>**TD Target in DQN**</mark>

In **standard DQN**, the TD target is:

$$
y_{\text{DQN}} = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

- **$\theta^-$**: target network parameters
- The **same network** is used to:
  - **Select** the action at $s'$ via $\max_{a'}$
  - **Evaluate** the Q-value of that action

Problem:

- When Q-values are noisy (e.g., early in training), the max operator tends to **select overestimated values**, causing **positive bias** in the TD target.
- This is known as **overestimation bias**, and it accumulates over time, leading to unstable learning or convergence to suboptimal policies.

<mark>**TD Target in Double DQN**</mark>

In **Double DQN**, the TD target is modified to **decouple** action selection and action evaluation:

$$
y_{\text{DoubleDQN}} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
$$

- Use the **online network** ($\theta$) to **select** the best action:
  
  $$
  a^* = \arg\max_{a'} Q(s', a'; \theta)
  $$

- Use the **target network** ($\theta^-$) to **evaluate** the selected action:
  
  $$
  Q(s', a^*; \theta^-)
  $$

Benefit:

- Reduces overestimation bias by **not evaluating with the same network** that was used to select.
- Helps avoid propagating over-optimistic estimates through the Q-function.

#### 2.8.5 Summary of Double DQN

- Double DQN is an **incremental improvement** over DQN.
- It **does not change the architecture** — it modifies only the target computation.
- It retains all benefits of DQN while reducing overestimation.

**Target in DQN**:

$$
y_{\text{DQN}} = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

**Target in Double DQN**:

$$
y_{\text{DoubleDQN}} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
$$

This design choice helps make Q-learning with function approximation more robust, particularly in the presence of noisy or untrained Q-values.

### 2.9 Networks in DQN and Double DQN

#### 2.9.1. In DQN

DQN maintains **two neural networks**, both approximating the Q-function:

**Online Q-Network:**   $Q(s, a; \theta)$

- This is the main network that is trained.
- Used for:
  - Action selection (e.g., $\epsilon$-greedy or $\arg\max_a Q(s, a; \theta)$)
  - Evaluating Q-values for the current transition
  - Backpropagation and gradient updates

**Target Q-Network:**  $Q(s, a; \theta^-)$

- A **frozen copy** of the online network.

- Used **only to compute the TD target** in training:
  
  $$
  y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
  $$

- Updated **periodically** (e.g., every $K$ steps) via:
  
  $$
  \theta^- \leftarrow \theta
  $$

#### 2.9.2 In Double DQN

Double DQN **still uses the same two networks** as DQN:

**Online Q-Network:** $Q(s, a; \theta)$

- Same role as in DQN.

- **Used to select** the action for the next state:
  
  $$
  a^* = \arg\max_{a'} Q(s', a'; \theta)
  $$

**Target Q-Network:** $Q(s, a; \theta^-)$

- Same role as in DQN.

- **Used to evaluate** the selected action:
  
  $$
  y = r + \gamma Q(s', a^*; \theta^-)
  $$

#### 2.9.3 Summary Table

| Component              | DQN                             | Double DQN                                         |
| ---------------------- | ------------------------------- | -------------------------------------------------- |
| **Online Network**     | $Q(s, a; \theta)$               | $Q(s, a; \theta)$                                  |
| **Target Network**     | $Q(s, a; \theta^-)$             | $Q(s, a; \theta^-)$                                |
| **Target Equation**    | $\max_{a'} Q(s', a'; \theta^-)$ | $Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$ |
| **Number of Networks** | 2                               | 2                                                  |

## 3. Actor-Critic Paradigm

Actor-critic methods separate:

- The **actor**: a policy network that outputs $\pi(a\|s)$
- The **critic**: a value function estimator, such as $Q^\pi(s, a)$

### 3.1 Limitations of Value-Based Methods (DQN, Double DQN)

While DQN and Double DQN advanced value-based reinforcement learning, they are inherently limited by the structure of Q-learning and its reliance on value functions alone.

**Key Limitations:**

**Incompatibility with Continuous Action Spaces**

$$
\pi(s) = \arg\max_a Q(s, a)
$$

This maximization is infeasible in continuous action spaces, where $\mathcal{A}$ is infinite and $\arg\max$ is non-differentiable.

**No Explicit or Differentiable Policy**

$$
\pi(s) = \arg\max_a Q(s, a)
$$

This prevents use of gradient-based policy optimization and restricts learning to discrete actions.

**Exploration Challenges**

$\epsilon$-greedy policies are simple but poorly suited for environments requiring structured exploration or stochasticity.

**Overestimation Bias in Bootstrapped Targets**

$$
y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
$$

This bias can destabilize training, especially with function approximators.

**No Direct Policy Optimization**

Q-learning maximizes value indirectly through value iteration, not through direct optimization of a reward-maximizing policy.

### 3.2 Concept of Actor-Critic Architecture

The **actor-critic architecture** introduces a modular structure:

$$
\pi_\theta(a \mid s), V^\pi(s), Q^\pi(s, a)
$$

This decouples action selection from value estimation, allowing each to be optimized independently.

Value Function Definitions:

$$
V^\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s \right]
$$

$$
Q^\pi(s,a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a \right]
$$

### 3.3 Advantages of Actor-Critic

| DQN Limitation                               | Actor-Critic Solution                                      |
| -------------------------------------------- | ---------------------------------------------------------- |
| $\arg\max_a Q(s, a)$ undefined in continuous | Actor outputs $a = \pi_\theta(s)$ directly                 |
| Implicit, non-trainable policy               | Actor is explicit: $\pi_\theta(a \mid s)$                  |
| No stochasticity                             | Actor can be stochastic $\pi(a \mid s)$                    |
| No direct policy learning                    | Actor trained via gradient ascent on return                |
| Q-function bootstrapping bias                | Critic used for evaluation, not direct target maximization |

Policy Gradient Objective

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot A^\pi(s, a) \right]
$$

Advantage Function

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

## 4. Soft Actor-Critic (SAC)