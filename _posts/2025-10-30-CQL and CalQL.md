---
title: "A Tutorial on CQL and Cal-QL: From Offline Conservatism to Online Fine-Tuning"
categories: tech
tags: [Reinforcement learning]
use_math: true
toc: true  # enables the sidebar TOC
toc_label: "On this page"  # optional, custom title for TOC
toc_sticky: true  # optional, makes the TOC stick while scrolling
---

This guide reviews two important algorithms in reinforcement learning, Conservative Q-Learning (CQL) and Calibrated Q-Learning (Cal-QL), explaining the problem each one solves and how their loss functions are constructed.

---

## 1. Conservative Q-Learning (CQL)

[cite_start]**Goal:** To learn effective policies from a static, **offline dataset** without any online interaction[cite: 5].

### The Problem: OOD Overestimation

[cite_start]The central challenge in offline RL is **distributional shift**[cite: 6]. [cite_start]A standard Q-learning algorithm is updated using data from the dataset $\mathcal{D}$ (collected by a "behavior policy" $\pi_{\beta}$), but it needs to evaluate the value of actions from its *new, learned policy* $\pi$[cite: 45].

[cite_start]These new actions, $a \sim \pi$, may be "out-of-distribution" (OOD)—that is, they were never tried in the dataset[cite: 46]. [cite_start]A neural network Q-function can easily "hallucinate" and assign these unseen OOD actions **erroneously high Q-values**[cite: 20]. [cite_start]The policy $\pi$ will then learn to exploit these "fake" high-value actions, resulting in a policy that performs terribly in the real world[cite: 46].

### The Solution: The CQL Conservative Regularizer

[cite_start]CQL's solution is to **force the Q-function to be "conservative"**[cite: 7]. [cite_start]It does this by adding a special regularizer to the standard Bellman error loss[cite: 9].

The core idea is to **create a gap** in the Q-values:

1.  **Push Down** Q-values for (potentially OOD) actions sampled from the learned policy $\pi$.
2.  **Push Up** Q-values for (in-distribution) actions sampled from the dataset $\mathcal{D}$.

[cite_start]This regularizer is expressed in the CQL loss function (using the practical $CQL(\mathcal{H})$ variant)[cite: 109, 110]:

$$
\min_{Q} \alpha \left( \underbrace{\mathbb{E}_{s \sim \mathcal{D}} [\log\sum_{a}\exp(Q(s,a))]}_{\text{1. Push-Down Term (Soft-Max)}} - \underbrace{\mathbb{E}_{s \sim \mathcal{D}, a \sim \hat{\pi}_{\beta}}[Q(s,a)]}_{\text{2. Push-Up Term (Dataset Actions)}} \right) + \mathcal{L}_{\text{Bellman}}
$$

* **Term 1 (Push-Down):** The `logsumexp` term is a "soft" maximum over all actions. By *minimizing* this, CQL effectively **pushes down** the Q-values of all actions, especially the high-value OOD ones that the policy $\pi$ would want to take.
* **Term 2 (Push-Up):** This is the expected Q-value of actions *from the dataset*. Because of the outer minimization and the negative sign, this term is *maximized*. This **pushes up** the Q-values for actions that were actually seen and are known to be safe.

[cite_start]**Result:** The Q-function learns to be pessimistic about unknown actions and optimistic about known, in-dataset actions[cite: 125, 127]. This prevents the policy from exploiting OOD actions.

---

## 2. Calibrated Q-Learning (Cal-QL)

[cite_start]**Goal:** To solve a new problem created by CQL: **poor performance during online fine-tuning**[cite: 8].

### The Problem: "Unlearning" from Over-Pessimism

[cite_start]CQL is very effective for *purely offline* learning, but it performs poorly when you try to fine-tune it online[cite: 8]. The authors of Cal-QL diagnosed why:

1.  **Over-Pessimism:** CQL's "push-down" regularizer is unbounded. [cite_start]It can make the Q-values **absurdly low** (e.g., a Q-value of -35 when the true best value is -5)[cite: 31, 32].
2.  [cite_start]**Scale Mismatch:** This creates a Q-function whose values are at a completely different *scale* from the true environmental returns[cite: 10].
3.  **"Unlearning" Dip:** When online fine-tuning begins, the agent explores. If it tries a *new, suboptimal* action (e.g., true value of -10), that action's return (-10) will look *much better* than the *overly-pessimistic* Q-value of the "good" pre-trained policy (-35).
4.  [cite_start]**Result:** The agent is "deceived" and updates its policy *towards* the new, bad action, causing a sharp performance drop ("unlearning")[cite: 10].

### The Solution: The Cal-QL "Pessimism Floor"

[cite_start]Cal-QL's solution is simple but highly effective: it modifies the CQL regularizer to add a **"pessimism floor,"** preventing the Q-values from dropping too low[cite: 44, 46].

[cite_start]This is the **one-line change** to the "push-down" term[cite: 13, 46]:

$$
\min_{Q} \alpha \left( \underbrace{\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi}[\textbf{max}(Q_{\theta}(s, a), V^{\mu}(s))]}_{\text{1. Calibrated Push-Down Term}} - \underbrace{\mathbb{E}_{s, a \sim \mathcal{D}}[Q_{\theta}(s, a)]}_{\text{2. Push-Up Term}} \right) + \mathcal{L}_{\text{Bellman}}
$$

* [cite_start]**Term 1 (Calibrated Push-Down):** This is the core contribution[cite: 46, 95].
    * $Q_{\theta}(s, a)$ is the Q-value of the action from the learned policy $\pi$.
    * $V^{\mu}(s)$ is the "floor." [cite_start]It's the estimated value of a *reference policy* $\mu$ (in practice, the behavior policy $\pi_{\beta}$ that generated the data)[cite: 44, 48]. [cite_start]This value is estimated using Monte-Carlo returns from the dataset[cite: 48, 100].

**How the `max` operation works:**
[cite_start]The `max` operator acts as a "clip" or "mask" on the "push-down" gradient[cite: 46].

* **If $Q_{\theta}(s, a) > V^{\mu}(s)$:** The Q-value is *above* the floor. The loss term is $\mathbb{E}[Q_{\theta}(s, a)]$, and the regularizer pushes it down (standard conservative behavior).
* **If $Q_{\theta}(s, a) \le V^{\mu}(s)$:** The Q-value has *hit or fallen below* the floor. The loss term becomes $\mathbb{E}[V^{\mu}(s)]$. Since $V^{\mu}(s)$ is a fixed target (a pre-calculated return), the gradient of this term with respect to $Q_{\theta}$ becomes **zero**. The "push-down" pressure *vanishes*.

**Result:** Cal-QL is still conservative, but it stops being pessimistic once the Q-value hits the "reasonable" floor set by the behavior policy. [cite_start]This **"calibrates"** the Q-function to the correct scale[cite: 10]. This solves the scale-mismatch problem, prevents the "unlearning" dip, and allows for stable, efficient online fine-tuning.

---

## Summary: CQL vs. Cal-QL

| Feature | **CQL (Conservative Q-Learning)** | **Cal-QL (Calibrated Q-Learning)** |
| :--- | :--- | :--- |
| **Problem Solved** | [cite_start]**OOD Overestimation** in offline RL[cite: 6, 20]. | [cite_start]**"Unlearning" Dip** during online fine-tuning[cite: 8, 30, 31]. |
| **Core Idea** | Be **conservative**. [cite_start]Create a "gap" by pushing down OOD action values and pushing up dataset action values[cite: 68]. | Be **calibrated**. [cite_start]Be conservative, but only down to a "floor" to maintain a reasonable value scale[cite: 10, 11]. |
| **Mechanism** | [cite_start]Unbounded "push-down" regularizer on $Q(a \sim \pi)$[cite: 68]. | [cite_start]Bounded "push-down" regularizer: $\max(Q(a \sim \pi), V^{\mu}(s))$[cite: 46]. |
| **Strength** | [cite_start]State-of-the-art for **purely offline** learning[cite: 10]. | [cite_start]State-of-the-art for **offline-to-online** fine-tuning[cite: 14]. |
| **Weakness** | [cite_start]Can be **overly-pessimistic**, leading to "unlearning" when fine-tuned[cite: 8, 30]. | (Not explicitly for offline-only tasks; designed for the fine-tuning setting). |