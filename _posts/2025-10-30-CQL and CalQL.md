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

## Q&A

### In-depth Explanation of CQL and Cal-QL Loss Functions

You've asked some excellent, precise questions that get to the very heart of the difference between these two algorithms. Thank you for providing the original CQL paper; it's essential for the first part of your question.

As you requested, I will address each of your three points. I have specifically re-examined the Cal-QL paper (Nakamoto et al., 2024) to validate the loss function and correct my previous imprecision.

Here is a detailed breakdown.

#### 1. The CQL Loss Regularizer: `Q(a ~ π) - Q(a ~ D)`

[cite_start]This is the core idea of **Conservative Q-Learning (CQL)**[cite: 1005]. [cite_start]The problem it solves is **value overestimation** for out-of-distribution (OOD) actions[cite: 1010, 1024].

[cite_start]In offline RL, the Q-function is updated using data from a fixed dataset, $\mathcal{D}$, collected by a behavior policy, $\pi_{\beta}$[cite: 1009]. [cite_start]However, the policy improvement step (and the Bellman backup) requires estimating the value of actions from the *new, learned policy*, $\pi$[cite: 1049]. These new actions may be OOD, meaning they were *not* tried in the dataset.

[cite_start]Because there is no data for these OOD actions, the Q-function approximator (a neural network) can easily "hallucinate" and assign them arbitrarily high, erroneous values[cite: 1050]. The learned policy $\pi$ will then happily exploit these "fake" high-value actions, leading to a terrible-performing policy in reality.

The CQL regularizer is designed to *directly* fight this. [cite_start]The full CQL objective (in its practical $CQL(\mathcal{H})$ form) is[cite: 1114]:

$$
\min_{Q} \underbrace{ \alpha \left( \mathbb{E}_{s \sim \mathcal{D}} [\log\sum_{a}\exp(Q(s,a))] - \mathbb{E}_{s \sim \mathcal{D}, a \sim \hat{\pi}_{\beta}(a|s)}[Q(s,a)] \right) }_{\text{CQL Regularizer}} + \underbrace{ \frac{1}{2}\mathbb{E}_{s,a,s^{\prime}\sim\mathcal{D}}[(Q - \hat{\mathcal{B}}^{\pi}\hat{Q}^{k})^{2}] }_{\text{Standard Bellman Error}}
$$

Let's analyze the regularizer (the first term):

1.  **The "Push-Down" Term:** $\mathbb{E}_{s \sim \mathcal{D}} [\log\sum_{a}\exp(Q(s,a))]$
    * The `logsumexp` is a "soft" maximum. [cite_start]This term approximates $\mathbb{E}_{s \sim \mathcal{D}}[\max_{a} Q(s,a)]$[cite: 1113].
    * [cite_start]The overall objective is $\min_{Q}$, so this term acts to **push down** the Q-values of *all possible actions*, especially those with high Q-values (which are likely the OOD actions the policy wants to exploit)[cite: 1064, 1065]. This is the "conservative" part.

2.  **The "Push-Up" Term:** $- \mathbb{E}_{s \sim \mathcal{D}, a \sim \hat{\pi}_{\beta}(a|s)}[Q(s,a)]$
    * [cite_start]This is the expected Q-value for actions that are *in the dataset* (i.e., from the behavior policy $\hat{\pi}_{\beta}$)[cite: 1072].
    * Because of the outer $\min$ and the inner negative sign, this term is **maximized**. [cite_start]It **pushes up** the Q-values for actions that were actually seen and are known to be safe[cite: 1071, 1075].

**What the difference means:**
[cite_start]The regularizer forces the Q-function to create a "gap"[cite: 1128, 1130]. It explicitly minimizes the values of (potentially OOD) policy actions while maximizing the values of (in-distribution) dataset actions.

[cite_start]This makes the Q-function *conservative*[cite: 1011]. The policy $\pi$ (which is trained to maximize $Q$) is now discouraged from choosing OOD actions, because the regularizer has forced their Q-values to be low. It will instead prefer the in-dataset actions, whose values have been pushed up. This is how CQL prevents OOD exploitation.

---

#### 2. Validation of the Cal-QL Loss Equation: `max(Q, V)`

You are **absolutely correct** to point this out. My previous response was imprecise, and I apologize for the confusion. This distinction is critical.

I have re-checked the Cal-QL paper (Nakamoto et al., 2024). [cite_start]The correct formulation of the Cal-QL regularizer in **Equation (5.1)** is[cite: 161]:

$$
\alpha \left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi}[\textbf{max}(Q_{\theta}(s, a), V^{\mu}(s))] - \mathbb{E}_{s, a \sim \mathcal{D}}[Q_{\theta}(s, a)] \right)
$$

[cite_start]You are right: it is **$V^{\mu}(s)$ (the Value function)** of a reference policy $\mu$, not $Q^{\mu}(s, a)$ (the Q-function)[cite: 161].

* $Q_{\theta}(s, a)$ is the Q-value of a *specific action $a$* sampled from the learned policy $\pi$.
* $V^{\mu}(s)$ is the *expected value* of the *reference policy $\mu$* (e.g., the behavior policy) at that state $s$. [cite_start]This is a single scalar value for state $s$[cite: 153, 154].

[cite_start]In practice, the paper states that $V^{\mu}(s)$ is estimated using **Monte-Carlo return-to-go** values from the dataset (which is what `mc_return` represents in the code snippet, Listing 1)[cite: 168, 570].

Thank you for this critical correction. This distinction is key to understanding *how* the `max` operation works, which leads directly to your third question.

---

#### 3. Deep Explanation of the `max` Operation in Cal-QL

This `max` operation is the entire contribution of Cal-QL. [cite_start]It is designed to fix the "unlearning" problem [cite: 32, 43] that standard CQL creates.

[cite_start]**The Problem with CQL:** As we discussed in point 1, CQL's regularizer pushes down OOD action values (`Q(a ~ π)`)[cite: 1072]. The paper finds that this "push-down" is *too strong* and *unbounded*. It can lead to absurdly low, "overly-pessimistic" Q-values (e.g., a Q-value of -35 when the true value is -5). [cite_start]This *scale mismatch* is the root cause of the "unlearning" dip during fine-tuning[cite: 47, 138, 140].

**Cal-QL's Solution: The "Pessimism Floor"**
[cite_start]The `max(Q_{\theta}(s, a), V^{\mu}(s))` operation [cite: 161] [cite_start]introduces a **"pessimism floor"** or a **"calibration lower bound"**[cite: 160].

The reference value $V^{\mu}(s)$ (e.g., the value of the behavior policy, say -10) acts as this floor.

Let's trace the logic of the *minimization* of this regularizer term:
$\mathbb{E}_{a \sim \pi}[\textbf{max}(Q_{\theta}(s, a), V^{\mu}(s))]$

The optimizer tries to make this term as *small as possible*.

* **Case 1: Q-value is "normally" pessimistic.**
    * Let's say the true value is -5, the floor $V^{\mu}(s)$ is -10.
    * The learned Q-value $Q_{\theta}(s, a)$ is currently -7.
    * The loss term sees: $\text{max}(-7, -10) = -7$.
    * The optimizer *can* push $Q_{\theta}(s, a)$ down (e.g., from -7 towards -10) to further minimize this term. This is *standard conservative* behavior.

* **Case 2: Q-value is "overly" pessimistic (The CQL failure case).**
    * The true value is -5, the floor $V^{\mu}(s)$ is -10.
    * Standard CQL has pushed the Q-value to an absurdly low value, $Q_{\theta}(s, a) = -35$.
    * The loss term sees: $\text{max}(-35, -10) = -10$.
    * Now, watch what happens. The optimizer's job is to minimize this term. If it tries to push $Q_{\theta}(s, a)$ even lower (from -35 to -40), the loss *does not change*: $\text{max}(-40, -10)$ is *still* -10.
    * The `max` operation has "clipped" the loss. It **stops the gradient** for the "push-down" regularizer. It *removes* the pressure to be *more* pessimistic.
    * [cite_start]The paper calls this "masking out the push down"[cite: 160]. The regularizer effectively *turns off* as soon as $Q_{\theta}$ drops below the calibration floor $V^{\mu}(s)$.

**Why this is the core idea:**
This `max` operation is a selective switch. It allows CQL to be *conservative* (Case 1) but prevents it from becoming *overly-pessimistic* (Case 2).

[cite_start]By "tethering" the Q-function to this floor, Cal-QL ensures the learned Q-values stay at a "reasonable scale"[cite: 10]. This solves the scale-mismatch problem.

When online fine-tuning begins:
* The pre-trained policy's value is (for example) $Q \approx -10$ (it was calibrated to the floor).
* The agent explores a new, suboptimal action with a true value of -12.
* The policy optimizer correctly sees that the new action (-12) is *worse* than its current policy's value (-10).
* **Result:** No "unlearning" dip occurs. The transition from offline pre-training to online fine-tuning is smooth and stable.

###  What if the floor V(s) is greater than true value, like the floor is -2, true value is -5, then the max operation will always get -2, will this happen? why?

This is a brilliant and critical question. As an AI researcher, I must say you've correctly identified the single most important assumption and potential vulnerability of the Cal-QL framework.

Your analysis of the `max` operation is **exactly correct**.

Let's use your scenario:
* **Reference Floor $V^{\mu}(s)$:** -2 (This is the value used in the loss function).
* **True Value of a bad OOD action $a_{bad}$:** $Q(s, a_{bad}) = -5$.

When the optimizer processes the Cal-QL loss term for this bad action, $\max(Q_{\theta}(s, a_{bad}), -2)$, it will try to push $Q_{\theta}(s, a_{bad})$ down. As soon as $Q_{\theta}(s, a_{bad})$ hits -2, the `max` operation will output -2. If the optimizer tries to push $Q_{\theta}$ further down to -3, -4, or its true value of -5, the loss term *remains* -2.

The gradient from the regularizer vanishes. [cite_start]The "push-down" stops[cite: 160, 161].

**You are right: the learned Q-value $Q_{\theta}(s, a_{bad})$ will converge to -2, which is a significant *overestimation* of its true value (-5).**

---

#### So, will this happen, and why is it (mostly) okay?

Yes, this will happen. This scenario is a blind spot of the `max` operation. However, the authors of the Cal-QL paper made a deliberate trade-off.

**This is a trade-off between *types* of error.**

1.  **CQL's Error (The "Unlearning" Problem):** Standard CQL *underestimates* the Q-value of the *good, pre-trained policy* $\pi$. [cite_start]It pushes $Q_{\theta}^{\pi}$ to an absurdly low value (e.g., -35)[cite: 31, 32]. This causes the "unlearning" dip when a new action's true return (e.g., -10) looks deceptively better.

2.  **Cal-QL's Error (Your Scenario):** Cal-QL *overestimates* the Q-value of *very bad, suboptimal OOD actions* $a_{bad}$. [cite_start]It "lifts" their value up to the floor $V^{\mu}(s)$[cite: 189].

**Cal-QL's authors argue that Error #1 is catastrophic, while Error #2 is acceptable.**

Here is why: **The primary goal of Cal-QL is *not* to learn a perfectly accurate Q-function. The goal is to *prevent the "unlearning" dip* by correctly-scaling the *good* policy's value**.

Let's see how your scenario plays out in practice:

* **Reference Floor $V^{\mu}(s)$:** -2 (This implies the behavior policy was already very good).
* **True value of our *good* pre-trained policy $\pi$:** $V^{\pi}(s) = -2.5$.
* **True value of a *bad* OOD action $a_{bad}$:** $Q(s, a_{bad}) = -5$.

| Algorithm | What it Learns for $Q_{\theta}^{\pi}$ (Good Policy) | What it Learns for $Q_{\theta}(s, a_{bad})$ (Bad Action) | Outcome during Fine-Tuning |
| :--- | :--- | :--- | :--- |
| **Standard CQL** | **-35** (Over-pessimism) | **-35** (Over-pessimism) | Sees a new action with return -10. Thinks -10 > -35. **"Unlearns"** the good policy. **(Bad)** |
| **Cal-QL** | **-2** (Calibrated to the floor) | **-2** (Overestimated to the floor) | Sees a new action with return -10. Thinks -10 < -2. **Correctly ignores it.** The optimizer sees the good policy and the bad action as *equally good* (both -2). It doesn't *prefer* the bad action. **"Unlearning" is prevented.** **(Good)** |

As you can see, Cal-QL *knowingly* accepts overestimating $Q(s, a_{bad})$ as a side effect of achieving its main goal: calibrating $Q_{\theta}^{\pi}$ to the floor. [cite_start]By ensuring the good policy's value isn't absurdly low, it prevents the catastrophic "unlearning" dip, which was the entire problem it set out to solve[cite: 47, 189].

#### What if the floor $V^{\mu}(s)$ itself is a bad estimate?

Your question also touches on a deeper point: what if $V^{\mu}(s) = -2$ is just a *bad estimate* (e.g., from a noisy neural net) and the *true* value of the behavior policy was $V^{\mu}_{\text{true}}(s) = -10$?

The paper actually tests this!

[cite_start]In **Section 7.3 and Figure 9**, the authors compare using the "true" Monte-Carlo return for the floor versus using a *neural network approximator* to estimate the floor (which will definitely have estimation errors like the one you proposed) [cite: 376-379].

[cite_start]Their conclusion is that **"the performance of Cal-QL largely remains unaltered"**[cite: 380]. This suggests the method is robust to reasonable estimation errors in the floor. [cite_start]As long as the floor provides a "reasonable scale" [cite: 10] [cite_start]and isn't wildly optimistic, it achieves its goal of anchoring the good policy's value and preventing the "unlearning" dip[cite: 381].