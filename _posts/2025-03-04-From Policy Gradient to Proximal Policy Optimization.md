---
title: "From Policy Gradient to Proximal Policy Optimization"
categories: tech
tags: [Reinforcement learning]
use_math: true
toc: true  # enables the sidebar TOC
toc_label: "On this page"  # optional, custom title for TOC
toc_sticky: true  # optional, makes the TOC stick while scrolling
---

Basis of Policy Gradient methods, from PG to PPO.

## 1. 策略梯度 Policy Gradient

策略梯度是policy optimization的一种方法，用于直接优化参数化的策略。这种方法基于策略优化框架，通过最大化期望奖励来优化策略。

### 1.1 策略梯度方法的基本概念

在强化学习中，智能体通过与环境交互来最大化累积的奖励。
一个策略 $\pi_{\theta}(a|s)$ 是基于当前状态 $s$ 和参数化的策略网络 $\theta$ （$\theta$指策略的网络参数） 来选择动作 $a$ 的概率分布。
策略梯度方法通过优化这个策略的参数 $\theta$ 来实现智能体性能的提高。

**期望奖励 (Expected Reward)**

期望奖励 $\bar{R}\_{\theta}$ 是智能体在策略 $\pi\_{\theta}$ 下，执行动作 $a$ 时从环境中获得的累计奖励的期望值。

形式化表示为：

$$
\bar{R}_{\theta}=\sum_rR(\tau)p_{\theta}(\tau)={\mathbb E}_{\tau\sim p_{\theta}}[R(\tau)] \tag{1}
$$

其中，$\tau$ 是一个轨迹，$p_{\theta}(\tau)$ 是轨迹的分布，$R(\tau)$ 是轨迹的总奖励。

这样从分布 $p_{\theta}(\tau)$ 中采样一个轨迹 $\tau$，然后就可以通过梯度上升来计算其最大化期望奖励。而要进行梯度上升，首先需要计算其梯度。

### 1.2 策略梯度推导

策略梯度的目标是通过调整策略参数 $\theta$ 来最大化期望奖励。通过计算期望奖励的梯度，可以得到更新策略的方向。期望奖励的梯度可以通过以下公式表示：

$$
\nabla \bar{R}_{\theta}=\sum_{\tau}R(\tau)\nabla p_{\theta}(\tau) \tag{2}
$$

对 $\bar{R}\_{\theta}$ 做梯度运算，其中只有 $p\_{\theta}(\tau)$ 依赖于 $\theta$，因此可以将梯度运算移到 $p\_{\theta}(\tau)$ 上；而奖励函数 $R(\tau)$ 不需要是可微的。同时，奖励函数并不依赖于策略参数 $\theta$，它是通过环境和动作轨迹 $\tau$ 计算得到的，是在外部环境中定义的。因此，奖励函数 $R(\tau)$ 是与策略参数 $\theta$ 无关的常数，其对于参数 $\theta$ 更新的梯度计算没有影响。

其中，根据：

$$
\nabla f(x)=\frac{\nabla f(x)}{f(x)}f(x)=f(x) \nabla\log f(x) \tag{3}
$$

可以得到：

$$
\nabla p_{\theta}(\tau)=p_{\theta}(\tau)\nabla\log p_{\theta}(\tau) \tag{4}
$$

将公式（4）代入公式（2）中，可以得到策略梯度的表达式：

$$
\nabla \bar{R}_{\theta}=\sum_{\tau}R(\tau)p_{\theta}(\tau)\frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}=\sum_{\tau}R(\tau)p_{\theta}(\tau)\nabla\log p_{\theta}(\tau) \tag{5}
$$

其表示了在 $p_{\theta}(\tau)$ 下，$R_{\tau}\nabla \log p_{\theta}(\tau)$ 的加权，也就是期望。因此可以得到：

$$
\nabla \bar{R}_{\theta}={\mathbb E}_{\tau\sim p_{\theta}(\tau)}[R(\tau)\nabla\log p_{\theta}(\tau)] \tag{6}
$$

推导过程整体可以总结为：

$$
\nabla \bar{R}_{\theta}=\sum_{\tau}R(\tau)\nabla p_{\theta}(\tau)=\sum_{\tau}R(\tau)p_{\theta}(\tau)\frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}={\mathbb E}_{\tau\sim p_{\theta}(\tau)}[R(\tau)\nabla\log p_{\theta}(\tau)] \tag{7}
$$

### 1.2* 期望奖励与奖励函数梯度的讨论

**期望奖励的性质**

期望奖励 $\bar{R}_{\theta}$ 是一个标量，表示根据轨迹的概率分布加权求和的过程 （见式1）。期望奖励计算的目标是通过调整策略参数 $\theta$ 来最大化奖励。虽然期望奖励是一个标量值，但它是根据多个轨迹的奖励与其概率分布加权求和的，因此可以对其进行求导以优化策略。

**奖励函数的梯度**

奖励函数 $R(\tau)$ 反映了每个轨迹的奖励，通常与策略参数 $\theta$ 无关。它只依赖于环境和轨迹，且是给定的常数。在计算期望奖励的梯度时，虽然奖励函数的梯度 $\nabla R(\tau)$ 在数学上是可能存在的（当然大多数奖励函数的设计都是不可微的），但由于它不依赖于策略参数 $\theta$，它不会影响期望奖励梯度的计算，因此可以忽略掉。

**期望奖励的梯度反映到数学上的意义**

期望奖励的梯度反映的是期望奖励对策略参数 $\theta$ 的变化率。在策略梯度方法中，我们关心的是如何调整策略来增加期望奖励。因此，期望奖励的梯度通过以下公式表示：

$$
\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau) \nabla \log p_{\theta}(\tau)]
$$

这表明，期望奖励对策略参数 $\theta$ 的导数，不仅仅是一个函数的梯度，而是一个期望值，它反映了奖励和策略分布之间的关系。实际计算时，我们忽略奖励函数的梯度 $\nabla R(\tau)$，因为它不依赖于 $\theta$，而重点计算概率分布 $p\_{\theta}(\tau)$ 的导数。

### 1.3 策略梯度的采样

但由于 式6 得到的期望值无法计算，因此采用 Monte Carlo 采样的方式采样 $N$ 个 $\tau$ 轨迹并计算每一个的值并累加，从而得到梯度的近似，即：

$$
\mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau) \nabla \log p_{\theta}(\tau)] \approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log p_{\theta}(\tau^n) \tag{8}                      
$$

其中，需要对 $\nabla \log p_{\theta}(\tau)$ 进行计算，这是策略梯度的梯度计算。

我们首先对 $\log p_{\theta}(\tau)$ 进行分解，根据轨迹概率的乘积表示，可以将轨迹的概率分布写为：

$$
p_{\theta}(\tau) = p(s_1) \prod_{t=1}^{T} p_{\theta}(a_t|s_t) p(s_{t+1}|s_t, a_t) \tag{9}
$$


其中，
$p(s\_1)$ 是初始状态的概率，$p(s_{t+1}|s_t, a_t)$ 是状态转移概率，


接下来根据 $\log$ 函数化乘积为求和的性质，可以将 $\log p\_{\theta}(\tau)$ 分解为：

$$
\log p_{\theta}(\tau) = \log p(s_1) + \sum_{t=1}^{T} \log p_{\theta}(a_t|s_t) + \sum_{t=1}^{T} \log p(s_{t+1}|s_t, a_t) \tag{10}
$$

进而，对 $\log p_{\theta}(\tau)$ 求导，只有与策略参数 $\theta$ 有关的部分才会对梯度产生影响，即：

$$
\nabla \log p_{\theta}(\tau) = \sum_{t=1}^{T} \nabla \log p_{\theta}(a_t|s_t) \tag{11}
$$

将其带入 式8 中，可以得到策略梯度的采样表达式：

$$
\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau) \nabla \log p_{\theta}(\tau)] \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_{\theta}(a_t^n|s_t^n) \tag{12}
$$

**学习率**

用梯度上升来更新参数 $\theta$ 时，需要设置一个学习率 $\eta$ , 用于控制参数更新的步长。学习率的设置会影响策略梯度的收敛速度和稳定性。通常，学习率的设置需要根据具体的任务和策略网络来调整。

引入学习率后，对策略更新的表示可以写为：

$$
\theta \leftarrow \theta + \eta \nabla \bar{R}_{\theta} \tag{13}
$$

### 1.3* 策略梯度方法的训练流程

根据上述策略梯度的推导，可以总结策略梯度方法的训练流程：

* 首先，智能体与环境交互，通过策略网络 $\pi_{\theta}$ 选择动作，执行动作并观察环境的反馈，得到奖励信号。在每个时间步 $t$，根据当前状态 $s_t$ 选择动作 $a_t$，并执行动作，得到下一个状态 $s_{t+1}$ 和奖励信号 $r_t$。直到达到终止条件。
* 在每个时间步 $t$，
  计算策略梯度 $\nabla \log p_{\theta}(a_t|s_t)$，并根据式12计算策略梯度的采样值。
* 对策略参数 $\theta$ 进行更新，通过梯度上升的方式更新参数，使得期望奖励最大化。更新参数的方式为 $\theta \leftarrow \theta + \eta \nabla \bar{R}_{\theta}$。
* 重复以上步骤，直到策略收敛或达到最大迭代次数。

### 1.4 策略梯度实现技巧：基线 Baseline

在策略梯度方法中，添加基线是一个重要的技巧，用于减少方差，提高学习的稳定性和效率。这个技巧常见于各种强化学习算法中，尤其是 REINFORCE 类方法，和 Actor-Critic 方法中。

基线函数 $b(s)$ 通常是一个状态值函数或动作值函数，它与当前状态 $s_t$ 或动作 $a_t$ 相关。

至于为什么要使用基线，在策略梯度中，计算期望奖励的梯度如下：

$$
\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau) \nabla \log p_{\theta}(\tau)] \tag{6}
$$

其中，$R_{\tau}$ 是轨迹的总奖励。在实际计算中，精确的期望奖励是无法获取的，我们只能通过 Monte Carlo 方法估计：

$$
\hat{R}(\tau) = \sum_{t=1}^{T} r_t \tag{14}
$$

但是奖励的方差可能非常大，尤其是在高维环境中，这样的计算方式可能会导致一些好的但没被采样到的动作被低估，一些坏的动作被高估。因此，引入基线 $b(s)$ 可以减少方差，提高学习的稳定性和效率。

**带基线的策略梯度估计**

带基线的策略梯度估计如下：

$$
\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[(R(\tau) - b(s_t)) \nabla \log p_{\theta}(\tau)] \tag{15}
$$

这里表示奖励减去基线后，再与梯度相乘。这里的 $b(s_t)$ 通常是对状态的估计，不依赖于动作，因此其通常是某种值函数，比如状态值函数。

### 1.5 策略梯度实现技巧：优势函数 Advantage Function

在策略梯度方法中，引入优势函数是另一个重要的技巧，用于减少方差，提高学习的稳定性和效率。优势函数是奖励函数和基线函数的差值，表示当前状态下采取动作的优势。

**优势函数的定义**

优势函数 $A(s_t, a_t)$ 表示在状态 $s_t$ 下采取动作 $a_t$ 的优势，通常定义为动作值函数和状态值函数的差值：

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \tag{16}
$$

其中，$Q(s_t, a_t)$ 是动作值函数，表示在状态 $s_t$ 下采取动作 $a_t$ 得到的期望回报；$V(s_t)$ 是状态值函数，表示在状态 $s_t$ 下的期望回报。

所以优势函数描述了一个特定动作 $a_t$ 相对于当前状态 $s_t$ 的优势。

**带优势函数的策略梯度估计**

带优势函数的策略梯度估计如下：

$$
\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[A(s_t, a_t) \nabla \log p_{\theta}(\tau)]
$$

**优势函数的计算**

优势函数的计算通常是通过时间差分方法来近似，比如：

$$
A(s_t, a_t) = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{17}
$$

其中，

* $\delta_t$ 是时间差分误差，用于衡量当前的奖励加上下一状态的估计值与当前状态值之间的差异
* $r_t$ 是当前时刻的奖励，$V(s_{t+1})$ 是下一时刻的状态值函数，$\gamma$ 是折扣因子

这种计算方式的好处是，它能够通过近似计算减少优势函数的估计方差，避免直接计算每个动作的回报。

## 2. 近端策略优化 PPO

### 2.1 同策略与异策略 On Policy and Off Policy

在策略梯度方法中，使用当前的策略网络 $\pi_{\theta}$ 选择动作完成交互采样和参数更新，这种方法称为同策略（On Policy）方法。同策略方法的核心思想是，策略网络 $\pi_{\theta}$ 在训练过程中进行采样，并且在更新策略时，使用的采样数据是由当前（未更新）策略生成的，换言之，数据采样和更新都依赖于相同的策略。

> 策略梯度方法的使用 $\pi_{\theta}$ 与环境进行交互，若参数从 $\theta$ 更新到 $\theta'$ 后，概率 $p_{\theta}$ 会随之改变，那么之前采集到的数据无论好坏，都无法继续使用，因此需要进行新的采样，在采样上花费的时间和资源较多。

异策略方法的核心思想是，采样和策略更新可以使用不同的策略。换言之，采样过程使用一个策略，更新过程使用另一个策略。异策略方法允许使用过去的经验执行多次策略更新，显著提高了采样效率。

>如果从同策略变为异策略，即使用若干 $\theta'$ 训练 $\theta$ ，多次使用 $\theta'$ 采样到的数据，多次执行梯度上升，更新参数。

>$\theta'$ 只采样一次，但采样多一点的数据，让 $\theta$ 进行多次更新。与环境交互的是 $\theta'$ ，与 $\theta$ 没有关系。因此可以通过 $\theta'$ 采集大量的数据给 $\theta$ 进行多次参数更新，直到 $\theta$ 训练到一定程度之后，$\theta'$ 再重新做采样，这是同策略变成异策略的好处。

因此我们定义，如果采样和更新使用的是同一个策略，称为同策略（On Policy）方法；如果采样和更新使用的是不同的策略，称为异策略（Off Policy）方法。

### 2.2 重要性采样 Importance Sampling

在异策略强化学习中，采样数据来自不同的策略，比如使用策略参数 $\theta'$ 采样数据，然后使用策略参数 $\theta$ 更新参数。
由于 $\theta'$ 和 $\theta$ 可能不是同分布，数据采样时的策略和更新时的策略可能会导致期望和方差的差异。
因此重要性采样（Importance Sampling）被用来解决这一问题，通过采集足够多次的采样，使两者的期望和方差接近。

**重要性采样的基本原理**

重要性采样的核心思想是对从 $\theta'$ 采样的轨迹数据进行加权，使其能够反映策略 $\theta$ 的期望。为了做到这一点，需要定义一个重要性权重。如果有一条轨迹 $\tau$，其概率分布为 $p_{\theta'}(\tau)$，也就是采样自策略 $\theta'$，那么在更新目标策略 $\theta$ 时，需要对轨迹的概率分布进行加权，来修正策略之间的分布差异，这个权重表示为：

$$
w(\tau) = \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} \tag{18}
$$

其中，$p_{\theta}(\tau)$ 是轨迹 $\tau$ 在策略 $\theta$ 下的概率，$p_{\theta'}(\tau)$ 是轨迹 $\tau$ 在策略 $\theta'$ 下的概率。比值 $w(\tau)$ 称为重要性采样权重。

**重要性采样的计算过程**

使用重要性采样后，计算目标策略 $\theta$ 的期望奖励时，需要对轨迹的奖励进行加权，即：

$$
\bar{R}_{\theta} = \sum_{\tau} R(\tau) p_{\theta}(\tau) = \sum_{\tau} R(\tau) \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} p_{\theta'}(\tau) = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}[R(\tau) w(\tau)] \approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) w(\tau^n) \tag{19}
$$

其中，$R(\tau)$ 是轨迹 $\tau$ 的奖励，$p_{\theta}(\tau)$ 是轨迹 $\tau$ 在策略 $\theta$ 下的概率，$p_{\theta'}(\tau)$ 是轨迹 $\tau$ 在策略 $\theta'$ 下的概率，$w(\tau)$ 是重要性采样权重。

注意最后一个等号使用的是近似等于，表示我们用有限个样本来近似真实的期望，因而是一个 Monte Carlo 估计。随着 $N \rightarrow \inf$ 该估计会收敛到真实期望值，这是大数定律和蒙特卡洛方法的基本原理。在实践中，往往只能采样有限的 $N$ 条轨迹，所以只能获得一个近似值，而非精确值。

**重要性采样的挑战**

* 方差问题：重要性采样的一个挑战是，当重要性采样权重 $w(\tau)$ 的方差很大时，会导致估计的期望奖励不稳定。因此，需要通过多次采样来减小方差，提高估计的准确性。
* 效率低下：当两者之间的差异很大时，即使通过加权修正，采样数据也可能无法有效地表示目标策略的期望，这会导致需要更多的样本来有效地训练策略。

**重要性采样的应用**

重要性采样是异策略强化学习中的一个重要技术，用于解决不同策略之间的分布差异。在异策略方法中，重要性采样可以帮助我们有效地利用历史数据，提高采样效率，减少采样成本。

之前的方法用策略 $\pi_{\theta}$ 采样数据，现在用 $\pi_{\theta'}$ 采样数据，用 $\pi_{\theta}$ 更新参数。$\pi_{\theta'}$ 的身份是另一个演员，其工作是做示范。

$$
\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[ \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_{\theta}(\tau) \right] \tag{20}
$$

### 2.3 近端策略优化 Proximal Policy Optimization

> PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, trying to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small.

通过重要性采样，可以将同策略变为异策略，但重要性采样要求 $p_{\theta}(s_t|a_t)$ 与 $p_{\theta'}(s_t|a_t)$ 之间的差异不要太大，否则会导致方差过大，训练不稳定。
为了解决这个问题， Proximal Policy Optimization (PPO) 提出了一种近端策略优化的方法，通过引入一个约束项，来限制两个策略之间的差异。

#### 2.3.1 PPO的前身信任域策略优化 Trust Region Policy Optimization

PPO的前身是 TRPO(Trust Region Policy Optimization)，TRPO 是一种基于约束的优化方法，通过引入一个 KL 散度约束，来限制两个策略之间的差异。

TRPO 的优化目标可以表示为：

$$
J_{\text{TRPO}}^{\theta'}(\theta) = {\mathbb E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)} A^{\theta'}(s_t, a_t) \right], KL(\theta, \theta')<\delta  \tag{21}
$$

将 KL 散度作为约束， 使得有约束问题求解起来比较麻烦， PPO对此做了改进。

#### 2.3.2 PPO的核心思想

PPO 中用于采样数据的策略参数 $\theta'$ 为 $\theta_{\text{old}}$ , 即行为策略也是 $\pi_{\theta}$ ， 因此 PPO 是一个同策略方法。

PPO优化有两个目标：

一项是优化 $J^{\theta'}(\theta)$ ，这是使用重要性采样时，要去优化的目标函数：

$$
J^{\theta'}(\theta)={\mathbb E}_{(s_t,a_t)\sim\pi_{\theta'}}
\left[\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}A^{\theta'}(s_t,a_t)\right] \tag{22}
$$

另一项是约束 $\theta$ 与 $\theta'$ 的差异：

$$
J^{\theta'}_{PPO}(\theta)=J^{\theta'}(\theta)-\beta\text{KL}(\theta,\theta') \tag{23}
$$

这项约束 $\theta$ 和 $\theta'$ 的KL散度（KL divergence），用于衡量 $\theta$ 和 $\theta'$ 的相似程度，用以保证两者相似，否则结果不佳。

KL散度指计算 $\theta$ 和 $\theta'$ 的行为距离，给定同样的状态，输出动作之间的差距。

#### 2.3.3 近端策略优化惩罚 KL PPO1

首先初始化一个策略参数 $\theta^0$。在每一个迭代中，用前一个训练迭代得到的演员的参数 $\theta^k$ 与环境交互， 采集大量 $s_t, a_t$ 数据，根据交互的结果估算 $A^{\theta^k}(s_t, a_t)$。

与策略梯度方法不同，策略梯度采集到的数据只能用作一次更新，下一次的数据需要重新采集。而 PPO 采集到的数据可以让 $\theta$ 用于多次更新。

另外在论文中还提到了自适应 KL 散度，暂时不做讨论。

PPO1 的优化目标为：

$$
J_{\text{PPO1}}^{\theta^k}(\theta)=J^{\theta^k}(\theta)-\beta\text{KL}(\theta,\theta^k) \tag{24}
$$

$$
J^{\theta^k}(\theta) \approx \sum_{(s_t,a_t)}\frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}A^{\theta^k}(s_t,a_t) \tag{25}
$$

**优势函数计算**

传统的优势函数计算是通过动作值函数与状态值函数的差值来计算。

#### 2.3.4 近端策略优化裁剪 Clipped Surrogate Objective PPO2

由于计算 KL 散度复杂，提出裁剪方法，优化目标表示为：

$$
J_{PPO2}^{\theta^k}\approx\sum_{(s_t,a_t)}\min\left[
\frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}A^{\theta^k}(s_t,a_t), \text{clip}\left( 
	\frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)},1-\epsilon,1+\epsilon
	\right)A^{\theta^k}(s_t,a_t)
\right] \tag{26}
$$

$\epsilon$是一个超参数。

**优势函数计算**

在 PPO2 中，优势函数的计算是通过时间差分误差来近似计算。

GAE (Generalized Advantage Estimation) 是一种计算优势函数的方法，它通过时间差分误差来近似计算优势函数，可以有效地减小方差，提高学习的稳定性和效率。

GAE 通过引入一个折扣因子 $\lambda$ 来平衡偏差和方差：

$$
A_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l} \tag{27}
$$

其中，$\delta_t$ 是时间差分误差 (TD error 见 式17)， $\gamma$ 是折扣因子，$\lambda$ 是 GAE 的超参数。

GAE 的作用是通过动态平衡未来奖励的影响和当前奖励的影响，从而减少计算优势函数时的方差，使得训练更为稳定。

计算流程如下：

* 收集回报：首先，使用当前策略与环境进行交互，收集一批数据（状态、动作、奖励、下一状态等）。
* 计算TD error：对于每个状态 $s_t$ , 使用下一个状态的值函数估计 $V(s_{t+1})$ , 计算时间差分误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
* 计算优势函数：使用时间差分误差 $\delta_t$ 计算 GAE 优势函数 $A_t^{GAE}$。

**值函数计算**

值函数的是通过 Critic 网络来计算的，包括当前状态的值函数 $V(s_t)$ 和下一个状态的值函数 $V(s_{t+1})$。

#### 2.3.5 PPO的训练流程

**一些定义**

* rollout 是指智能体与环境交互的过程，它产生一个包含状态、动作、奖励、下一状态等信息的轨迹（或时间步序列）。智能体通过当前策略与环境交互并收集数据，这些数据称为回滚数据。每个回滚（或称为轨迹）包含多个时间步数据，从智能体的初始状态到最终状态。
* batch size 批量大小；指的是每次训练更新时使用的时间步的数据，使用 batch size 个时间步数据来进行一次优化。
* epoch 迭代次数；指对 rollout data 进行多次优化的次数；在每个 epoch 中，整个 rollout data 或部分 rollout data 会被用来进行多次优化。增加 epoch 可以提高训练的效果，但也会增加训练时间。
* mini-batch size 是每次从回滚数据中取出的训练样本数量。在训练时，批量数据会被拆分成多个小批量，分别用于更新模型。例如，如果 `batch_size = 1024` 且 `mini_batch_size = 256` ，则每次从1024个时间步中拆分出4个256大小的小批量， 用于分别更新策略。在每个epoch中，回滚数据被拆分成多个小批量，每个小批量通过梯度下降进行参数更新。这样做有助于加速训练并避免过拟合。Mini-batch的使用可以提高训练的稳定性和效率，通常会使用小批量随机梯度下降（mini-batch SGD）来进行优化。

这些参数之间的关系是：

* Rollout 数据是从环境中收集的，用于训练策略。在PPO中，每次从环境中收集一批数据（回滚数据），并将其存储在回滚缓冲区。
* 当回滚数据达到设定的 batch size（例如 1024 时间步）时，开始进入优化阶段。整个数据集（回滚数据）将被分成多个 mini-batches，每个 mini-batch 会独立进行优化。
* Epoch 的次数决定了每个回滚数据集被优化的次数。如果设置了 10 次 epoch，那么每次收集的数据会被用来进行 10 次独立的优化步骤。
* Mini-batch size 决定了每个小批量的样本数。数据通过mini-batch的方式分成更小的块，每个块分别用于优化，从而实现更加高效和稳定的训练。

假设我们设置了以下参数：

* batch_size = 1024：每次训练时使用 1024 个时间步的数据。
* epochs = 10：每个回滚数据将被用于 10 次优化。
* mini_batch_size = 256：每次更新时使用 256 个时间步的数据。

那么，训练流程会是：

* 从环境中收集 1024 个时间步的数据（回滚数据）。
* 将这 1024 个时间步的数据分成 4 个 mini-batches（每个 mini-batch 256 个时间步）。
* 在每次优化时，PPO 会对每个 mini-batch 进行独立的优化步骤。
* 整个数据集（1024个时间步的数据）将会被用于 10 次优化（即10次epoch）。

**训练流程**

a) rollout data collection

b) GAE calculation

c) PPO optimization (update actor and critic networks)

* Actor和Critic网络通过反向传播在mini-batches中更新。Actor输出策略（动作概率），而Critic则估计状态值函数 $V(s_t)$ 。
* Actor 更新：基于目标函数（惩罚或剪切）和GAE，最小化损失函数。确保更新不会偏离当前策略太远，保持稳定性。
* Critic更新：Critic的损失是均方误差（MSE），即估计值和目标值之间的差异。目标值通常通过GAE和折扣奖励来计算。
  d) policy update
* mini-batch训练：在收集足够的数据后，训练过程包括多次遍历收集到的数据。在每个epoch中，将回滚数据划分为mini-batches，每个mini-batch上进行多次优化步骤，使用**随机梯度下降（SGD）**来更新策略参数。
* 剪切和目标KL散度：PPO通常会添加KL散度约束，用来限制新旧策略之间的差异。如果KL散度超过设定阈值，则提前停止训练以防止策略更新过大。这个提前停止对避免不稳定非常重要。
* 计算损失和梯度后，Actor和Critic网络的参数通过Adam优化器（或其他指定优化器）进行更新。
  e) repeat from a)

除此之外，还包含了一些其他的技巧，比如：

* Entropy Regularization 熵正则化，熵被用作正则化项来鼓励策略的探索性。目标函数中会加入熵损失，防止策略过早收敛到次优解。
* Learning Rate Scheduling 学习率调度动态调整学习率，通常学习率在训练初期较高，随着训练进程逐渐减小。这允许在初期进行大步更新，而在训练后期进行细致的调整。

#### 2.3.6 critic网络的更新与MSE计算

在PPO中，**Critic网络** 主要负责估计状态值函数 $ V(s_t) $，并通过 **均方误差（MSE）** 来训练其参数。Critic网络的目标是最小化预测值和目标值之间的差异，从而准确地估算每个状态的值。

**Critic的损失函数**

Critic网络的损失函数基于 **均方误差（MSE）**，计算的是 **网络预测的值函数** $ V_{\theta}(s_t) $ 与 **目标值** $ V_{\text{target}}(s_t) $ 之间的差异。这个目标值是通过 **时间差分（TD）误差** 或 **广义优势估计（GAE）** 来计算的。

Critic的损失函数如下所示：

$$
L_{\text{critic}} = \mathbb{E}_t \left[ \left( V_{\theta}(s_t) - V_{\text{target}}(s_t) \right)^2 \right]
$$

- $ V_{\theta}(s_t) $ 是 Critic网络对当前状态 $ s_t $ 的预测值。

- $ V_{\text{target}}(s_t) $ 是由 **时间差分（TD）误差** 或 **GAE** 计算得到的目标值，通常形式为：

  $$
  V_{\text{target}}(s_t) = r_t + \gamma V(s_{t+1})
  $$

**如何计算目标值** $ V_{\text{target}}(s_t) $

**目标值** $ V_{\text{target}}(s_t) $ 的计算由 **TD误差** 或 **GAE** 给出，代表了 **当前状态** 的值函数预测，结合 **当前奖励** 和 **下一状态的值函数预测**。在PPO中，目标值通常是基于下一状态的值函数预测 $ V(s_{t+1}) $ 和即时奖励 $ r_t $ 来计算的：

$$
V_{\text{target}}(s_t) = r_t + \gamma V(s_{t+1})
$$

这意味着目标值不仅依赖于当前状态的估算值 $ V(s_t) $，还依赖于 **Critic网络对下一状态的预测值 $ V(s_{t+1}) $**。

**Critic网络的更新过程**

在每次训练中，Critic网络使用 **均方误差（MSE）损失** 来优化其参数。损失函数计算的是网络对 **当前状态值** 和 **目标值** 之间的误差。

1. **收集数据**：通过与环境交互，收集回滚数据（状态、动作、奖励、下一状态）。
2. **计算TD误差或GAE**：通过计算每个状态的 **TD误差** 或 **GAE**，得到目标值 $ V_{\text{target}}(s_t) $。
3. **计算损失函数**：使用 **均方误差** 计算预测值 $ V_{\theta}(s_t) $ 和目标值 $ V_{\text{target}}(s_t) $ 之间的差异。
4. **反向传播优化网络**：通过反向传播更新Critic网络的参数，使其预测值 $ V_{\theta}(s_t) $ 更加接近目标值 $ V_{\text{target}}(s_t) $。

**最终目标**

Critic网络的最终目标是通过最小化 **MSE损失**，使得其输出的 **状态值函数预测** 趋近于实际的 **目标值**。这样一来，Critic网络就能精确地为 **Actor网络** 提供反馈，推动 **PPO** 策略的优化过程。

**小结**

- **Critic网络** 通过计算 **均方误差（MSE）** 来优化状态值函数的预测，损失函数计算的是预测值和目标值之间的误差。
- **目标值** $ V_{\text{target}}(s_t) $ 由 **TD误差** 或 **GAE** 计算得到，通常结合了当前的奖励和下一状态的预测值。
- 通过最小化损失函数，Critic网络不断改进其值函数估计，从而为 **Actor网络** 提供更准确的反馈，保证 **PPO** 策略的稳定更新。

#### 2.3.7 on-policy or off-policy for PPO?

PPO 到底是 on-policy 还是 off-policy 呢？

根据定义，用于采样的策略和用于更新的策略是同一个策略，则为同策略。

但是 PPO 比较 mixing。

* `n_epochs` 为 1 的话，那确实采样和用来更新的完全是一个策略；但一旦 `n_epochs` 大于 1 ，那么很明显，从第二次更新开始，用于采样的和用来更新的已经不是一个策略了。
* 但是由于 PPO 的重要性采样和 clip 机制，限制了策略的更新幅度，从这个角度将其认为成 on-policy 也可以，但是严格意义上只要不一样还是可以叫做 off-policy，所以对 PPO 讨论是 on or off 有些没意义。
