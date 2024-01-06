---
title: Spinning Up
---

# [Spinning Up](https://spinningup.openai.com/en/latest/index.html)

---

## [Part 1](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

In RL, *agents* act within an *environment*. *Rewards* are received for how good
the environment state is. The goal of RL is to maximize the total sum of
rewards, called the return.

### Definitions:

- **States/observations:**
    - A *state* is the full information of the world the agent is in
    - The agent gets to see an *observation* of the state
    - This observation might exclude some information
    - Environments where information is excluded in the observation are
      *partially-observed*, other states are *fully-observed*
- **Action Spaces:**
    - *Action spaces* are the set of all valid actions within an environment
    - There are discrete and continuous action spaces
        - E.g. chess vs driving
- **Policies:**
    - A *policy* is the algorithm an agent uses for choosing an action in a state
    - There are *deterministic* and *stochastic* policies:
    - In deep RL, policies are parameterized ($\theta$) using a neural network
    - Deterministic policies:
        - ArgMax of the output of a neural network
        - Usually denoted using $\mu$, as: $a_t = \mu_\theta (s_t)$
    - Stochastic policies:
        - Sample from the outputs of the network
        - Usually denoted using $\pi$, as: $a_t \sim \pi_\theta (\cdot | s_t)$
        - Categorical stochastic policies:
            - For discrete action spaces
            - Neural network with softmaxed outputs which are sampled from as a
              probability distribution
            - Log-likelihood is given by taking the log of a logit
        - Diagonal gaussian stochastic policies:
            - For continuous action spaces
            - Probability distributions are gaussians with covariance 0 between
              them
            - Therefore, the gaussians are parameterized by a mean vector $\mu$
              and diagonal of the covariance matrix, $\sigma$
            - Log standard deviations either are parameters or outputs of the
              neural network dependent upon state
                - Log ($\log \sigma$) is used as it can be $-\infty, \infty$
                  instead of $\sigma$ which must be non-negative
            - The mean action vector is always an output of the neural network
            - All together, these make up gaussians which can be sampled from
            - Log-likelihood: $\log \pi_\theta (a | s) = -\frac{1}{2}
              (\sum^k_{i=1}(\frac{(a_i - \mu_i)^2}{\sigma^2_i} + 2 \log
              \sigma_i) + k \log 2 \pi)$
- **Trajectories:**
    - A trajectory ($\tau$) is a history of states and actions taken in order
    - The first state is sampled from the start-state distribution ($s_0 \sim
      \rho_0(\cdot)$
    - After the start state, the next state is governed by the *state-transition
      function* based on only the current state and action taken
        - Only these are needed because of the Markov property
    - The state-transition function can be deterministic ($s_{t+1} = f(s_t,
      a_t)$) or stochastic ($s_{t+1} \sim P(\cdot | s_t, a_t)$)
    - Trajectories are also known as "episodes" or "rollouts"
- **Reward/return:**
    - *Reward* is dependent on *current state*, *current action*, and *next
      state*: $r_t = R(s_t, a_t, s_{t+1})$
        - It can often be simplified to drop the dependency on next state
          ($r_t = R(s_t, a_t)$) or action ($r_t = R(s_t)$)
    - Rewards over a trajectory can be *finite* or *infinite* horizon and
      *discounted* or *un-discounted*
    - Finite-horizon undiscounted return: $R(\tau) = \sum^T_{t=0}r_t$
    - Infinite-horizon discounted return: $R(\tau) = \sum^\infty_{t=0} \gamma^t
      r_t, \quad \gamma \in (0, 1)$
- **Expected return:**
    - Probability of a trajectory: $P(\tau | \pi) = \rho_0(s_0) \prod^{T-1}_{T=0}
      P(s_{t+1} | s_t, a_t) \pi (a_t | s_t)$
    - Expected return: $J(\pi) = \int_{\tau} P(\tau | \pi) R(\tau) =
      \mathbb{E}_{\tau \sim \pi} [R(\tau)]$
- **Value functions:**
    - Value functions measure how "good" a state or state action pair is
    - The inputs are either states or state-action pairs, they either follow the
      optimal policy or a given policy
    - On-policy value function: $V^\pi(s) = \mathbb{E}_{\tau \sim
      \pi}[R(\tau)|s_0 = s]$
    - On-policy action-value function: $Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi}
      [R(\tau) | s_0 = s, a_0 = a)]$
    - Optimal value function: $V^*(s) = \max_\pi \mathbb{E}_{\tau \sim \pi}
      [R(\tau) | s_0 = s]$
    - Optimal action-value function: $Q^*(s, a) = \max_\pi \mathbb{E}_{\tau \sim
      \pi} [R(\tau) | s_0 = s, a_0 = a]$
- **Optimal action in a state:** $a^*(s) = \arg \max_a Q^*(s, a)$
    - This is somewhat how Deep-Q networks work, by improving the Q-function
      via satisfying Bellman equations, ultimately using the function to find the
      best action in a state
- **Bellman equations:**
    - Self-consistency equations, formulating that the value of the current
      state is equal to the current reward, plus the value of your next state
    - On-policy Bellman equations:
        - $V^\pi(s) = \mathbb{E}_{a \sim \pi, s' \sim P}[r(a, s) + \gamma V^\pi
          (s')]$
        - $Q^\pi(s, a) = \mathbb{E}_{s' \sim P}[r(s, a) + \mathbb{E}_{a' \sim
          \pi}[Q^\pi(s', a')]$
    - Optimal Bellman equations:
        - $V^*(s) = \max_a \mathbb{E}_{s' \sim P}[r(a, s) + \gamma V^\pi(s')]$
        - $Q^*(s, a) = \mathbb{E}_{s' \sim P}[r(s, a) + \gamma \max_{a'} Q^*(s',
          a')]$
    - Main difference is that optimal functions must pick the best action, where
      on-policy takes the expected action via the policy
    - The "Bellman Backup" for a state (action pair) is the right side of the
      Bellman equations
- **Advantage functions:**
    - *Advantage functions* measure how good an action is relative to other
      actions in the state
    - "How much better is it to take a certain action vs. the expectation action
      from the policy"
    - $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$
- **Markov Decision Processes (MDPs):**
    - Formal way of describing a setting/problem
    - Shows the Markov property, the future is independent of the past (besides
      current state/action)
    - MDPs are 5-tuples: $\langle S, A, R, P, \rho_0 \rangle$
        - $S$: set of valid states
        - $A$: set of valid actions
        - $R$: reward function, $r_t = R(s_t, a_t, s_{t+1})$
        - $P$: state transition function, $s' = P(s'|s,a)$
        - $\rho_0$: starting state distribution

The goal of RL is to maximize the expected return over a trajectory from agent
actions given by a policy. This can be expressed as finding the optimal policy,
$\pi^*$, where $\pi^* = \arg \max_\pi J(\pi)$

## [Part 2](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

### Model-Based RL

One of the biggest differentiators in RL is if the algorithm models the
environement.

- Model-based RL can either learn or be given a model which predicts state
transitions and rewards
- This is advantageous because it allows the agent to
search and plan ahead
- These models can either be learned or given:
- **Learned models:**
    - Learning a model can be difficult, due to bias the agent being able to
      exploit the biases in the model
- **Given models:**
    - This is common for games, such as go or chess

### What to Learn

- **Explicit Policies:**
    - They learn a function $\pi_\theta(a|s)$
    - Optimization is (almost) always on-policy (must use data collected from
      the most recent version of the policy)
    - Often need to learn approximator for the value function $V^\pi(s)$
    - A2C, PPO do this
- **Q-Learning:**
    - Learn a Q-function by satisfying the Bellman equation
    - Off-policy, can use old data
    - Actions are decided via the action which has the highest value for the
      Q-function
    - Deep-Q-networks use this!
- **Model-based:**
    - Pure planning involves not representing the policy, instead use search to
      select best actions
    - Expert iteration involves search along with a policy, which generates
      possible actions to search from

## [Part 3](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

**Deriving policy gradients**

*Five helpful facts*

1. Prob of trajectory: $$P(\tau | \theta) = \rho_0(s_0) \prod_{t=0}^T P(s_{t+1}
   | s_t, a_t) \pi_\theta(a_t | s_t)$$
2. Log-derivative trick: $$\nabla_\theta P(\tau | \theta) = P(\tau | \theta)
   \nabla_\theta \log P(\tau | \theta)$$
3. Log-prob of a trajectory: $$\log P(\tau | \theta) = \log \rho_0(s_0) +
   \sum_{t=0}^T [\log P(s_{t+1} | s_t, a_t) + \log \pi_\theta(a_t | s_t)]$$
4. Gradients of env functions: $$\rho_0(s_0) = P(s_{t+1} | s_t, a_t) = R(\tau)
   = 0$$
5. Grad-log-prob of trajectory: $$\nabla_\theta \log P(\tau | \theta) =
   \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t | s_t)$$

**Basic policy gradient:**

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T
\nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau)]$$

It can be estimated via: $\nabla_\theta J(\pi_\theta) \sim \frac{1}{|D|}
\sum_{\tau \in D} \sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t | s_t)
R(\tau)$

*"Expected Grad-Log-Prob Lemma":*

$$\mathbb{E}_{x \sim P_\theta}[\nabla_\theta \log P_\theta(x)] = 0$$

The simple gradient includes rewards earned before the action was taken, which
doesn't make sense. We can equivalently express the gradient as:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T
\nabla_\theta \log \pi_\theta (a_t | s_t) \sum_{t' = t}^T R(s_{t'}, a_{t'},
s_{t'+1})]$$

The "reward-to-go" is $\hat{R}_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})$

Because of EGLP, we can multiply by functions $b(s_t)$ which only depend on
state:

$$\mathbb{E}_{a_t \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a_t | s_t)
b(s_t)] = 0$$

Therefore, we can add or subtract terms from the "reward" part of the policy
gradient without changing the expectation. We can subtract (our prediction of)
the value function from the reward ($b(s_t) = V^\pi(s_t)$).

As we don't actually know $V^\pi(s_t)$, we will typically estimate it via
another neural network. The neural network typically minimizes a mean-squared
loss function:

$$V_\phi = \arg \min_{V_\phi} \mathbb{E}_{s_t, \hat{R}_t, \sim
\pi_k}[(V_\phi(s_t) - \hat{R}_t)^2]$$

Note that the states and rewards are from the current epoch, $k$.

The *general form* of the policy gradient is:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T
\nabla_\theta \log \pi_\theta(a_t | s_t) \Phi_t]$$

$\Phi_t$ can be any of the following and still have the same expectation:

- $\Phi_t = R(\tau)$
- $\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})$
- $\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)$
- $\Phi_t = Q^{\pi_\theta}(s_t, a_t)$: (on-policy action-value function)
- $\Phi_t = A^{\ph_\theta}(s_t, a_t)$: (advantage function)

