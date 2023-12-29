---
title: Spinning Up
---

# [Spinning Up](https://spinningup.openai.com/en/latest/index.html)

---

## [Part 1](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

In RL, *agents* act within an *environment*. *Rewards* are recieved for how good
the environment state is. The goal of RL is to maximize the total sum of
rewards, called the return.

### Definitons:

- **States/observations:**
    - A *state* is the full information of the world the agent is in
    - The agent gets to see an *obervation* of the state
    - This observation might exclude some information
    - Environments where information is excluded in the observation are
      *partially-observed*, other states are *fully-observed*
- **Action Spaces:**
    - *Action spaces* are the set of all valid actions within an environment
    - There are discrete and continuous action spaces
        - E.g. chess vs driving
- **Policies:**
    - A *policy* is the algorithm an agent uses for chosing an action in a state
    - There are *deterministic* and *stochastic* policies:
    - In deep RL, policies are parameterized ($\theta$) using a neural network
    - Deterministic policies:
        - Argmax of the output of a neural network
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
    - Rewards over a trajectory can be *finite* or *infinite* horzion and
      *discounted* or *un-discounted*
    - Finite-horzion undiscounted return: $R(\tau) = \sum^T_{t=0}r_t$
    - Infinite-horizon discounted return: $R(\tau) = \sum^\infty_{t=0} \gamma^t
      r_t, \quad \gamma \in (0, 1)$
