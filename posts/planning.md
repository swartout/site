---
title: Napkin Math on the Power of Planning in ML (Unfinished)
---

OUTLINE:

- introduce current paradigm (expensive train, cheap inference)
    - introduce noam brown uw talk
    - compare between language models and traditional game planning
        - language models are one expensive training, then cheap inference
            - similar to software
        - "solving" a game has been a goal, figure out what the perfect solution is in advance
- introduce test-train scaling tradeoff
    - game playing agents need to use planning
        - support:
            - alphago
            - deep blue
            - pleurbus (figure out how to spell lol)
    - keep anthropic "scaling scaling laws" included
        - include plot
        - 10x train-time compute ~ 15x test-time compute
    - introduce o1, cot, other sampling method
- give analogy of distributing total compute over moves/positions
    - caveats:
        - general knowledge is necessary!
        - training is not distributed evenly across entire state space
    - chess:
        - estimate one min per move
            - ~70e10 flops
    - language, e.g. fixed length, llama flops training vs generation vs state space

- references:
    - noam brown uw video: https://youtu.be/eaAonE58sLU?si=9OI_iN5VCQ4PNhJu
    - deep blue: https://www.ibm.com/history/deep-blue, 1.138e10 flops/sec
        - 45, 48, 56, 49, 19
        - 90 min + 30 sec increments
    - llama 3: https://arxiv.org/abs/2407.21783, 405b has 3.8e25 flops
        - ~1e12 inference FLOPs
    - number of chess positions: https://en.wikipedia.org/wiki/Shannon_number

*Much of this comes from and is inspired by a UW [talk on planning from Noam Brown](https://youtu.be/eaAonE58sLU?si=9OI_iN5VCQ4PNhJu), check it out!*

Currently, models are trailed like software is created - a large upfront cost, followed by a relativly low production (inference) cost. Llama 3 405B was trained on about $3.8e25$ FLOPs, likely hundreds of millions of dollars of compute, while inference can be run for pennies. However, we're starting to see this change, and games give us an interesting example where we can do some back-of-the-napkin math to give us intuition. Solving complex games like go and chess have long been a goal of game theorists - can we find the outcome of chess under perfect play? Solving a game involves a large upfront cost: making a plan over the entire game, for every possible situation. However, following that plan in a game is relativly simple, similar to the train-test "effort" tradeoff for software and LLMs.

Chess and go have also seen two of the greatest success stories of AI: through Deep Blue in 1997 and AlphaGo in 2016 beating world champion players. Even though AI has far surpassed human level, these games are far from being solved. These are not successes simply from ML (Deep Blue didn't use ML at all!), but successes in planning. The paper [Scaling Scaling Laws with Board Games] shows the necessary increase in training compute for an amount of increasing test-time compute, roughtly for a $15$x of test-time compute, $10$x of train-time compute can be eliminated.

[TODO: include plot]

Using planning to scale test-time compute was a massive success in games as well as one we're starting to see in LLMs. Chain-of-thought (CoT) prompting, instructing the LLM to "plan" it's thoughts, is the baseline technique for getting good results out of LLMs. OpenAI's o1 model also uses CoT as well as other tricks to "think" over a solution. Both of these examples scale up test-time compute, while also scaling results.

A rough way of considering this train-test compute scaling is to consider on average, how much compute is used on a specific example or position. In other words, what the average amount of compute per example or position for train and test time is. This is not a perfect or precise metric, there are many caveats:

- Training is not evenly distributed over the whole space of possible strings (for language models) or sentences. It instead is focused on the tiny subset of "reasonable" examples.
- Likewise, not all strings or positions are "hard" to come up with a move. Many positions in chess are trivial as well as many next-token predictions, so training is focused on it.
- The values are very rough estimates 

While you should take the following with a grain of salt and several orders of magnitude, I still believe it is useful to consider the difference in train-test compute per example:

**Deep Blue:**

Deep Blue has a [reported](https://www.ibm.com/history/deep-blue) FLOPs count of $1.138e10$. Assuming full utilization of compute and 60 seconds per move, that's $1.138e10 \cdot 60 \simeq 7e11$ FLOPs per move.

Besides opening and endgame tables (I assume), Deep Blue did not precompute any

**Llama 3 405B:**
