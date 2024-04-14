#### Questions@:, Lecture 1 Questions
- Derive how to incrementally update a running average (how to compute
  an average of $N$ numbers using the average of the first $N-1$ numbers). [5]

- Describe multi-arm bandits and write down the $\epsilon$-greedy algorithm
  for solving it. [5]

- Define a Markov Decision Process, including the definition of a return. [5]

- Describe how does a partially observable Markov decision process extend the
  Markov decision process and how is the agent altered. [5]

- Define a value function, such that all expectations are over simple random
  variables (actions, states, rewards), not trajectories. [5]

- Define an action-value function, such that all expectations are over simple
  random variables (actions, states, rewards), not trajectories. [5]

- Express a value function using an action-value function, and express an
  action-value function using a value function. [5]

- Define optimal value function and optimal action-value function. Then define
  optimal policy in such a way that its existence is guaranteed. [5]

#### Questions@:, Lecture 2 Questions
- Write down the Bellman optimality equation. [5]

- Define the Bellman backup operator. [5]

- Write down the value iteration algorithm. [5]

- Define the supremum norm $||\cdot||_\infty$ and prove that Bellman backup
  operator is a contraction with respect to this norm. [10]

- Formulate and prove the policy improvement theorem. [10]

- Write down the policy iteration algorithm. [10]

- Write down the tabular Monte-Carlo on-policy every-visit $\epsilon$-soft algorithm. [10]

- Write down the Sarsa algorithm. [10]

- Write down the Q-learning algorithm. [10]

#### Questions@:, Lecture 3 Questions
- Elaborate on how can importance sampling estimate expectations with
  respect to $\pi$ based on samples of $b$. [5]

- Show how to estimate returns in the off-policy case, both with (1) ordinary
  importance sampling and (2) weighted importance sampling. [10]

- Write down the Expected Sarsa algorithm and show how to obtain
  Q-learning from it. [10]

- Write down the Double Q-learning algorithm. [10]

- Show the bootstrapped estimate of $n$-step return. [5]

- Write down the update in on-policy $n$-step Sarsa (assuming you already
  have $n$ previous steps, actions and rewards). [5]

- Write down the update in off-policy $n$-step Sarsa with importance
  sampling (assuming you already have $n$ previous steps, actions and rewards). [10]

- Write down the update of $n$-step Tree-backup algorithm (assuming you already
  have $n$ previous steps, actions and rewards). [10]

- Assuming function approximation, define Mean squared value error. [5]

- Write down the gradient Monte-Carlo on-policy every-visit $\epsilon$-soft algorithm. [10]

#### Questions@:, Lecture 4 Questions
- Write down the semi-gradient $\epsilon$-greedy Sarsa algorithm. [10]

- Prove that semi-gradient TD update is not an SGD update of any loss. [10]

- What are the three elements causing off-policy divergence with function
  approximation? Write down the Baird's counterexample. [10]

- Explain the role of a replay buffer in Deep Q Networks. [5]

- How is the target network used and updated in Deep Q Networks? [5]

- Explain how is reward clipping used in Deep Q Networks. What other
  clipping is used? [5]

- Formulate the loss used in Deep Q Networks. [5]

- Write down the Deep Q Networks training algorithm. [10]

- Explain the difference between DQN and Double DQN. [5]

- Describe prioritized replay (how are transitions sampled from the replay
  buffer, how up-to-date are the priorities [according to which we sample],
  how are unseen transitions boosted, how is importance sampling used to account
  for the change in the sampling distribution). [10]

- How is the action-value function computed in dueling networks? [5]

#### Questions@:, Lecture 5 Questions
- Describe a fully connected layer in Noisy nets (parametrization, computation,
  effective noise generation). [5]

- In Distributional RL, describe how is the distribution of rewards represented
  and how it is predicted using a neural network. [5]

- Write down the distributional Bellman equation, describe how are the atom
  probabilities of a reward distribution modeled, and write down the loss used
  to train a distributional Q network (including the mapping of atoms, which
  does not need to be mathematically flawless -- it is enough to describe how it
  should be done). [10]

- More questions about distributional RL to be added later.

#### Questions@:, Lecture 6 Questions
- Formulate the policy gradient theorem. [5]

- Prove the part of the policy gradient theorem showing the value
  of $\nabla_{\boldsymbol\theta} v_\pi(s)$. [10]

- Assuming the policy gradient theorem, formulate the loss used by the REINFORCE
  algorithm and show how can its gradient be expressed as an expectation
  over states and actions. [5]

- Write down the REINFORCE algorithm. [10]

- Show that introducing baseline does not influence validity of the policy
  gradient theorem. [5]

- Write down the REINFORCE with baseline algorithm. [10]

- Write down the trajectory formulation of the operator version of REINFORCE,
  and show that the usual REINFORCE performs one gradient step to minimize the
  same utility function. [10]

- Write down the state-action formulation of the operator version of REINFORCE,
  and show that the usual REINFORCE performs one gradient step to minimize the
  same utility function. [10]

- Write down the one-step Actor-critic algorithm. [10]

- How and why is entropy regularization used in policy gradient algorithms? [5]

- The Asynchronous advantage actor-critic (A3C) policy may utilize recurrent
  neural networks. How is the training structured to allow backpropagation
  through them (would vanilla DQN, vanilla REINFORCE, vanilla actor-critic work
  with recurrent neural networks)? [5]

#### Questions@:, Lecture 8 Questions
- Explain the difference between a regular Actor-critic and Parallel Advantage
  Actor Critic algorithms. [5]

- Considering continuous actions modeled by a normal distribution with
  diagonal covariance, describe how is the policy distribution computed
  (network architecture, output activation functions) and how does the loss of
  a simple REINFORCE algorithm look like. [5]

- Formulate the deterministic policy gradient theorem for
  $\nabla_{\boldsymbol\theta} v_\pi(s)$. [5]

- Formulate the deterministic policy gradient theorem for
  $\nabla_{\boldsymbol\theta} J(\boldsymbol\theta)$. [5]

- Prove the part of the deterministic policy gradient theorem showing the value
  of $\nabla_{\boldsymbol\theta} v_\pi(s)$. [10]

- Write down the critic loss (or its derivative) and the actor policy loss (or
  its derivative) of the Deep Deterministic Policy Gradients (DDPG) algorithm. Make
  sure to distinguish the target networks from the ones being trained. [10]

- How is the return estimated in the Twin Delayed Deep Deterministic Policy
  Gradient (TD3) algorithm? [5]

- Write down the critic loss (or its derivative) and the actor policy loss (or
  its derivative) of the Twin Delayed Deep Deterministic Policy Gradient (TD3)
  algorithm. Make sure to distinguish the target networks from the ones being
  trained. [10]

- Write down how is the reward augmented in Soft actor critic, and the
  definitions of the soft action-value function and the soft (state-)value function.
  Then, define the modified Bellman backup operator $\mathcal{T}_\pi$ (be sure
  to indicate whether you are using the augmented or non-augmented reward),
  whose repeated application converges to the soft actor-value function $q_\pi$,
  and prove it. [10]

- Considering soft policy improvement of a policy $\pi$, write down the update
  formula for the improved policy $\pi'$, and prove that the soft action-value
  function of the improved policy is greater or equal to the soft action-value
  function of the original policy. [10]

- Write down how are the critics and target critics updated in the Soft actor
  critic algorithm. [5]

- Write down how is the actor updated in the Soft actor critic algorithm,
  including the policy reparametrization trick. [5]

- Regarding the entropy penalty coefficient $\alpha$ in the Soft actor critic,
  define what constrained optimization problem we are solving, what is the
  corresponding Lagrangian (and whether we are minimizing/maximizing it
  with respect to the policy and $\alpha$), and what does the $\alpha$ update
  looks like. [5]

#### Questions@:, Lecture 9 Questions
- Define a one-step TD error and express the $n$-step return as a sum of them. [5]

- Define a one-step TD error and express the $n$-step return with off-policy
  correction using control variates as a sum of TD errors. [5]

- Define the $\lambda$-return. [5]

- Define the $n$-step truncated $\lambda$-return. [5]

- Define a one-step TD error and express the $n$-step truncated $\lambda$-return
  as a sum of them. [5]

- Define a one-step TD error and express the $n$-step truncated $\lambda$-return with
  off-policy correction as a sum of them. [5]

- Define the V-trace estimate and write down the policy to whose value function
  the V-trace estimate converges to. [10]

- Explain why the fixed point of the V-trace operator does not depend on the
  truncation of all but the last importance sampling ratios. [10]

- Write down the critic loss (or its derivative) and the actor policy loss (or
  its derivative) of the IMPALA algorithm, including the V-trace formula. [10]

- Sketch the population based training used in the IMPALA algorithm. [5]

- In PopArt normalization, the value function is computed based on a normalized
  value predictor $n$ as $\sigma n + \mu$. Describe how to maintain $\sigma$ and
  $\mu$, how to compute normalized advantage based on return $G$, and how is the
  normalized value predictor modified when the estimates of $\sigma$ and $\mu$
  change. [10]

- Define the transformed Bellman operator. [5]

- Define the transformed Bellman operator. Then, assuming $h$ is strictly
  monotonically increasing function and considering a deterministic Markov
  decision process, show to what does a transformed Bellman operator
  $\mathcal{T}_h$ converge and prove it. [10]

- Write down the return transformation used for Atari environments (for example
  by R2D2). [5]

- Describe the replay buffer elements in R2D2. What is the difference between
  the zero-state and stored-state strategies, and how is burn-in used? [5]

- Write down the Retrace operator and describe the three possibilities of
  setting the traces $c_t$: importance sampling, Tree-backup($\lambda$) and
  Retrace($\lambda$). [10]
