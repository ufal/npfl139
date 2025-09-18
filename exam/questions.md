#### Questions@:, Lecture 1 Questions
- Derive how to incrementally update a running average (how to compute
  an average of $N$ numbers using the average of the first $N-1$ numbers). [5]

- Describe multi-arm bandits and write down the $\epsilon$-greedy algorithm
  for solving it. [5]

- Define a Markov Decision Process, including the definition of a return. [5]

- Describe how a partially observable Markov decision process extends a
  Markov decision process and how the agent is altered. [5]

- Define a value function, such that all expectations are over simple random
  variables (actions, states, rewards), not trajectories. [5]

- Define an action-value function, such that all expectations are over simple
  random variables (actions, states, rewards), not trajectories. [5]

- Express a value function using an action-value function, and express an
  action-value function using a value function. [5]

- Define optimal value function, optimal action-value function, and the
  optimal policy. [5]

#### Questions@:, Lecture 2 Questions
- Write down the Bellman optimality equation. [5]

- Define the Bellman backup operator. [5]

- Write down the value iteration algorithm. [5]

- Define the supremum norm $||\cdot||_\infty$ and prove that Bellman backup
  operator is a contraction with respect to this norm. [10]

- Formulate and prove the policy improvement theorem. [10]

- Write down the policy iteration algorithm. [10]

- Write down the tabular Monte-Carlo on-policy every-visit $\epsilon$-soft algorithm. [5]

- Write down the Sarsa algorithm. [5]

- Write down the Q-learning algorithm. [5]

#### Questions@:, Lecture 3 Questions
- Elaborate on how can importance sampling estimate expectations with
  respect to $\pi$ based on samples of $b$. [5]

- Show how to estimate returns in the off-policy case, both with (a) ordinary
  importance sampling and (b) weighted importance sampling. [10]

- Write down the Expected Sarsa algorithm and show how to obtain
  Q-learning from it. [10]

- Write down the Double Q-learning algorithm. [10]

- Show the bootstrapped estimate of $n$-step return. [5]

- Write down the update in on-policy $n$-step Sarsa (assuming you already
  have $n$ previous steps, actions, and rewards). [5]

- Write down the update in off-policy $n$-step Sarsa with importance
  sampling (assuming you already have $n$ previous steps, actions, and rewards). [10]

- Write down the update of $n$-step Tree-backup algorithm (assuming you already
  have $n$ previous steps, actions, and rewards). [10]

- Assuming function approximation, define Mean squared value error. [5]

- Write down the gradient Monte-Carlo on-policy every-visit $\epsilon$-soft algorithm. [10]

#### Questions@:, Lecture 4 Questions
- Write down the semi-gradient $\epsilon$-greedy Sarsa algorithm. [10]

- Prove that semi-gradient TD update is not an SGD update of any loss. [10]

- What are the three elements causing off-policy divergence with function
  approximation? Write down the Baird's counterexample. [10]

- Explain the role of a replay buffer in Deep Q Networks and describe how
  a single element of a replay buffer looks like. [5]

- How is the target network used and updated in Deep Q Networks? [5]

- Explain how is reward clipping used in Deep Q Networks. What other
  clipping is used? [5]

- Formulate the loss used in Deep Q Networks. [5]

- Write down the Deep Q Networks training algorithm. [10]

- Explain the difference between DQN and Double DQN, and between Double DQN
  and Double Q-learning. [5]

- Describe prioritized replay (how are transitions sampled from the replay
  buffer, how up-to-date are the priorities [according to which we sample],
  how are unseen transitions boosted, how is importance sampling used to account
  for the change in the sampling distribution). [10]

- Describe a data structure that can be used to implement prioritized replay
  buffer, so that it has given maximum capacity and insertion and sampling runs
  in time logarithmic with respect to the maximum number of elements. [10]

- How is the action-value function computed in dueling networks? [5]

#### Questions@:, Lecture 5 Questions
- Describe a fully connected layer in Noisy nets (parametrization, computation,
  effective noise generation). [5]

- Write down the distributional Bellman backup operator, define Wasserstein distance,
  and state in which metric is the distributed Bellman backup operator
  a $\gamma$-contraction. [5]

- Considering C51, describe how is the distribution of rewards represented
  and how it is predicted using a neural network. [5]

- Considering distibutional Q network (C51), describe how the predicted
  distributions are represented (what are the atoms, how do we get their
  probability), and write down the loss used to train a distributional Q network
  and an algorithm to compute it (including the mapping of atoms, which does not
  need to be mathematically flawless, but enough to describe how it should be
  done). [10]

- Write down the final loss function in Rainbow, describe what atoms are,
  and explain how is an atom logit computed for a given state and action. [5]

- How exactly are predicted distributions represented in quantile regression?
  What are the advantages of quantile regression compared to C51? [5]

- Assume $F_Z$ is a cummulative density function of $Z$ and that $Z_\theta$
  is a quantile distribution. Write down the 1-Wasserstein distance betwen
  the two distributions, and explicitly write down how the closest $Z_\theta$
  looks like, assuming $F_Z^{-1}$ is continuous. [10]

#### Questions@:, Lecture 6 Questions
- Assume we can get samples with a distribution $P$. Write down the
  loss to minimize if we want to estimate the mean of the distribution and
  prove it. [5]

- Assume we can get samples with a distribution $P$. Write down the
  loss to minimize if we want to estimate the median of the distribution and
  prove it. [5]

- Assume we can get samples with a distribution $P$. Write down the
  loss to minimize if we want to estimate a quantile $τ$ and prove it. [5]

- Explain how we can solve the problem of quantile regression not being smooth
  around zero, including the formula of the result. [5]

- Write down the QR-DQN-1 training algorithm including the quantile Huber loss
  $\rho_\tau^\kappa$ (it is fine to use $\kappa=1$). How does the inputs and
  outputs of the network look like (including their shapes)? [10]

- Describe the network inputs and outputs (including their shapes) of DQN, C51,
  QR-DQN, IQN. [5]

- Describe the network architecture of IQN, including how the quantile $\tau$
  is represented. Then write down the training algorithm, including the quantile
  Huber loss $\rho_\tau^\kappa$ (it is fine to use $\kappa=1$). [10]

#### Questions@:, Lecture 7 Questions
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

- Write down the one-step Actor-critic algorithm. [10]

- How and why is entropy regularization used in policy gradient algorithms?
  What are the differences to $\epsilon$-smooth policies? [5]

- The Asynchronous advantage actor-critic (A3C) policy may utilize recurrent
  neural networks. How is the training structured to allow backpropagation
  through them (would vanilla DQN, vanilla REINFORCE, vanilla actor-critic work
  with recurrent neural networks)? [5]

- Explain the difference between a regular Actor-critic and Parallel Advantage
  Actor Critic algorithms. [5]

#### Questions@:, Lecture 8 Questions
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
  and explain why. [10]

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
  look like. [5]

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

#### Questions@:, Lecture 10 Questions
- What is the value $L_\pi(\tilde\pi)$ used in TRPO to approximate
  $v_{\tilde\pi}$? What is the value the TRPO algorithm maximizes, and under
  what constraint? [5]

- Write down the PPO algorithm, including the generalized advantage estimation
  (the $n$-step truncated lambda return. [10]

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
  setting the traces $c_t$: importance sampling, Tree-backup($\lambda$), and
  Retrace($\lambda$). [10]

#### Questions@:, Lecture 11 Questions
- Considering multi-arm bandits, write down the UCB algorithm. [5]

- Describe the inputs and outputs of a neural network used in AlphaZero, and
  describe the inputs and outputs of a Monte-Carlo tree search. [5]

- Write down the loss used in AlphaZero algorithm. [5]

- What quantities are kept in a node of a AlphaZero Monte-Carlo tree search? [5]

- How are actions selected in a AlphaZero Monte-Carlo tree search? [10]

- What does AlphaZero use to maintain exploration in a Monte-Carlo tree search,
  i.e., how does AlphaZero make sure that even actions with very low prior are
  sampled enough? [5]

- Describe the backup phase of AlphaZero Monte-Carlo tree search, i.e., the
  steps you perform when you reach a leaf during the tree search. [5]

- How are the actions selected in AlphaZero self-play? [5]

#### Questions@:, Lecture 12 Questions
- Describe the three components of a MuZero model including their exact inputs
  and outputs, and show how they are used to traverse the MCTS tree. [10]

- Describe the MCTS in MuZero – action selection (including the exact
  action-values used), how are the three components of a MuZero model used
  during the tree traversal and leaf evaluation, and the updates during
  the backup phase. [10]

- Assuming we already have a filled replay buffer, describe the MuZero
  training – the losses and the target values used in them (make sure to
  mention where all the values come from, whether from the replay buffer
  or how are they computed). [10]

- In AlphaZero, define the empirical visit count distribution generated
  by the AlphaZero action selection. Then define the distribution
  $\bar{\boldsymbol\pi}$ which the visit count distribution converges to;
  write both the objective the $\bar{\boldsymbol\pi}$ minimizes, and also
  the resulting solution. [10]

- Describe the Gumbel-Max trick, i.e., write down how to perform sampling from
  a categorical distribution using an $\operatorname{argmax}$, including the
  procedure for sampling from the $\operatorname{Gumbel}(0, 1)$ distribution.
  [5]

- Describe the Gumbel-Top-k trick, i.e., write down how to sample top $n$
  actions from a categorical distribution when given its logits and suitable
  Gumbel noise. [5]

- In GumbelZero, assuming $n$ actions are sampled using the Gumbel-Top-k trick,
  write down how to select an action $A_{n+1}$ so that choosing this action is
  a policy improvement (i.e., that it has greater or equal expected reward than
  the original policy), and prove it. [10]

#### Questions@:, Lecture 13 Questions
- Describe the components of a typical latent-space model in PlaNet
  (the transition, observation, and reward functions; and the encoder)
  and the components of a recurrent state-space model (RSSM). [5]

- Derive the variational lower bound on $\log p(o_{1:T} | a_{1:T})$ used in
  PlaNet (you can utilize the Jensen's inequality
  $\log \mathbb{E} [x] \ge \mathbb{E} [\log x]$). [10]

- Consider a model with a discrete categorical latent variable $\boldsymbol z$
  sampled from $p(\boldsymbol z; \boldsymbol \theta)$, with a loss
  $L(\boldsymbol z; \boldsymbol \omega)$. Describe how we compute the derivative
  of the loss $L$ with respect to the parameters $\boldsymbol \theta$ using
  (a) a straight-through estimator, and (b) a REINFORCE-like gradient estimator
  with a baseline. [5]

- Consider a discrete categorical variable sampled from logits $\boldsymbol l$.
  Define the $\operatorname{Gumbel-softmax}(\boldsymbol l, T)$ distribution with
  logits $\boldsymbol l$ and a temperature $T$ (no need to describe sampling
  from $\operatorname{Gumbel}(0, 1)$), and describe the main difference between
  the $\operatorname{Gumbel-softmax}(\boldsymbol l, T)$ and the
  $\operatorname{softmax}(\boldsymbol l)$ distributions. [5]

- Consider a model with a discrete categorical latent variable $\boldsymbol z$
  sampled from $p(\boldsymbol z; \boldsymbol \theta)$, with a loss
  $L(\boldsymbol z; \boldsymbol \omega)$. Describe how we compute the derivative
  of the loss $L$ with respect to the parameters $\boldsymbol \theta$ using
  (a) a Gumbel-softmax estimator, and (b) a straight-through Gumbel-softmax
  estimator. [5]

- Write down an algorithm implementing a straight-through estimator of
  a discrete categorical latent variable using automatic differentiation
  (i.e., in TensorFlow or Pytorch). [5]

- Describe the six components of the DreamerV2 recurrent state-space model
  (RSSM). [5]

- Explain the KL balancing used in DreamerV2. [5]

- Describe the training of both a critic and an actor in DreamerV2 (including
  the explicit losses). [10]

#### Questions@:, Lecture 14 Questions
- Define the Partially observable stochastic game. [5]

- Define the Partially observable stochastic game, Agent-environment cycle game,
  and prove that every POSG can be represented as AECG and vice versa. [10]

- In a Partially observable stochastic game, define the expected return for
  a given agent (including the probability of a trajectory), define the
  best response, and define the Nash equilibrium. [10]

- In Reinforcement learning from human feedback, define what models are trained
  and write down the three parallel processes performed during training. [5]

- In Reinforcement learning from human feedback, define a trajectory segment,
  write down how we estimate the probability that one trajectory segment
  is rated to be better than another trajectory segment according to the
  Bradley-Terry model, and then define the loss we minimize to fit the estimated
  reward in RLHF. [10]

- When summarizing from human feedback, define the loss we minimize to fit the
  reward model, and then describe how we train the human feedback policies
  (make sure to describe all quantities of the loss in detail). What are the
  roles of the used KL term? [10]

- In InstructGPT, write down the difference in collecting the rating compared
  to plain RLHF, define the loss used to train the reward model, and finally
  describe the reward used to train the PPO and PPO-ptx models. [5]
