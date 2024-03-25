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

- Explain the difference between a regular Actor-critic and Parallel Advantage
  Actor Critic algorithms. [5]
