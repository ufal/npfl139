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

#### Questions@:, Lecture 4 Questions
- Assuming function approximation, define Mean squared value error. [5]

- Write down the gradient Monte-Carlo on-policy every-visit $\epsilon$-soft algorithm. [10]

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

#### Questions@:, Lecture 5 Questions
- Describe a data structure that can be used to implement prioritized replay
  buffer, so that it has given maximum capacity and insertion and sampling runs
  in time logarithmic with respect to the maximum number of elements. [10]

- How is the action-value function computed in dueling networks? [5]

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
