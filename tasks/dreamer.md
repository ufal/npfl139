### Assignment: dreamer
#### Date: Deadline: Jun 30, 22:00
#### Points: 5 points

Implement the Dreamer algorithm for the continuous
[CarRacing-v3 environment](https://gymnasium.farama.org/environments/box2d/car_racing/)
from the [Gymnasium library](https://gymnasium.farama.org/).
For this assignment, we downscale the observations of the provided
[CarRacingFS-v3](https://github.com/ufal/npfl139/tree/master/labs/npfl139/envs/car_racing.py)
environment by a factor of two to `np.uint8` images of shape $48×48×3$.

As before, the environment supports frame skipping (`args.frame_skipping`) and
the evaluation respect your value; the default of 3 is used in the reference solution.

**To pass this assignment, you must train a Dreamer-like algorithm.**
In ReCodEx, you are expected to submit an already trained model,
which is evaluated on 100 different tracks with a total time
limit of 15 minutes, with the required average return of at least 600.
My solution usually achieves the first positive evaluation return during
the first 150 training episodes and trains to 800+ during larger hundreds of
episodes.

Start with the [dreamer.py](https://github.com/ufal/npfl139/tree/master/labs/13/dreamer.py)
template. Mind the 20MB submission limit in ReCodEx—the template provides
a saving method for either a full model or a model with only the components needed
for evaluation. You can visually inspect the trained world model by using the
[show_dreams.py](https://gihub.com/ufal/npfl139/tree/master/labs/13/show_dreams.py) script,
which shows the original observation, the encoded observation (using the
encoder), and the imagination (the “dream” without the access to observations once the
imagination begins); spacebar resets the imagination and minus/plus adjust the
frame rate. However, you need the **full** model to run this script, not just the
evaluation-only one.

The template already specifies the following hyperparameters (which you can of
course change):
- the batch size and the chunk size, the replay buffer length, and
  the training/evaluation ratios;
- the structure of the discrete latent variable (32 categorical distributions
  each with 32 classes, taken from DreamerV2);
- the KL balancing and the KL loss weight (taken from DreamerV2);
- the learning rates for the world model and the agent (these are larger than
  the DreamerV2 circa by a factor of 10; training with the original ones is
  slower but more stable);
- the gradient clipping norm of 10 (100 is used in DreamerV2; with value of
  1 the model does not train).

Regarding the missing hyperparameters, you need to specify:
- the hidden layer size (of various layers and the GRU; I use small hundreds);
- the number of convolution channels (I use smaller powers of 2);
- the activation and normalization (DreamerV3 uses SiLU activation and RMSNorm
  after every hidden linear layer; I reuse it and use GroupNorm after every 
  hidden convolution, but you might experiment with other choices);
- the agent hyperparameters (I use standard gamma, trace lambda, target critic
  tau; I reuse the DreamerV2 entropy penalty of 1e-3, but I am not sure if it
  helps, and I reuse the value of 15 for the imagination horizon).
