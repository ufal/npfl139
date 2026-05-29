### Assignment: rlvr
#### Date: Deadline: Jun 30, 22:00
#### Points: 4 points

Following the reinforcement learning from verifiable rewards paradigm, improve
the performance of a small LLM on a mathematical task using GRPO (or any other
purely RL algorithm).

An abstract interface of an LLM task is represented using the new
[npfl139.llm.Task](https://github.com/ufal/npfl139/blob/master/labs/npfl139/llm/task.py)
class. A task has LLM instructions, can generate examples consisting of
prompts and answers, is capable of extracting an answer from the model response,
and can evaluate a whole dataset of responses. In this assignment, we use
[npfl139.llm.ArithmeticTask](https://github.com/ufal/npfl139/blob/master/labs/npfl139/llm/arithmetic_task.py),
where the goal is to evaluate an expression with 2-5 numbers (0-9) and operators (`+`, `-`, `*`),
for example in the following prompt:
```
Compute 6 + 2 * 5 - 9 * 4.
```

In order to allow reasonably fast training and CPU-only ReCodEx evaluation,
we use an extremely small [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
model, and we train it using [LoRA](https://huggingface.co/docs/peft/v0.19.0/conceptual_guides/adapter#low-rank-adaptation-lora)
(Low-Rank Adaptation) with only 9.8M trainable parameters.
The model is trained using `bfloat16` precision, so a GPU with CC 8.0 is
recommended; the reference solution comfortably fits on a 16GB GPU.

To pass in ReCodEx, you must submit a trained LoRA checkpoint, which
just fits under the 20MB submission limit. The submitted LLM is evaluated against
a random (but fixed) test set with 256 examples and it must correctly answer
at least 50% of the prompts. The evaluation of a reference solution takes 3.5 minutes,
with the maximum limit being 15 minutes. When you run a solution with the
`--recodex` argument, it performs a ReCodEx-like evaluation on a locally
generated development set using `--seed` as the random seed.

Start with the [rlvr.py](https://github.com/ufal/npfl139/tree/master/labs/14/rlvr.py) template
containing already implemented LLM functionality: loading the pre-trained model,
creating a LoRA adapter and serializing it, generating responses for a batch of
prompts, and computing their probability in a differentiable way using teacher
forcing. To pass, you must implement a purely RL algorithm of your choice that meets the
required performance threshold; the template describes the use of a simplified GRPO
algorithm capable of achieving over 90% accuracy.

Regarding hyperparameters, the reference solution utilizes:
- a batch size of 64 (generating 8 randomly sampled responses for each of
  8 distinct training prompts);
- training using a single batch and single epoch per iteration (while the
  original GRPO/PPO uses multiple batches and epochs per iteration, which
  could result in faster training, but is more involved);
- a constant learning rate of 1e-4 (with 5e-5 and 2e-4 also working reasonably
  well);
- evaluation on a fixed development set with 256 examples every 10 training
  iterations.
