### TOC: FAQ

### TOCEntry: Install

- _Installing to central user packages repository_

  You can install all required packages to central user packages repository using
  `python3 -m pip install --user --extra-index-url=https://download.pytorch.org/whl/cu118 torch~=2.2.0 torchaudio~=2.2.0 torchvision~=0.17.0 gymnasium~=1.0.0a1 pygame~=2.5.2 ufal.pybox2d~=2.3.10.3 mujoco==3.1.1 imageio~=2.34.0`.

  The above command installs CUDA 11.8 PyTorch build, but you can change `cu118` to:
  - `cpu` to get CPU-only (smaller) version,
  - `cu121` to get CUDA 12.1 build,
  - `rocm5.7` to get AMD ROCm 5.7 build.

- _Installing to a virtual environment_

  Python supports virtual environments, which are directories containing
  independent sets of installed packages. You can create a virtual environment
  by running `python3 -m venv VENV_DIR` followed by
  `VENV_DIR/bin/pip install --extra-index-url=https://download.pytorch.org/whl/cu118 torch~=2.2.0 torchaudio~=2.2.0 torchvision~=0.17.0 gymnasium~=1.0.0a1 pygame~=2.5.2 ufal.pybox2d~=2.3.10.3 mujoco==3.1.1 imageio~=2.34.0`.
  (or `VENV_DIR/Scripts/pip` on Windows).

  Again, apart from the CUDA 11.8 build, you can change `cu118` to:
  - `cpu` to get CPU-only (smaller) version,
  - `cu121` to get CUDA 12.1 build,
  - `rocm5.7` to get AMD ROCm 5.7 build.

- _**Windows** installation_

  - On Windows, it can happen that `python3` is not in PATH, while `py` command
    is – in that case you can use `py -m venv VENV_DIR`, which uses the newest
    Python available, or for example `py -3.11 -m venv VENV_DIR`, which uses
    Python version 3.11.

  - If you encounter a problem creating the logs in the `args.logdir` directory,
    a possible cause is that the path is longer than 260 characters, which is
    the default maximum length of a complete path on Windows. However, you can
    increase this limit on Windows 10, version 1607 or later, by following
    the [instructions](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).

- _**GPU** support on Linux and Windows_

  PyTorch supports NVIDIA GPU or AMD GPU out of the box, you just need to select
  appropriate `--extra-index-url` when installing the packages.

- _**GPU** support on macOS_

  The support for Apple Silicon GPUs in PyTorch+Keras is currently not great.
  Apple is working on `mlx` backend for Keras, which might improve the situation
  in the future.

  One could in theory use the TensorFlow backend, but the latest release of
  `tensorflow-metal==1.1.0` works with TensorFlow 2.14, which does not support
  Keras 3.


### TOCEntry: ReCodEx

- _What files can be submitted to ReCodEx?_

  You can submit multiple files of any type to ReCodEx. There is a limit of
  **20** files per submission, with a total size of **20MB**.

- _What file does ReCodEx execute and what arguments does it use?_

  Exactly one file with `py` suffix must contain a line starting with `def main(`.
  Such a file is imported by ReCodEx and the `main` method is executed
  (during the import, `__name__ == "__recodex__"`).

  The file must also export an argument parser called `parser`. ReCodEx uses its
  arguments and default values, but it overwrites some of the arguments
  depending on the test being executed – the template should always indicate which
  arguments are set by ReCodEx and which are left intact.

- _What are the time and memory limits?_

  The memory limit during evaluation is **1.5GB**. The time limit varies, but it should
  be at least 10 seconds and at least twice the running time of my solution.

- _Do agents need to be trained directly in ReCodEx?_

  No, you can pre-train your agent locally (unless specified otherwise in the task
  description).
