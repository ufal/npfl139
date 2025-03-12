#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-1, 1, 100)
F = (np.tanh(3 * xs) + 1) / 2
G = (np.tanh(4 * (xs + 0.5)) + 1) / 4 + (np.tanh(3 * (xs - 0.3)) + 1) / 4

plt.figure()
plt.plot(xs, F, label="CDF F")
plt.plot(xs, G, label="CDF G")
plt.fill_between(xs, F, G, color="lightgray", alpha=0.7, label="1-Wasserstein Distance")
plt.xlabel("x")
plt.ylabel("q")
plt.xticks([])
plt.yticks(np.arange(0, 1.01, step=0.25))
plt.gca().set_aspect(1)
plt.grid(True)
plt.title("1-Wasserstein Distance Illustration")
plt.legend(loc="upper left")
plt.savefig("wasserstein-1.svg", transparent=True, bbox_inches="tight")
