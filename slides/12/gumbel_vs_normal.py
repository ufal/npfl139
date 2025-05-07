#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

xs = np.linspace(-4, 5.5, 100)
ys = 1/np.sqrt(2*np.pi) * np.exp(-xs**2/2)

plt.plot(xs, np.exp(-xs - np.exp(-xs)), label="Gumbel(0, 1) PDF")
plt.plot(xs, 1/np.sqrt(2*np.pi) * np.exp(-xs**2/2), label="Normal(0, 1) PDF")
plt.xlabel("x")
plt.xlim(-3.5, 5)
plt.ylabel("PDF")
plt.grid(True)
plt.title("Standard Gumbel and Standard Normal PDF")
plt.legend(loc="upper right")
plt.savefig("gumbel_vs_normal.svg", transparent=True, bbox_inches="tight")
