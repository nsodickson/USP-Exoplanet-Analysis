import math
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10, 5))
plt.title("Fourier Series Angular Frequencies in 1/Days")

# Angular Frequency Calculations

baseline = 4.1 * 365
spacing = 3
x = np.linspace(0, baseline, int(baseline * 10))

"""
print()
for i in range(1, 26):
    period = baseline * (1 / i)
    freq = 2 * math.pi / period
    print(f"Index: {i}, Period (Days): {period}, Angular Frequency (1/Days) {freq}")
    plt.plot(np.sin(x * freq) - spacing * i, color="red")
    plt.plot(np.cos(x * freq) - spacing * i, color="blue")
print()
"""

bin_size = 5
kernel = np.ones(bin_size)
kernel = np.sin(bin_size) / np.sum(np.sin(bin_size))
plt.plot(np.linspace(0, bin_size - 1, bin_size), kernel)

plt.show()