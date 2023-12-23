import numpy as np
import matplotlib.pyplot as plt

# Define the range for the x-axis and y-values for the Dirac delta functions
x = np.array([2, 3])
y = np.array([1, 0.33])

# Plot the Dirac delta functions as vertical lines
plt.figure(figsize=(8, 4))
plt.vlines(x, 0, y, colors='b', linestyles='solid', label='Pics étoiles')
plt.scatter(x, y, color='r')  # Marks the top of the peaks
plt.ylim(0, 1.2)  # Set y-axis limit for better visualization
plt.xlim(0, 4)  # Set y-axis limit for better visualization

plt.xlabel('d')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
