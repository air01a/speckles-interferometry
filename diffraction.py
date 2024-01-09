import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# Paramètres de la simulation
lambda_light = 500e-9  # Longueur d'onde de la lumière (en mètres)
diameter = 0.000600       # Diamètre de la fente circulaire (en mètres)
distance = 1          # Distance écran-fente (en mètres)
k = 2 * np.pi / lambda_light  # Nombre d'onde

# Créer une grille de points pour le champ de diffraction
x = np.linspace(-0.001, 0.001, 400)
y = np.linspace(-0.001, 0.001, 400)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)

# Calculer le motif de diffraction
beta = k * diameter * r / (2 * distance)
intensity = (2 * j1(beta) / beta)**2
intensity /= np.max(intensity)  # Normalisation

# Afficher le motif de diffraction
plt.imshow(intensity, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.title("Diffraction Pattern of a Circular Aperture")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()