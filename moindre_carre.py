import numpy as np
from scipy.optimize import minimize, least_squares
from pyplot_utils import show_image, show_image_3d
import matplotlib.pyplot as plt

# Définir la fonction F(x, y)
def F(xy, params):
    x0_1, y0_1, A1, x0_2, y0_2, A2, a, b, B = params
    x, y = xy
    r1 = np.sqrt((x - x0_1)**2 + (y - y0_1)**2)
    r2 = np.sqrt((x - x0_2)**2 + (y - y0_2)**2)
    return A1 * np.exp(-b * r1**2) / (1 + a * r1**2) + A2 * np.exp(-b * r2**2) / (1 + a * r2**2) + B

def residuals(params, x_data, y_data, z_data):
    predicted_z = F((x_data, y_data), params)
    return predicted_z - z_data

def cost_function(params, x_data, y_data):
    total_error = 0
    for x, y in zip(x_data, y_data):
        total_error += (F((x), *params) - y)**2
    return total_error

image = np.load("a1799.npy")
image = (image - image.min())/(image.max() - image.min())
show_image(image)
y_indices, x_indices = np.indices(image.shape)

# Applatissez les données
x_data_flat = x_indices.ravel()
y_data_flat = y_indices.ravel()
z_data_flat = image.ravel()

x_data = x_data_flat
y_data = y_data_flat
z_data = z_data_flat

"""mask = z_data_flat > 0
x_data = x_data_flat[mask] / image.shape[1]
y_data = y_data_flat[mask] / image.shape[0]
z_data = z_data_flat[mask]

im = np.zeros_like(image)


for i,v in enumerate(x_data):
    print(x_data[i],y_data[i],z_data[i])
    im[int(y_data[i]*image.shape[0]),int(x_data[i]*image.shape[1])] = z_data[i]

show_image(im)"""
indice_max = np.argmax(image)

# Convertir cet indice en coordonnées 2D
y_max, x_max = np.unravel_index(indice_max, image.shape)

print("Max coordinates", x_max, y_max)
# Données : x_data, y_data et z_data (z_data étant les valeurs observées de F)
#x_data = np.array([...])  # Vos données x
#y_data = np.array([...])  # Vos données y
#z_data = np.array([...])  # Vos valeurs observées de F
epsilon = 0.001
# Paramètres initiaux (estimations initiales)
initial_params = [x_max/image.shape[1], y_max/image.shape[0], 1, 0.5, 0.5, 0.5, 0, 0 ,0.]
lower_bounds = [x_max/image.shape[1]-epsilon, y_max/image.shape[0]-epsilon,1-epsilon,0.1,0.1,0.1,-10,-10,-10]
upper_bounds = [x_max/image.shape[1]+epsilon, y_max/image.shape[0]+epsilon,1,1,1,1,10,10,10]
# Minimiser la fonction de coût
#result = minimize(cost_function, initial_params, args=(x_data, y_data, z_data),bounds=bounds)
result = least_squares(residuals, initial_params, bounds=(lower_bounds, upper_bounds), args=(x_data, y_data, z_data))
x1, y1, A1, x2, y2, A2, a, b, B = result.x
x1 *=  image.shape[1]
y1 *=  image.shape[0]
x2 *=  image.shape[1]
y2 *=  image.shape[0]
# Afficher les paramètres optimisés
print("Paramètres optimisés :", A1, A2, x1, y1, x2, y2, a, b, B )
print("dist :", 0.099*(((x2-x1)**2+(y2-y1)**2)**0.5))


predicted_z = F((x_data, y_data), result.x)

# Calcul des résidus
residuals = predicted_z - z_data

# Tracer les résidus
plt.scatter(x_data, y_data, c=residuals, cmap='RdYlBu')
plt.colorbar(label='Résidus')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Carte des Résidus')
plt.show()