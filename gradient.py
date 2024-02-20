import numpy as np
from pyplot_utils import show_image, show_image_3d
from scipy.optimize import minimize,least_squares


def df_dA1(x, y, x01, y01, A1, b, a):
    r1 = np.sqrt((x - x01)**2 + (y - y01)**2)
    return np.exp(-b * r1**2) / (1 + a * r1**2)

def df_dA2(x, y, x02, y02, A2, b, a):
    r2 = np.sqrt((x - x02)**2 + (y - y02)**2)
    return np.exp(-b * r2**2) / (1 + a * r2**2)

def df_da(x, y, x01, y01, x02, y02, A1, A2, b, a):
    r1 = np.sqrt((x - x01)**2 + (y - y01)**2)
    r2 = np.sqrt((x - x02)**2 + (y - y02)**2)
    return -A1 * r1**2 * np.exp(-b * r1**2) / (1 + a * r1**2)**2 - A2 * r2**2 * np.exp(-b * r2**2) / (1 + a * r2**2)**2

def df_db(x, y, x01, y01, x02, y02, A1, A2, b, a):
    r1 = np.sqrt((x - x01)**2 + (y - y01)**2)
    r2 = np.sqrt((x - x02)**2 + (y - y02)**2)
    return -A1 * r1**2 * np.exp(-b * r1**2) / (1 + a * r1**2) - A2 * r2**2 * np.exp(-b * r2**2) / (1 + a * r2**2)

def df_dx01(x, y, x01, y01, A1, b, a):
    r1 = np.sqrt((x - x01)**2 + (y - y01)**2)
    return A1 * (2 * (x - x01) * np.exp(-b * r1**2) * b / (1 + a * r1**2) - 2 * (x - x01) * r1**2 * np.exp(-b * r1**2) * a / (1 + a * r1**2)**2)

def df_dy01(x, y, x01, y01, A1, b, a):
    r1 = np.sqrt((x - x01)**2 + (y - y01)**2)
    return A1 * (2 * (y - y01) * np.exp(-b * r1**2) * b / (1 + a * r1**2) - 2 * (y - y01) * r1**2 * np.exp(-b * r1**2) * a / (1 + a * r1**2)**2)

def df_dx02(x, y, x02, y02, A2, b, a):
    r2 = np.sqrt((x - x02)**2 + (y - y02)**2)
    return A2 * (2 * (x - x02) * np.exp(-b * r2**2) * b / (1 + a * r2**2) - 2 * (x - x02) * r2**2 * np.exp(-b * r2**2) * a / (1 + a * r2**2)**2)

def df_dy02(x, y, x02, y02, A2, b, a):
    r2 = np.sqrt((x - x02)**2 + (y - y02)**2)
    return A2 * (2 * (y - y02) * np.exp(-b * r2**2) * b / (1 + a * r2**2) - 2 * (y - y02) * r2**2 * np.exp(-b * r2**2) * a / (1 + a * r2**2)**2)


def F(xy, params):
    x,y = xy
    x0_1, y0_1, A1, x0_2, y0_2, A2, a, b, B = params
    r1 = np.sqrt((x - x0_1)**2 + (y - y0_1)**2)
    r2 = np.sqrt((x - x0_2)**2 + (y - y0_2)**2)
    return A1 * np.exp(-b * r1**2) / (1 + a * r1**2) + A2 * np.exp(-b * r2**2) / (1 + a * r2**2) + B

# Calcul du gradient de la fonction de coût
def gradient(params, x_data, y_data, z_data):
    grad = np.zeros_like(params)
    """epsilon = 1e-5  # Un petit nombre pour calculer la dérivée numérique
    for i in range(len(params)):
        params_plus = np.array(params)
        params_plus[i] += epsilon
        cost_plus = cost_function(params_plus, x_data, y_data, z_data)
        cost_minus = cost_function(params, x_data, y_data, z_data)
        grad[i] = (cost_plus - cost_minus) / epsilon"""
    
    funcs =  [df_dx01, df_dy01,df_dA1, df_dx02, df_dy02, df_dA2, df_da, df_db]
    for i, func in enumerate(funcs):
        grad[i]= np.mean(2*func(x_data,y_data, *params)*(F(x_data,y_data,*params)-z_data))
    return grad

# Fonction de coût
def cost_function(params, x_data, y_data, z_data):
    total_error = 0
    for x, y, z in zip(x_data, y_data, z_data):
        total_error += (F((x, y), params) - z)**2
    return total_error

# Descente de gradient
def gradient_descent(x_data, y_data, z_data, initial_params, learning_rate, num_iterations):
    params = initial_params
    for _ in range(num_iterations):
        grad = gradient(params, x_data, y_data, z_data)
        params = params - learning_rate * grad
        if _ % 100 == 0:  # Afficher le progrès tous les 100 itérations
            print("Iteration", _, "cost =", cost_function(params, x_data, y_data, z_data))
    return params

image = np.load("test.npy")
image = (image - image.min())/(image.max() - image.min())
print(image.max())
show_image(image)
y_indices, x_indices = np.indices(image.shape)

# Applatissez les données
x_data = x_indices.ravel()
y_data = y_indices.ravel()
z_data = image.ravel()

#mask = z_data_flat > 0.2
#x_data = x_data_flat[mask] / image.shape[1]
#y_data = y_data_flat[mask] / image.shape[0]
#z_data = z_data_flat[mask]


import cv2
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
print(max_loc)
#im = np.zeros_like(image)


#for i,v in enumerate(x_data):
#    print(x_data[i],y_data[i],z_data[i])
#    im[int(y_data[i]*image.shape[0]),int(x_data[i]*image.shape[1])] = z_data[i]

#show_image(im)

# Données : x_data, y_data et z_data (z_data étant les valeurs observées de F)
#x_data = np.array([...])  # Vos données x
#y_data = np.array([...])  # Vos données y
#z_data = np.array([...])  # Vos valeurs observées de F

# Paramètres initiaux (estimations initiales)
initial_params = [max_loc[0]/image.shape[1], max_loc[1]/image.shape[0], 1,0, 0, 0.5, 0.2, 0.2 ,0.]
bounds = [(0.1, 0.9), (0.1, 0.9), (1, 1), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9),(0, 1), (0, 1), (0, 1)]
# Paramètres de la descente de gradient
learning_rate = 0.1
num_iterations = 1000
result = least_squares(cost_function, initial_params, args=(x_data, y_data,z_data))
print(result.x)
x0_1, y0_1, A1, x0_2, y0_2, A2, a, b, B = result.x
x0_1 *= image.shape[1]
y0_1 *= image.shape[0]
x0_2 *= image.shape[1]
y0_2 *= image.shape[0]

print("First point : ", x0_1, y0_1, A1)
print("Second point : ", x0_2, y0_2, A2)
dist = ((x0_1-x0_2)**2 + (y0_1-y0_2)**2)**0.5
print("Distance :",dist)
"""
# Exécution de la descente de gradient
optimized_params = gradient_descent(x_data, y_data, z_data, initial_params, learning_rate, num_iterations)
print(optimized_params)
[x1, y1, A1, x2, y2, A2, a, b, B] = optimized_params
x1 *=  image.shape[1]
y1 *=  image.shape[0]
x2 *=  image.shape[1]
y2 *=  image.shape[0]
# Afficher les paramètres optimisés
print("Paramètres optimisés :", A1, A2, x1, y1, x2, y2, a, b, B )
print("dist :", 0.099*(((x2-x1)**2+(y2-y1)**2)**0.5))"""