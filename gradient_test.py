import numpy as np
from pyplot_utils import show_image, show_image_3d

def df_da(x, a,b, c):
    return x**2

def df_db(x,a,b,c):
    return x

def df_dc(x,a,b,c):
    return 1

def F(x, params):
    a,b,c = params
    return a*x**2 +b*x + c

# Calcul du gradient de la fonction de coût
def gradient(params, x_data, y_data):
    grad = np.zeros_like(params)

    funcs =  [df_da, df_db, df_dc]
    for i, func in enumerate(funcs):
        grad[i]= np.mean(2*func(x_data, *params)*(F(x_data,params)-y_data))
    return grad

# Fonction de coût
def cost_function(params, x_data, y_data):
    total_error = 0
    for x, y in zip(x_data, y_data):
        total_error += (F((x), params) - y)**2
    return total_error

# Descente de gradient
def gradient_descent(x_data, y_data, initial_params, learning_rate, num_iterations, batch_size):
    params = initial_params
    n_batch = len(x_data) // batch_size
    for _ in range(num_iterations):
        for i in range(n_batch):
            max_i = (i+1)*batch_size
            if max_i>len(x_data):
                max_i = len(x_data)
            grad = gradient(params, x_data[i*batch_size:max_i], y_data[i*batch_size:max_i])
            params = params - learning_rate * grad
        if _ % 10 == 0:  # Afficher le progrès tous les 100 itérations
            print("Iteration", _, "cost =", cost_function(params, x_data, y_data))
    return params


(a,b,c)= (500, 30, -10)
x = np.random.rand(1000)
y = a*x**2+b*x+c


initial_params = [1, 1,1]

# Paramètres de la descente de gradient
learning_rate = 0.1
num_iterations = 300
batch_size = 5

# Exécution de la descente de gradient
optimized_params = gradient_descent(x, y, initial_params, learning_rate, num_iterations, batch_size)
print(optimized_params)
[a,b,c] = optimized_params

# Afficher les paramètres optimisés
print("Paramètres optimisés :",a,b,c )
