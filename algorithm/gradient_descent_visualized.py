# Let's visualize gradient descent function

import numpy as np
import matplotlib.pyplot as plt


# 2D visualization

def y_function(x):
    return x**2

def y_derivative(x):
    return 2 * x

x = np.arange(-100,100,0.1)
y = y_function(x)

current_position = (80,y_function(80))

learning_r = 0.01

for _ in range(1000):
    new_x = current_position[0] - learning_r * y_derivative(current_position[0])
    new_y = y_function(new_x)
    current_position = (new_x,new_y)

    plt.plot(x, y)
    plt.scatter(current_position[0], current_position[1], color="red")
    plt.pause(0.001)
    plt.clf()


# 3D visualization

def z_function(x,y):
    return np.sin(5 * x) * np.cos(5 * y)/5

def calculate_gradient(x,y):
    return np.cos(5*x) * np.cos(5*y), -np.sin(5*x) * np.sin(5*y)

x = np.arange(-1,1,0.05)
y = np.arange(-1,1,0.05)

X, Y = np.meshgrid(x,y)
Z = z_function(X,Y)

concurrent_pos = (0.7, 0.4, z_function(0.7,0.4))
learning_rate = 0.01

ax = plt.subplot(projection="3d", computed_zorder = False)

for _ in range(1000):

    x_derivative, y_derivative = calculate_gradient(concurrent_pos[0],concurrent_pos[1])
    x_new, y_new = concurrent_pos[0] - learning_rate * x_derivative, concurrent_pos[1] - learning_rate * y_derivative

    concurrent_pos = (x_new, y_new, z_function(x_new,y_new))
    ax.plot_surface(X, Y, Z, cmap="viridis", zorder = 0)
    ax.scatter(concurrent_pos[0], concurrent_pos[1], concurrent_pos[2], color = "magenta", zorder = 1)
    plt.pause(0.001)
    ax.clear()

