import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

# Equation we want to solve:
# dT/dt = alpha * d^2T/dx^2

# T is function of x and t
# T[x,t]

# initialize T
# Time and space grid
total_time = 400
width = 50

time_grid_size = 1
space_grid_size = 1
# 151/2 = 75.5
# round(75.4

y, x = round(total_time / time_grid_size) + 1 , round(width / space_grid_size) + 1
# + 1이 필요한 이유: 점의 개수가 40이면, 길이는 39임

T = np.zeros((x, y))  # 100x100 grid for space and time
# print(T.shape)
# print(T)
alpha = 0.001  # thermal diffusivity constant

# initial condition
T[:, 0] = np.sin(np.linspace(0, 6 * np.pi, x)) # initial temperature distribution

"""
plt.imshow(T, cmap='hot')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('Initial Temperature Distribution')
plt.show()
"""
# Solver

dt = 0.01
dx = 0.01
r = alpha * dt / (dx)**2

for n in range( T.shape[1] - 1):
    # print('n', n)

    T[0, n+1] = T[0, n] + (alpha * dt / dx) * ( T[1, n] - T[0, n] )

    for i in range( T.shape[0] - 2):
        i += 1
        T[i,n+1] = T[i,n] + r * ( T[i-1,n] - 2*T[i,n] + T[i+1,n] )    
    
    X = T.shape[0] - 1
    T[X, n+1] = T[X, n] + (alpha * dt / dx) * ( T[X, n] - T[X-1, n] )

print('r :',r)

plt.imshow(T, cmap='hot', extent=(0, 10, 0, 10))
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('Initial Temperature Distribution')
plt.tight_layout()
plt.show()