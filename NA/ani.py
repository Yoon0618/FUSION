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

y, x = round(total_time / time_grid_size) + 1 , round(width / space_grid_size) + 1

T = np.zeros((x, y))
alpha = 0.001  # thermal diffusivity constant

# initial condition
T[:, 0] = np.sin(np.linspace(0, 12 * np.pi, x)) # initial temperature distribution
# T[:round(x/2), 0] = 1
# T[round(x/2):, 0] = -1

# Solver
dt = 0.01
dx = 0.01
r = alpha * dt / (dx)**2

for n in range( T.shape[1] - 1):
    T[0, n+1] = T[0, n] + (alpha * dt / dx) * ( T[1, n] - T[0, n] )
    for i in range( T.shape[0] - 2):
        i += 1
        T[i,n+1] = T[i,n] + r * ( T[i-1,n] - 2*T[i,n] + T[i+1,n] )    
    X = T.shape[0] - 1
    T[X, n+1] = T[X, n] + (alpha * dt / dx) * ( T[X, n] - T[X-1, n] )

print('r :',r)

fig, ax = plt.subplots()
# Display the temperature distribution at t=0
im = ax.imshow(T[:,0].reshape(-1,1), cmap='hot', aspect='auto', extent=[0,1,0,width]) # Reshape for 1D data
plt.colorbar(im, ax=ax, label='Temperature')
ax.set_xlabel('Time Step (Arbitrary)') # X-axis label is a bit arbitrary for 1D data animation
ax.set_ylabel('Position (x)')
ax.set_title('Temperature Distribution over Time')


def update(frame):
    # Update the data of the image
    # We are plotting T(x) at each time step 'frame'
    # T has shape (space, time), so T[:, frame] gives T(x) at time 'frame'
    im.set_array(T[:,frame].reshape(-1,1)) # Reshape for 1D data
    im.set_clim(vmin=T.min(), vmax=T.max()) # Update color limits if temperature range changes
    ax.set_title(f'Temperature Distribution at t = {frame}')
    return [im]

# Create animation
# frames is the number of time steps in T
ani = animation.FuncAnimation(fig, update, frames=T.shape[1], blit=True, interval=5)

plt.tight_layout()
plt.show()
