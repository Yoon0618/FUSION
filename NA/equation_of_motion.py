import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, mu_0, elementary_charge
from scipy.integrate import quad

# solving 1st order ODE about y(t)
# y' + f(t)*y = g(t)
# y' = lim (h->0) (y(t+h) - y(t))/h
# y(t+h) = y(t) + h*(g(t) - f(t)*y(t))

# numerical solution
y_0 = 1
t_0 = 0
h = 0.001
T = 40
ys = [y_0]
i = 0
for t in np.arange(t_0, t_0 + T, h):

    f = lambda t: 0.1 * t
    g = lambda t: np.sin(t)
    
    y_next = ys[i] + h * (g(t) - f(t) * ys[i])
    print(f"t={t:.4f}, y={ys[i]:.4f}, y_next={y_next:.4f}")
    ys.append(y_next)
    i += 1

# algebraic solution

# Calculate t points
t_points = np.arange(t_0, t_0 + T, h)

# Function to calculate algebraic solution for a single t value
def algebraic_solution(t):
    # Function to integrate
    def integrand(tau):
        return np.exp(0.05 * tau**2) * np.sin(tau)
    
    # Numerical integration from 0 to t
    integral, _ = quad(integrand, 0, t)
    
    # Calculate solution with initial condition
    y = np.exp(-0.05 * t**2) * (integral + y_0)
    return y

# Calculate algebraic solution for all t points
algebraic_ys = [algebraic_solution(t) for t in t_points]

# Add to plot
plt.plot(t_points, algebraic_ys, 'r--', label='Algebraic Solution')

# plot results
plt.plot(np.arange(len(ys)) * h, ys, label='Numerical Solution', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.title('Comparison of Numerical and Algebraic Solutions')
plt.show()

# Phase two. nonlinear 1st order ODE
# y' - 2*y/t = -t**2*y**2
