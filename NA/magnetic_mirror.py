import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, mu_0, elementary_charge
from matplotlib.colors import LogNorm

# B field of current loop
# position of the current loop: position = (x, y, z) (meters)
# radius of the current loop: R (meters)
# current in the loop: I (amperes)

class MagneticDipole():
    instances_created = 0 
    def __init__(self, dipole_position=(0,0,0), R=1, I=100, orientation=(1,0,0)):
        MagneticDipole.instances_created += 1
        self.count = MagneticDipole.instances_created

        self.dipole_position = dipole_position
        self.R = R
        self.I = I
        self.orientation = orientation

        self.normal_vec = np.array(self.orientation)
        self.normal_abs = np.linalg.norm(self.normal_vec)
        self.normal_hat = self.normal_vec/self.normal_abs
        self.m_vec = pi * self.R**2 * self.I * self.normal_hat

    def field(self, interest_position):
        """
        interest_position: 계산할 위치 (tuple or array)
        반환: B 벡터 (numpy array)
        
        B(r) = μ0/(4π) * [3(m·r̂) r̂ - m] / r^3 (dipole position = 0)
        """

        eta_vec = np.array(interest_position) - np.array(self.dipole_position)
        eta_abs = np.linalg.norm(eta_vec)
        if eta_abs == 0:
            raise ValueError("eta_abs = 0")
        eta_hat = eta_vec/eta_abs

        return mu_0/(4*pi) * ( 3*np.dot(self.m_vec, eta_hat)*eta_hat - self.m_vec ) / eta_abs**3
    
    # 어떤 기능인지 알아볼 것
    __call__ = field  # dipole(point) 로도 호출 가능

    def __repr__(self):
        return f"{self.count}th magnetic dipole, dipole_position={self.dipole_position}, R={self.R}, I={self.I}"
    def __str__(self):
        return f"{self.count}th magnetic dipole, dipole_position={self.dipole_position}, R={self.R}, I={self.I}"
    
# class MagneticField():
#     def __init__(self, *dipoles):
#         self.dipole_list = list(dipoles)

# def biot_savart():


class NetField():
    def __init__(self, *dipoles):
        self.dipole_list = list(dipoles)
    def superposition(self, interest_position):
        total = np.zeros(3) # 태초에, 아무것도 없었다.
        for dipole in self.dipole_list:
            total += dipole(interest_position)
        return total
    __call__ = superposition

dip1 = MagneticDipole()
dip2 = MagneticDipole()
print(dip1)
print(dip2)

net_field = NetField(dip1, dip1)((1,1,1))
print(net_field)

# --- Plotting the Magnetic Field ---

# 1. Define the dipoles for a magnetic mirror configuration
# Two dipoles placed along the z-axis, pointing in the same direction
dipole_position1 = (0,0,-1.5)
dipole_position3 = (0,0,0)
dipole_position2 = (0,0,1.5)
dipole_position4 = (0,0,0.75)
dipole_position5 = (0,0,-0.75)

orientation = (0,0,1)

dipole1 = MagneticDipole(dipole_position=dipole_position1, orientation=orientation, R=1, I=500)
dipole2 = MagneticDipole(dipole_position=dipole_position2, orientation=orientation, R=1, I=500)
dipole3 = MagneticDipole(dipole_position=dipole_position3, orientation=orientation, R=1, I=500)
dipole4 = MagneticDipole(dipole_position=dipole_position4, orientation=orientation, R=1, I=500)
dipole5 = MagneticDipole(dipole_position=dipole_position5, orientation=orientation, R=1, I=500)

# 2. Create a NetField object
net_field_mirror = NetField(dipole1, dipole2, dipole3, dipole4, dipole5)

# 3. Set up a 2D grid (x-z plane, at y=0) to calculate the field
x = np.linspace(-4, 4, 40)
z = np.linspace(-4, 4, 40)
X, Z = np.meshgrid(x, z)

# 4. Calculate the magnetic field vector at each point on the grid
Bx = np.zeros_like(X)
Bz = np.zeros_like(Z)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = (X[i, j], 0, Z[i, j])
        try:
            B_vec = net_field_mirror(pos)
            Bx[i, j] = B_vec[0]
            Bz[i, j] = B_vec[2]
        except ValueError:
            # Handle cases where the position is exactly at a dipole
            pass

# 5. Calculate the magnitude of the magnetic field
B_magnitude = np.sqrt(Bx**2 + Bz**2)

# 6. Plot the results using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the magnitude as a color map (contour plot)
# Use LogNorm for better visualization of varying field strengths
contour = ax.contourf(X, Z, B_magnitude, levels=20, cmap='viridis', norm=LogNorm())
fig.colorbar(contour, ax=ax, label='Magnetic Field Magnitude |B| (T)')

# Overlay the field lines (streamplot)
ax.streamplot(X, Z, Bx, Bz, color='white', density=1.2, linewidth=0.8, broken_streamlines=False)

# Add titles and labels
ax.set_title('Magnetic Mirror Field Lines and Magnitude (x-z plane)')
ax.set_xlabel('x-axis (m)')
ax.set_ylabel('z-axis (m)')
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()
