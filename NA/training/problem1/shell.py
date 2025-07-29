import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 파라미터
r = 2.0
dr = 0.1
theta = np.linspace(0, np.pi, 60)
phi = np.linspace(0, np.pi, 60)  # 반쪽만!

theta, phi = np.meshgrid(theta, phi)

# 바깥 껍질 좌표
x_out = (r + dr) * np.sin(theta) * np.cos(phi)
y_out = (r + dr) * np.sin(theta) * np.sin(phi)
z_out = (r + dr) * np.cos(theta)

# 안쪽 껍질 좌표
x_in = r * np.sin(theta) * np.cos(phi)
y_in = r * np.sin(theta) * np.sin(phi)
z_in = r * np.cos(theta)

# 표면 그리기
ax.plot_surface(x_out, y_out, z_out, color='blue', alpha=0.5, edgecolor='k', linewidth=0.2)
ax.plot_surface(x_in, y_in, z_in, color='white', alpha=0.5, edgecolor='k', linewidth=0.2)

# 측면 벽면 (자른 곳)
for phi_cut in [0, np.pi]:
    x_wall = np.array([[r*np.sin(theta_)*np.cos(phi_cut), (r+dr)*np.sin(theta_)*np.cos(phi_cut)] for theta_ in theta[0]])
    y_wall = np.array([[r*np.sin(theta_)*np.sin(phi_cut), (r+dr)*np.sin(theta_)*np.sin(phi_cut)] for theta_ in theta[0]])
    z_wall = np.array([[r*np.cos(theta_), (r+dr)*np.cos(theta_)] for theta_ in theta[0]])
    ax.plot_surface(x_wall, y_wall, z_wall, color='lightgray', alpha=0.5)

# 보기 설정
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-2.5, 2.5])
ax.axis('off')
ax.set_title("미소 구껍질 dV = 4πr²dr", fontsize=14)

plt.show()
