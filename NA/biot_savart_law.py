import numpy as np
import matplotlib.pyplot as plt

from numpy import cos, sin
from scipy.constants import pi, mu_0, elementary_charge

"""
Biot-Savart Law
B_vec(r_vec) = mu_0/(4*pi) * int_C I(r_vec') * dl_vec cross eta_vec / eta**3

C = vector along tha path C
"""

class LineCurrent():
    def __init__(self, wire_start_position=(0,0,0), wire_end_position=(1,0,0), I=1,N = 20):
        self.wire_start_position = np.array(wire_start_position)
        self.wire_end_position = np.array(wire_end_position)
        self.I = I

        self.N = N
        self.wire_positions = np.zeros((self.N, 3))
        self.t_values = np.linspace(0, 1, self.N).reshape(self.N, 1)  # shape (N,1)

        self.wire_positions[:] = self.wire_start_position*(1-self.t_values) + self.wire_end_position*self.t_values

    def field(self, position_vec):
        self.position_vec = np.array(position_vec)

        self.current_directions = self.wire_positions[1:] - self.wire_positions[:-1]
        self.last = self.current_directions[-1:]  # shape: (1, 3) ← important: use slicing!
        self.current_directions = np.concatenate((self.current_directions, self.last), axis=0)

        self.dl_vec = self.current_directions
        self.eta_vec = self.position_vec - self.wire_positions
        self.eta = np.linalg.norm(self.eta_vec)
        self.integrand = I * np.cross(self.dl_vec, self.eta_vec) / self.eta**3
        return mu_0/(4*pi) * np.sum(self.integrand, axis=0)

    __call__ = field

class LoopCurrent():
    def __init__(self, center_position=(0,0,0), radius=1, I=1, orientation=(1,0,0), N=20):
        self.center_position = np.array(center_position)
        self.radius = radius
        self.I = I
        self.orientation = np.array(orientation)
        self.N = N

        self.wire_positions = np.zeros((self.N, 3))

        self.x = orth_of(self.orientation)
        self.y = orth_of(self.orientation, self.x)
        self.t_values = np.linspace(0, 2*pi, self.N).reshape(self.N, 1)  # shape (N,1)

        self.wire_positions = np.cos(self.t_values) * self.radius * self.x + np.sin(self.t_values) * self.radius * self.y + self.center_position
        # print(self.x)
        # print(self.y)
        print(self.wire_positions)
        print(self.wire_positions.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        quiver3d(ax, self.center_position, self.orientation, length=0.2, normalize=True)
        quiver3d(ax, self.center_position, self.x, length=0.2, normalize=True, color='red')
        quiver3d(ax, self.center_position, self.y, length=0.2, normalize=True, color='green')
        print(f'wire_positions[:,0]: {wire_positions[:, 0]}')
        ax.scatter(wire_positions[:, 0], wire_positions[:, 1], wire_positions[:, 2], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def field(self, position_vec):
        self.position_vec = np.array(position_vec)

    __call__ = field

class BiotSavart():
    def __init__(self, path, parameter=(0,1), I=1, N=100, closed=True):
        self.path = path # path is set of parametric equation of path. ex: (0, cos(t*pi), sin(t*pi))
        self.parameter = parameter # range of parameter. ex: -pi to pi -> (-1*pi, pi)
        self.I = I # current
        self.N = N # number of points
        self.closed = closed # True if path is closed

        # Scale t from [0,1] to the parameter range
        t_start, t_end = self.parameter

        # check path is really closed or unclosed
        if closed and not np.allclose(self.path(t_start), self.path(t_end), atol=1e-3):
            print(f"Warning: path is not closed. \n first point vs last point: {self.path(t_start)} != {self.path(t_end)} in atol=1e-3")
        elif not closed and np.allclose(self.path(t_start), self.path(t_end), atol=1e-3):
            print(f"Warning: path is closed. \n first point vs last point: {self.path(t_start)} == {self.path(t_end)} in atol=1e-3")

        # for closed loop, path(0) == path(-1). so reject last t by [:-1]
        if closed:
            t_values = np.linspace(t_start, t_end, (self.N+1))[:-1]
        else:
            t_values = np.linspace(t_start, t_end, self.N)
        
        # Create positions by evaluating the path function at each t value
        # Assuming path is a function that takes a parameter and returns a 3D point
        self.wire_positions = np.zeros((self.N, 3))
        for i, t in enumerate(t_values):
            self.wire_positions[i] = self.path(t)
        
        # Calculate current directions. except last direction!
        self.current_directions = self.wire_positions[1:] - self.wire_positions[:-1]
        
        # direction is function of two positions. so, wire position[i] is mismatch with direction[i]. lets take average of positions...
        self.wire_positions_ = (self.wire_positions[:-1] + self.wire_positions[1:]) / 2

        # Boundary
        # if path is closed, last direction is same to first one.
        # if not, last current direction is simply copy of direction right before.
        if closed:
            last_direction = self.wire_positions[0] - self.wire_positions[-1]
            last_position = (self.wire_positions[-1] + self.wire_positions[0])/ 2
            # last_position = np.zeros(3)
        else:
            last_direction = self.current_directions[-1]
            last_position = self.wire_positions[-1]

        last_direction, last_position = last_direction.reshape(1,3), last_position.reshape(1,3)

        self.current_directions = np.concatenate((self.current_directions, last_direction), axis=0)
        self.wire_positions_avg = np.concatenate((self.wire_positions_, last_position), axis=0)

    def plot_configuration(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.wire_positions[:, 0], self.wire_positions[:, 1], self.wire_positions[:, 2], marker='o')
        ax.quiver(self.wire_positions[:, 0], self.wire_positions[:, 1], self.wire_positions[:, 2], 
                  self.current_directions[:, 0], self.current_directions[:, 1], self.current_directions[:, 2],
                  length=1, normalize=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def get_current_elements(self):
        return self.wire_positions_avg, self.current_directions

    def field(self, position_vec):
        self.position_vec = np.array(position_vec)
        dl_vec = self.current_directions
        eta_vec = self.position_vec - self.wire_positions_avg
        eta = np.linalg.norm(eta_vec, axis=1).reshape(-1, 1)
        sum = np.sum( np.cross(dl_vec, eta_vec)/eta**3, axis=0)
        self.result = mu_0*self.I/(4*pi) * sum
        return self.result
        

class NetField():
    def __init__(self, *fields):
        self.fields = list(fields)
    
    def add_field(self, field):
        """Add a new field source to the superposition"""
        self.fields.append(field)
    
    def field(self, position_vec):
        """Calculate the net field at the given position"""
        position_vec = np.array(position_vec)
        
        # Initialize net field as zeros
        net_field = np.zeros(3)
        
        # Add contribution from each field
        for field_obj in self.fields:
            net_field += field_obj.field(position_vec)
        
        return net_field
    
    # Allow calling the instance like a function
    __call__ = field

def plot_3d_field(field, X=(-1, 1), Y=(-1,1), Z=(-1, 1), grid_points=12, field_scale=0.2, show_current=True):
    """
    Plot the magnetic field vectors in a given X, Y, Z range.
    
    Parameters:
    -----------
    field : BiotSavart or NetField instance
        The field object to plot
    X, Y, Z : float, tuple, or array
        If float, interpreted as symmetric limits (-X to X)
        If tuple, interpreted as (min, max) ranges
    grid_points : int
        Number of points in each dimension for the grid
    field_scale : float
        Scaling factor for the field vector arrows
    show_current : bool
        Whether to show the current path
    """
    # Create a grid of points
    if isinstance(X, (int, float)) and isinstance(Y, (int, float)) and isinstance(Z, (int, float)):
        x = np.linspace(-X, X, grid_points)
        y = np.linspace(-Y, Y, grid_points)
        z = np.linspace(-Z, Z, grid_points)
    elif isinstance(X, tuple) and isinstance(Y, tuple) and isinstance(Z, tuple):
        x = np.linspace(X[0], X[1], grid_points)
        y = np.linspace(Y[0], Y[1], grid_points)
        z = np.linspace(Z[0], Z[1], grid_points)
    else:
        x, y, z = X, Y, Z
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z)
    positions = np.stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], axis=1)
    
    # Calculate field at each point
    B = np.array([field.field(pos) for pos in positions])
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Plot field vectors
    B_magnitudes = np.sqrt(B[:, 0]**2 + B[:, 1]**2 + B[:, 2]**2)
    max_magnitude = np.max(B_magnitudes) if np.any(B_magnitudes) else 1
    length = field_scale / max_magnitude if max_magnitude > 0 else field_scale
    
    ax.quiver(positions[:, 0], positions[:, 1], positions[:, 2], 
            B[:, 0], B[:, 1], B[:, 2], 
            length=length, normalize=False, color='blue', label='Magnetic field')
    
    # Show current paths if requested
    if show_current:
        if hasattr(field, 'fields'):  # NetField has multiple fields
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            for i, field_obj in enumerate(field.fields):
                color = colors[i % len(colors)]
                ax.scatter(field_obj.wire_positions[:, 0], field_obj.wire_positions[:, 1], 
                            field_obj.wire_positions[:, 2], color=color, marker='o', 
                            label=f'Current path {i+1}')
        else:  # BiotSavart has a single current path
            ax.scatter(field.wire_positions[:, 0], field.wire_positions[:, 1], field.wire_positions[:, 2], 
                        color='red', marker='o', label='Current path')
            
            ax.quiver(field.wire_positions[:, 0], field.wire_positions[:, 1], field.wire_positions[:, 2], 
                        field.current_directions[:, 0], field.current_directions[:, 1], field.current_directions[:, 2],
                        length=field_scale, normalize=True, color='green', label='Current direction')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()
    return fig, ax

def plot_2d_field(field, X=(-1, 1), Y=0, Z=(-1, 1), grid_size=100, show_current=True, streamline=True):
    """
    Create a 2D plot of the magnetic field in a plane with colormap showing field strength.
    
    Parameters:
    -----------
    field : BiotSavart or NetField instance
        The field object to plot
    X, Y, Z : float or tuple
        If float, represents a fixed value for that coordinate
        If tuple, represents the range (min, max) for that coordinate
    grid_size : int
        Number of points in each dimension for the grid
    show_current : bool
        Whether to show the current path
    streamline : bool
        Whether to show streamlines of the field
    """
    # Determine which plane to plot
    if isinstance(Y, (int, float)):
        # Plot in XZ plane
        plane = 'XZ'
        axis_1_range = X if isinstance(X, tuple) else (-X, X)
        axis_2_range = Z if isinstance(Z, tuple) else (-Z, Z)
        fixed_value = Y
        fixed_idx = 1
        axis_1_idx, axis_2_idx = 0, 2
        xlabel, ylabel = 'X', 'Z'
    elif isinstance(Z, (int, float)):
        # Plot in XY plane
        plane = 'XY'
        axis_1_range = X if isinstance(X, tuple) else (-X, X)
        axis_2_range = Y if isinstance(Y, tuple) else (-Y, Y)
        fixed_value = Z
        fixed_idx = 2
        axis_1_idx, axis_2_idx = 0, 1
        xlabel, ylabel = 'X', 'Y'
    elif isinstance(X, (int, float)):
        # Plot in YZ plane
        plane = 'YZ'
        axis_1_range = Y if isinstance(Y, tuple) else (-Y, Y)
        axis_2_range = Z if isinstance(Z, tuple) else (-Z, Z)
        fixed_value = X
        fixed_idx = 0
        axis_1_idx, axis_2_idx = 1, 2
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("One of X, Y, or Z must be a scalar to define a 2D plane")
    
    # Create grid and calculate field
    axis_1 = np.linspace(axis_1_range[0], axis_1_range[1], grid_size)
    axis_2 = np.linspace(axis_2_range[0], axis_2_range[1], grid_size)
    axis_1_grid, axis_2_grid = np.meshgrid(axis_1, axis_2)
    
    positions = np.zeros((grid_size * grid_size, 3))
    positions[:, axis_1_idx] = axis_1_grid.flatten()
    positions[:, axis_2_idx] = axis_2_grid.flatten()
    positions[:, fixed_idx] = fixed_value
    
    B_field = np.array([field.field(pos) for pos in positions])
    B_field = B_field.reshape(grid_size, grid_size, 3)
    
    # Calculate field magnitude and components
    B_magnitude = np.sqrt(np.sum(B_field**2, axis=2))
    u = B_field[:, :, axis_1_idx]
    v = B_field[:, :, axis_2_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = ax.pcolormesh(axis_1_grid, axis_2_grid, B_magnitude, cmap='plasma', shading='auto')
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Field Magnitude (T)')
    
    # Calculate normalized field components for streamplot and quiver
    magnitude = np.sqrt(u**2 + v**2)
    mask = magnitude > 0
    u_norm = np.zeros_like(u)
    v_norm = np.zeros_like(v)
    u_norm[mask] = u[mask] / magnitude[mask]
    v_norm[mask] = v[mask] / magnitude[mask]
    
    # Add streamlines if requested
    if streamline:
        # Use fewer points for streamplot to avoid clutter
        stream_density = 0.75
        ax.streamplot(axis_1, axis_2, u.T, v.T, density=stream_density, 
                        color='white', linewidth=0.7, arrowsize=0.8, broken_streamlines=False)
    
    # Plot direction arrows
    skip = max(1, grid_size // 20)
    ax.quiver(axis_1_grid[::skip, ::skip], axis_2_grid[::skip, ::skip], 
                u_norm[::skip, ::skip], v_norm[::skip, ::skip], 
                scale=30, width=0.002, color='k')
    
    # Show current paths
    if show_current:
        if hasattr(field, 'fields'):  # NetField
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            for i, field_obj in enumerate(field.fields):
                if hasattr(field_obj, 'wire_positions'):
                    wire_pos_1 = field_obj.wire_positions[:, axis_1_idx]
                    wire_pos_2 = field_obj.wire_positions[:, axis_2_idx]
                    wire_pos_fixed = field_obj.wire_positions[:, fixed_idx]
                    
                    mask = np.abs(wire_pos_fixed - fixed_value) < 0.1
                    if np.any(mask):
                        color = colors[i % len(colors)]
                        ax.plot(wire_pos_1[mask], wire_pos_2[mask], 'o-', 
                                color=color, label=f'Current path {i+1}')
        elif hasattr(field, 'wire_positions'):  # BiotSavart
            wire_pos_1 = field.wire_positions[:, axis_1_idx]
            wire_pos_2 = field.wire_positions[:, axis_2_idx]
            wire_pos_fixed = field.wire_positions[:, fixed_idx]
            
            mask = np.abs(wire_pos_fixed - fixed_value) < 0.1
            if np.any(mask):
                ax.plot(wire_pos_1[mask], wire_pos_2[mask], 'ro-', label='Current path')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Magnetic Field in {plane} Plane ({["X","Y","Z"][fixed_idx]} = {fixed_value})')
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig, ax
    
path1 = lambda t: (0, cos(t*2*pi), sin(t*2*pi))
path2 = lambda t: (0.5, cos(t*2*pi), sin(t*2*pi))
path3 = lambda t: (1, cos(t*2*pi), sin(t*2*pi))
path4 = lambda t: (-0.5, cos(t*2*pi), sin(t*2*pi))
path5 = lambda t: (-1, cos(t*2*pi), sin(t*2*pi))
path_start_cap= lambda t: (-1.5, cos(t*2*pi), sin(t*2*pi))
path_end_cap = lambda t: (1.5, cos(t*2*pi), sin(t*2*pi))
t = (0, 1)

loop1 = BiotSavart(path1, t, closed=True)
loop2 = BiotSavart(path2, t, closed=True)
loop3 = BiotSavart(path3, t, closed=True)
loop4 = BiotSavart(path4, t, closed=True)
loop5 = BiotSavart(path5, t, closed=True)
loop_start_cap = BiotSavart(path_start_cap, t, closed=True, I=10)
loop_end_cap = BiotSavart(path_end_cap, t, closed=True, I=10)

B_field = NetField(loop1, loop2, loop3, loop4, loop5, loop_start_cap, loop_end_cap)
# plot_3d_field(B_field, (-1.1,1.1), (-1.1,1.1), (-1.1,1.1))
plot_2d_field(B_field, (-3,3), 0, (-1.1,1.1), streamline=True)

# test = BiotSavart(path2, t, closed=False)
# loop1.plot_configuration()
# loop1.plot3d_field((-0.5,0.5), (0,1), (0,1))
# loop1.plot2d_field((-0.5,0.5), 0, (0,1.3)) # 원인 불명, y=0인 경우에만 작동함
input("Break!")


def quiver3d(ax, pos, vec, **kwargs):
    try:
        return ax.quiver(pos[:,0], pos[:,1], pos[:,2], vec[:,0], vec[:,1], vec[:,2], **kwargs)
    except:
        try:
            return ax.quiver(pos[0], pos[1], pos[2], vec[0], vec[1], vec[2], **kwargs)
        except:
            return ax.quiver(pos[0], pos[1], pos[2], vec[:,0], vec[:,1], vec[:,2], **kwargs)

def orth_of(*vector):
    vectors = list(vector)
    
    if len(vectors) == 1:
        vector = np.asarray(vector, dtype=np.float64)

        if np.allclose(vector, 0):
            raise ValueError("Zero vector has no unique orthogonal vector.")

        # 임의의 벡터와 외적 — v와 평행하지 않은 벡터를 고름
        if np.dot(vector, np.array([0, 1, 0])) < 0.9 * np.linalg.norm(vector):
            other = np.array([0, 1, 0])
        else:   
            other = np.array([1, 0, 0])
    
        ortho = np.cross(vector, other)
        return ortho / np.linalg.norm(ortho)  # 단위 벡터로 정규화
    
    elif len(vectors) == 2:
        v1, v2 = vectors
        orth = np.cross(v1, v2)
        return orth / np.linalg.norm(orth)
    
    else:
        raise ValueError("Too many input.")

