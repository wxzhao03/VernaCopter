import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import animation
from basics.scenarios import Scenarios

class Visualizer:
    """
    A class for visualizing trajectories, animating them, and calculating distances to objects.

    Attributes
    ----------
    x : numpy.ndarray
        Waypoints (only positions).
    scenario_name : str
        Name of the scenario.
    objects : dict
        Dictionary containing object boundaries.
    dt : float
        Time step.
    dT : float
        Time to reach target.
    n_points : int
        Number of targets.
    times : numpy.ndarray
        Array of time steps between two targets.
    N : int
        Total number of time steps.
    """
    def __init__(self, x, scenario): 
        """
        Initialize the Visualizer object.

        Parameters
        ----------
        x : numpy.ndarray
            Array of waypoints.
        scenario : Scenarios
            The scenario object containing objects and scenario name.
        animate : bool, optional
            Whether to enable animation (default is False).
        """
        self.x = x[:6, :]                               # waypoints (positions and velocities)
        self.scenario_name = scenario.scenario_name     # scenario name
        self.objects = scenario.objects                 # objects
        self.dt = 0.05                                  # time step
        self.dT = 1                                     # time to reach target
        n = int(self.dT/self.dt)                        # number of time steps between two targets
        self.n_points = self.x.shape[1]                 # number of targets
        self.times = np.linspace(0, self.dT, n)         # time array
        T = (self.n_points-1)*self.dT                   # total time
        self.N = int(T/self.dt)                         # number of time steps
    
    def visualize_trajectory(self):
        """
        Visualize the trajectory in a 3D plot with scenario-specific object representations.

        The function plots the trajectory waypoints, start position, and scenario-specific objects 
        (e.g., obstacles, goals, walls) using color-coded surfaces. Axes limits and visibility 
        are set according to the scenario.

        Returns
        -------
        tuple
            Matplotlib figure and axis objects for further customization or saving.
        """

        # Create the figure and 3D axis
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Trajectory')

        # Plot the trajectory and starting point
        ax.scatter(self.x[0,:], self.x[1,:], self.x[2,:], c='b', label='Trajectory')
        ax.scatter(self.x[0,0], self.x[1,0], self.x[2,0], c='g', label='Start', s=10)

        # Plot scenario-specific objects      
        if self.scenario_name == "reach_avoid": 
            self._visualize_reach_avoid(ax)
        elif self.scenario_name == "treasure_hunt":
            self._visualize_treasure_hunt(ax)

        ax.set_axis_off() # disable axes

        return fig, ax
    
    def visualize_trajectory_rho_gradient(self, rho_time_series):
        """
        Visualize trajectory in 3D with color gradient based on robustness values.
    
        Parameters
        ----------
        rho_time_series : numpy.ndarray
            Robustness values at each trajectory point.
        
        Returns
        -------
        tuple
            Matplotlib figure and axis objects.
        """

        if len(rho_time_series) != self.x.shape[1]:
            raise ValueError("Length of risk_time_series must match number of trajectory points.")
        
        # Normalize robustness values to [0, 1] with non-linear scaling
        norm_rho = np.zeros_like(rho_time_series, dtype=float)
        # Apply piecewise linear mapping
        mask1 = rho_time_series <= 0.5
        mask2 = (rho_time_series > 0.5) & (rho_time_series <= 1.0)
        mask3 = (rho_time_series > 1.0) & (rho_time_series <= 1.5)
        mask4 = rho_time_series > 1.5

        norm_rho[mask1] = 0.0  
        norm_rho[mask2] = (rho_time_series[mask2] - 0.5) / 0.5 * 0.5
        norm_rho[mask3] = 0.5 + (rho_time_series[mask3] - 1.0) / 0.5 * 0.5
        norm_rho[mask4] = 1.0
        # Create custom colormap
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('rho_colormap', ['#8B0000', '#FF0000', '#FFFF00', '#00FF00', '#006400'])
        colors = cmap(norm_rho)

        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Trajectory with Risk Coloring')
        # Plot trajectory segments with color based on local robustness
        for i in range(self.x.shape[1] - 1):
            ax.plot(
                self.x[0, i:i+2],
                self.x[1, i:i+2],
                self.x[2, i:i+2],
                color=colors[i],
                linewidth=2
            )
            
        ax.scatter(self.x[0, 0], self.x[1, 0], self.x[2, 0], c='blue', s=40, label='Start')

        if self.scenario_name == "reach_avoid": 
            self._visualize_reach_avoid(ax)
        elif self.scenario_name == "treasure_hunt":
            self._visualize_treasure_hunt(ax)
        # Add colorbar
        import matplotlib.cm as cm
        norm = mpl.colors.Normalize(vmin=0, vmax=2)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)

        cbar.set_ticks([0.5, 1.0, 1.5])
        cbar.set_ticklabels(['0.5', '1.0', '1.5'])

        ax_cbar = cbar.ax
        ax_cbar.text(1.0, 1.9, 'No Risk', transform=ax_cbar.transData,rotation=0, va='bottom', ha='center', fontsize=9)
        ax_cbar.text(1.0, 0.1, 'High Risk', transform=ax_cbar.transData,rotation=0, va='top', ha='center', fontsize=9)

        ax.set_axis_on()
        ax.view_init(elev=80, azim=-90) 
        return fig, ax
    
    def visualize_trajectory_rho_gradient_2d(self, rho_time_series): 
        """
        Visualize trajectory in 2D with color gradient based on robustness values.
    
        Parameters
        ----------
        rho_time_series : numpy.ndarray
            Robustness values at each trajectory point.
        
        Returns
        -------
        tuple
            Matplotlib figure and axis objects.
        """
        if len(rho_time_series) != self.x.shape[1]:
            raise ValueError("Length of risk_time_series must match number of trajectory points.")
        # Normalize robustness values
        norm_rho = np.zeros_like(rho_time_series, dtype=float)
        # Create color gradient from normalized robustness
        mask1 = rho_time_series <= 0.5
        mask2 = (rho_time_series > 0.5) & (rho_time_series <= 1.0)
        mask3 = (rho_time_series > 1.0) & (rho_time_series <= 1.5)
        mask4 = rho_time_series > 1.5

        norm_rho[mask1] = 0.0  
        norm_rho[mask2] = (rho_time_series[mask2] - 0.5) / 0.5 * 0.5
        norm_rho[mask3] = 0.5 + (rho_time_series[mask3] - 1.0) / 0.5 * 0.5
        norm_rho[mask4] = 1.0
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('rho_colormap', ['#8B0000', '#FF0000', '#FFFF00', '#00FF00', '#006400'])
        colors = cmap(norm_rho)
        # Create 2D plot with adjusted margins
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(left=-0.025, bottom=0.042, right=0.95, top=0.978)
        ax.set_title('Trajectory with Risk Coloring')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        # Plot trajectory
        for i in range(self.x.shape[1] - 1):
            ax.plot( self.x[0, i:i+2], self.x[1, i:i+2], color=colors[i], linewidth=3)

        ax.plot([], [], color='orange', linewidth=3, label='Planned Trajectory')
        # Mark starting position
        ax.scatter(self.x[0, 0], self.x[1, 0], c='blue', marker='o', s=40, zorder=10)
        # Visualize scenario in 2D
        if self.scenario_name == "reach_avoid": 
            self._visualize_reach_avoid_2d(ax, show_labels=True)
        elif self.scenario_name == "treasure_hunt":
            self._visualize_treasure_hunt_2d(ax, show_labels=False)

        import matplotlib.cm as cm
        norm = mpl.colors.Normalize(vmin=0, vmax=2)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)

        cbar.set_ticks([0.5, 1.0, 1.5])
        cbar.set_ticklabels(['0.5', '1.0', '1.5'])

        ax_cbar = cbar.ax
        ax_cbar.text(1.0, 1.9, 'No Risk', transform=ax_cbar.transData,rotation=0, va='bottom', ha='center', fontsize=9)
        ax_cbar.text(1.0, 0.1, 'High Risk', transform=ax_cbar.transData,rotation=0, va='top', ha='center', fontsize=9)
      
        return fig, ax

    def _visualize_reach_avoid_2d(self, ax,show_labels=True):
        """
        Visualize objects for reach_avoid scenario in 2D.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        show_labels : bool
            Whether to show object labels.
        """
        from matplotlib.patches import Rectangle
    
        for obj_name in self.objects:
            xmin, xmax, ymin, ymax, zmin, zmax = self.objects[obj_name]
        
            if 'obstacle' in obj_name:
                color = 'red'
                alpha = 0.3
            else:  # goal
                color = '#28d778'
                alpha = 0.5
        
            width = xmax - xmin
            height = ymax - ymin
            rect = Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)
        
            if show_labels:
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                ax.text(center_x, center_y, obj_name, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 5)

    def _visualize_treasure_hunt_2d(self, ax,show_labels=True):
        """
        Visualize objects for treasure_hunt scenario in 2D.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        show_labels : bool
            Whether to show labels for key objects.
        """
        from matplotlib.patches import Rectangle
    
        colors = {
            'wall': '#a3a3a3',
            'key': '#28d778',
            'door': '#c2853d',
            'chest': '#FFD700',
            'bounds': '#a3a3a3',
        }
    
        alphas = {
            'wall': 0.3,
            'key': 0.7,
            'door': 0.7,
            'chest': 0.7,
            'bounds': 0.1,
        }
    
        for obj_name in self.objects:
            xmin, xmax, ymin, ymax, zmin, zmax = self.objects[obj_name]
        
            # Determine object type
            obj_type = None
            for keyword in ['key', 'wall', 'chest', 'door', 'bounds']:
                if keyword in obj_name:
                    obj_type = keyword
                    break
        
            if obj_type is None:
                continue
        
            # Create rectangle
            width = xmax - xmin
            height = ymax - ymin
            rect = Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=1.5,
                edgecolor='black',
                facecolor=colors[obj_type],
                alpha=alphas[obj_type]
            )
            ax.add_patch(rect)
        
            # Add text labels for important objects
            if show_labels and obj_type in ['key', 'chest', 'door']:
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                ax.text(center_x, center_y, obj_name,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='white', 
                                alpha=0.8,
                                edgecolor='black'))
    
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
    
    def _visualize_reach_avoid(self, ax):
        """
        Visualize objects specific to the 'reach_avoid' scenario.
        """
        for object in self.objects:
            center, length, width, height = self.get_clwh(object)
            X, Y, Z = self.make_cuboid(center, (length, width, height))
            color = 'r' if 'obstacle' in object else '#28d778' # red for obstacles, green for goals
            
            ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=0.2, linewidth=1., edgecolor='k')

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4, 5)
        ax.set_zlim(0, 5)


    def _visualize_treasure_hunt(self, ax):
        """
        Visualize objects specific to the 'treasure_hunt' scenario.
        """

        colors = {
            'wall': '#a3a3a3',
            'key': '#28d778',
            'door': '#c2853d',
            'chest': '#FFD700',
            'bounds': '#a3a3a3',
        }

        alphas = {
            'wall': 0.05,
            'key': 0.5,
            'door': 0.5,
            'chest': 0.5,
            'bounds': 0.02,
        }

        for object in self.objects:
            # Determine the object type
            keywords = ['key', 'wall', 'chest', 'door', 'bounds']
            for keyword in keywords:
                if keyword in object:
                    object_type = keyword
                    break
            
            # Plot the object
            center, length, width, height = self.get_clwh(object)
            X, Y, Z = self.make_cuboid(center, (length, width, height))
            ax.plot_surface(X, Y, Z, color=colors[object_type], rstride=1, cstride=1, alpha=alphas[object_type], linewidth=1., edgecolor='k')          

            # Add text label
            text_positions = {
                'key': (center[0], center[1], center[2] + 1.2),
                'door': (center[0], center[1] - 2, center[2] + 0.5),
                'chest': (center[0] - 1.5, center[1], center[2] - 0.2),
            }

            if object_type in text_positions:
                ax.text(*text_positions[object_type], object, horizontalalignment='center', verticalalignment='center')

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_zlim(0, 6.4)

    def get_clwh(self, object):
        """
        Get the center, length, width, and height of an object.

        Parameters
        ----------
        object : str
            The name of the object.

        Returns
        -------
        tuple
            Center, length, width, and height of the object.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.objects[object]
        center = ((xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2)
        length = xmax - xmin
        width = ymax - ymin
        height = zmax - zmin
        return center, length, width, height

    def make_cuboid(self, center, size):
        """
        Create data arrays for cuboid plotting.

        Parameters
        ----------
        center : tuple
            Center of the cuboid (x, y, z).
        size : tuple
            Dimensions of the cuboid (length, width, height).

        Returns
        -------
        tuple
            Arrays for cuboid coordinates (X, Y, Z).
        """

        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o = [a - b / 2 for a, b in zip(center, size)]
        # get the length, width, and height
        l, w, h = size
        x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],                # x coordinate of points in bottom surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],                 # x coordinate of points in upper surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],                 # x coordinate of points in outside surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]]]                 # x coordinate of points in inside surface
        y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],                # y coordinate of points in bottom surface
            [o[1], o[1], o[1] + w, o[1] + w, o[1]],                 # y coordinate of points in upper surface
            [o[1], o[1], o[1], o[1], o[1]],                         # y coordinate of points in outside surface
            [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]     # y coordinate of points in inside surface
        z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
            [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],     # z coordinate of points in upper surface
            [o[2], o[2], o[2] + h, o[2] + h, o[2]],                 # z coordinate of points in outside surface
            [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                 # z coordinate of points in inside surface
        return np.asarray(x), np.asarray(y), np.asarray(z)
    
    
    def visualize_comparison_2d(self, rho_original, x_new, rho_series_new, label_original="Current Trajectory", label_new="Proposed Trajectory"):
        """
        Visualize side-by-side comparison of two trajectories with robustness coloring.
        
        Parameters
        ----------
        rho_original : numpy.ndarray
            Robustness values for current trajectory.
        x_new : numpy.ndarray
            New trajectory waypoints.
        rho_series_new : numpy.ndarray
            Robustness values for new trajectory.
        label_original : str
            Label for current trajectory.
        label_new : str
            Label for proposed trajectory.
            
        Returns
        -------
        tuple
            Matplotlib figure and axis objects.
        """
        # Validate input dimensions
        if len(rho_original) != self.x.shape[1]:
            raise ValueError("Length of rho_original must match number of trajectory points.")
        if len(rho_series_new) != x_new.shape[1]:
            raise ValueError("Length of rho_series_new must match number of new trajectory points.")
        # Normalize robustness
        norm_rho_orig = np.zeros_like(rho_original, dtype=float)
        
        mask1 = rho_original <= 0.5
        mask2 = (rho_original > 0.5) & (rho_original <= 1.0)
        mask3 = (rho_original > 1.0) & (rho_original <= 1.5)
        mask4 = rho_original > 1.5
        
        norm_rho_orig[mask1] = 0.0  
        norm_rho_orig[mask2] = (rho_original[mask2] - 0.5) / 0.5 * 0.5
        norm_rho_orig[mask3] = 0.5 + (rho_original[mask3] - 1.0) / 0.5 * 0.5
        norm_rho_orig[mask4] = 1.0
        # Normalize robustness for new trajectory
        norm_rho_new = np.zeros_like(rho_series_new, dtype=float)
        
        mask1_new = rho_series_new <= 0.5
        mask2_new = (rho_series_new > 0.5) & (rho_series_new <= 1.0)
        mask3_new = (rho_series_new > 1.0) & (rho_series_new <= 1.5)
        mask4_new = rho_series_new > 1.5
        
        norm_rho_new[mask1_new] = 0.0  
        norm_rho_new[mask2_new] = (rho_series_new[mask2_new] - 0.5) / 0.5 * 0.5
        norm_rho_new[mask3_new] = 0.5 + (rho_series_new[mask3_new] - 1.0) / 0.5 * 0.5
        norm_rho_new[mask4_new] = 1.0
        # Create colormap
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('rho_colormap', 
                                                ['#8B0000', '#FF0000', '#FFFF00', '#00FF00', '#006400'])
        colors_orig = cmap(norm_rho_orig)
        colors_new = cmap(norm_rho_new)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(left=-0.025, bottom=0.042, right=0.95, top=0.978)
        
        ax.set_title('Trajectory Comparison with Risk Coloring')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Plot original trajectory
        for i in range(self.x.shape[1] - 1):
            ax.plot(self.x[0, i:i+2],self.x[1, i:i+2],color=colors_orig[i],linewidth=3)
        
        ax.plot([], [], color='orange', linewidth=3, label=label_original)

        # Plot new trajectory as dashed line
        for i in range(x_new.shape[1] - 1):
            ax.plot(x_new[0, i:i+2], x_new[1, i:i+2], color=colors_new[i], linewidth=2.5, linestyle='--')
        
        ax.plot([], [], color='gray', linestyle='--', linewidth=2.5, label=label_new)
        
        ax.scatter(self.x[0, 0], self.x[1, 0], c='blue', marker='o', s=40, zorder=10)
        
        if self.scenario_name == "reach_avoid": 
            self._visualize_reach_avoid_2d(ax, show_labels=True)
        elif self.scenario_name == "treasure_hunt":
            self._visualize_treasure_hunt_2d(ax, show_labels=True)
        
        import matplotlib.cm as cm
        norm = mpl.colors.Normalize(vmin=0, vmax=2)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)
        
        cbar.set_ticks([0.5, 1.0, 1.5])
        cbar.set_ticklabels(['0.5', '1.0', '1.5'])
        
        ax_cbar = cbar.ax
        ax_cbar.text(1.0, 1.9, 'No Risk', transform=ax_cbar.transData,rotation=0, va='bottom', ha='center', fontsize=9)
        ax_cbar.text(1.0, 0.1, 'High Risk', transform=ax_cbar.transData,rotation=0, va='top', ha='center', fontsize=9)
        
        ax.legend(loc='upper right')
        
        return fig, ax