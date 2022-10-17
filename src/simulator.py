import numpy as np
import numpy.typing as npt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
from matplotlib.pyplot import figure, show
from scipy.integrate import solve_ivp

from systems.reaction_wheel_system import ReactionWheelSystem
from systems.base_system import BaseSystem

# Constants
g = 9.81 # gravity (m/s^2)

# Pendulum dimensions
l_1 = 1 # rod length (m)
l_c1 = 0.5 # distance along rod to center of mass (m)

r = 0.1 # radius of reaction wheel (m)

m_1 = 0.5 # mass of rod (kg)
m_2 = 2 # mass of reaction wheel (kg)

I_c1 = 1/12 * m_1 * l_1**2 # moment of inertia of rod about center of mass (kg m^2)
I_c2 = 1/2 * m_2 * r**2 # moment of inertia of reaction wheel about center of mass (kg m^2)

# Torques
tau_1 = 0
tau_2 = 0


class Simulator:
    def __init__(self, system: BaseSystem, duration=5, fps=30, speed=1) -> None:
        self.system = system
        
        self.duration = duration # (sec)
        self.fps = fps # (frames/sec)
        
        self.frames = self.fps * self.duration # (frames)
        self.dt = 1 / self.fps # (sec)
        
        self.t = np.linspace(0, self.duration, self.frames, endpoint=False)
        assert self.t[1] - self.t[0] == self.dt
        
        self.writer = FFMpegWriter(fps=self.fps * speed)
    
    def run(self, Q_0=np.array([0, 0, 0, 0]), figsize=(6, 6), dpi=100):
        """
        Run the simulation and render the animation.
        """
        self.simulate(Q_0)
        
        self.fig = figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.animation = FuncAnimation(
            fig=self.fig,
            func=self.render_frame,
            frames=self.frames,
            interval=self.dt * 1000
        )
        
        # show()
        
        return self

    def save(self, filename: str):
        """
        Save the animation to a file.
        """
        self.animation.save(f"assets/outputs/{filename}", writer=self.writer)
        
        return self
    
    def render_frame(self, i: int) -> None:
        """
        Render a single frame of the animation at timestep i.
        """
        
        # Define circles representing the anchor point and the reaction wheel
        anchor = Circle((0, 0), radius=0.05, facecolor='k', edgecolor='k', zorder=10)
        wheel = Circle((self.x1[i], self.y1[i]), radius=r, facecolor='b', edgecolor='b', zorder=10)
        
        self.ax.clear()
        
        # Center the image on the fixed anchor point and ensure the axes are equal
        self.ax.set_xlim(- l_1 - r, l_1 + r)
        self.ax.set_ylim(- l_1 - r, l_1 + r)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Plot the pendulum rod
        self.ax.plot([0, self.x1[i]], [0, self.y1[i]], color='k', linewidth=2)
        
        # Plot the base joint
        self.ax.add_patch(anchor)
        
        # Plot the reaction wheel
        self.ax.add_patch(wheel)

        # Indicate the reaction wheel's direction
        self.ax.plot([self.x1[i], self.x2[i]], [self.y1[i], self.y2[i]], color='red', linewidth=2, zorder=11)

    def simulate(self, Q_0: npt.NDArray[np.float64]) -> None:
        Q = solve_ivp(
            fun=self.system.deriv,
            t_span=(0, self.duration),
            y0=Q_0,
            t_eval=self.t
        ).y

        theta_1 = Q[0]
        theta_2 = Q[2]
        
        self.x1 = l_1 * np.cos(theta_1)
        self.y1 = l_1 * np.sin(theta_1)
        self.x2 = self.x1 + r * np.cos(theta_1 + theta_2)
        self.y2 = self.y1 + r * np.sin(theta_1 + theta_2)


if __name__ == "__main__":
    reaction_wheel = ReactionWheelSystem()
    sim = Simulator(reaction_wheel, 60, 60, 10)
    
    sim.run(np.array([0, 0, 0, 0])).save("case-1.mp4")
