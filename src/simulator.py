import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
from matplotlib.pyplot import figure
from scipy.integrate import solve_ivp

from systems.reaction_wheel_system import ReactionWheelSystem
from systems.system import RoboticSystem as RS
from systems.system import Vec


class Simulator:
    """
    Simulates an arbitrary robotic system and renders the animation.
    """
    
    def __init__(self, system: RS, duration=5, fps=30, speed=1) -> None:
        self.system = system
        
        self.duration = duration                # (sec)
        self.fps = fps                          # (frames/sec)
        self.frames = self.fps * self.duration  # rendered frames (frames)
        self.dt = 1 / self.fps                  # timestep (sec)
        
        self.t_range = np.linspace(0, self.duration, self.frames, endpoint=False)
        assert self.t_range[1] - self.t_range[0] == self.dt
        
        self.writer = FFMpegWriter(fps=self.fps * speed)
    
    def run(self, Q_0: Vec = np.array([0, 0, 0, 0]), figsize: tuple[float, float] = (6, 6), dpi: float = 100):
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
        
        self.ax.clear()
        
        # Center the image on the fixed anchor point and ensure the axes are equal
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Plot joints
        for joint in self.joints:
            if joint.edgecolor is not None and joint.facecolor is not None:
                self.ax.add_patch(
                    Circle(
                        (joint.x[i], joint.y[i]),
                        radius=joint.radius,
                        edgecolor=joint.edgecolor,
                        facecolor=joint.facecolor,
                        zorder=joint.zorder
                    )
                )
            else:
                self.ax.add_patch(
                    Circle(
                        (joint.x[i], joint.y[i]),
                        radius=joint.radius,
                        color=joint.color,
                        zorder=joint.zorder
                    )
                )
        
        # Plot links
        for link in self.links:
            self.ax.plot(
                [link.start.x[i], link.end.x[i]],
                [link.start.y[i], link.end.y[i]],
                linewidth=link.linewidth,
                color=link.color,
                zorder=link.zorder
            )

    def simulate(self, Q_0: Vec) -> None:
        """
        Simulate the system and store the results.
        """
        
        Q = solve_ivp(
            fun=self.system.deriv,
            t_span=(0, self.duration),
            y0=Q_0,
            t_eval=self.t_range
        ).y

        theta_t_vec = Q[:self.system.n]
        
        self.links = self.system.links(theta_t_vec)
        self.joints = self.system.joints(theta_t_vec)


if __name__ == "__main__":
    reaction_wheel = ReactionWheelSystem(
        l_1=0.5,
        l_c1=0.25,
        m_1=1,
        m_2=5,
        r=0.1,
    )
    
    sim = Simulator(
        system=reaction_wheel,
        duration=10,
        fps=60
    )
    
    sim.run().save("case-1.mp4")
