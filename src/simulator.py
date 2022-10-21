from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
from matplotlib.pyplot import figure, show
from scipy.integrate import solve_ivp

from systems.reaction_wheel import ReactionWheel, RWParams
from systems.system import RoboticSystem as RS
from systems.system import Vec


@dataclass
class SimConfig:
    system: RS
    duration: int = 5
    fps: int = 30
    speed: int = 1
    show: bool = False


class Simulator:
    """
    Simulates an arbitrary robotic system and renders the animation.
    """

    def __init__(self, config: SimConfig) -> None:
        self.system: RS = config.system

        self.show = config.show

        self.duration: int = config.duration            # (sec)
        self.fps: int = config.fps                      # (frames/sec)
        self.frames: int = self.fps * self.duration     # rendered frames (frames)
        self.dt: float = 1 / self.fps                   # timestep (sec)

        self.t_range = np.linspace(0, self.duration, self.frames, endpoint=False)
        assert self.t_range[1] - self.t_range[0] == self.dt

        self.writer = FFMpegWriter(fps=self.fps * config.speed)

    def run(self, Q_0: Vec = np.array([0, 0, 0, 0]), figsize: tuple[float, float] = (6, 6), dpi: float = 100):
        """
        Run the simulation and render the animation.
        """

        start_sim = perf_counter()
        self.simulate(Q_0)
        end_sim = perf_counter()

        print(f"Simulation time: {end_sim - start_sim:.3f} sec")

        self.fig = figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.artists = []

        self.animation = FuncAnimation(
            fig=self.fig,
            func=self.update_frame,
            frames=self.frames,
            init_func=self.init_frame,
            interval=self.dt * 1000,
            blit=True
        )

        if self.show:
            show()

        return self

    def save(self, filename: str = ""):
        """
        Save the animation to a file.
        """

        filename = self.system.__class__.__name__ if len(filename) == 0 else filename

        timestamp = datetime.now().isoformat(sep='T', timespec='seconds')
        Path(f"assets/outputs/{filename}").mkdir(parents=True, exist_ok=True)

        start_anim = perf_counter()
        self.animation.save(f"assets/outputs/{filename}/{filename}_{timestamp}.mp4", writer=self.writer)
        end_anim = perf_counter()

        print(f"Animation time: {end_anim - start_anim:.3f} sec")

        return self

    def init_frame(self):
        self.ax.clear()

        # Center the image on the fixed anchor point and ensure the axes are equal
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal', adjustable='box')

        for joint in self.joints:
            if joint.edgecolor is not None and joint.facecolor is not None:
                self.artists.append(self.ax.add_patch(
                    Circle(
                        (joint.x[0], joint.y[0]),
                        radius=joint.radius,
                        edgecolor=joint.edgecolor,
                        facecolor=joint.facecolor,
                        zorder=joint.zorder
                    )
                ))
            else:
                self.artists.append(self.ax.add_patch(
                    Circle(
                        (joint.x[0], joint.y[0]),
                        radius=joint.radius,
                        color=joint.color,
                        zorder=joint.zorder
                    )
                ))

        # Plot links
        for link in self.links:
            self.artists.extend(self.ax.plot(
                [link.start.x[0], link.end.x[0]],
                [link.start.y[0], link.end.y[0]],
                linewidth=link.linewidth,
                color=link.color,
                zorder=link.zorder
            ))

        return self.artists

    def update_frame(self, i: int):
        """
        Updates the state of the animation at time step i.
        """

        # Plot joints
        a_i = 0
        for joint in self.joints:
            self.artists[a_i].set(center=(joint.x[i], joint.y[i]))
            a_i += 1

        # Plot links
        for link in self.links:
            self.artists[a_i].set_data(
                [link.start.x[i], link.end.x[i]],
                [link.start.y[i], link.end.y[i]]
            )
            a_i += 1

        return self.artists

    def simulate(self, Q_0: Vec) -> None:
        """
        Simulate the system and store the results.
        """

        Q = solve_ivp(
            fun=self.system.deriv,
            t_span=(0, self.duration),
            y0=Q_0,
            method="DOP853",
            t_eval=self.t_range
        ).y

        theta_t_vec = Q[:self.system.n]

        self.links = self.system.links(theta_t_vec)
        self.joints = self.system.joints(theta_t_vec)


if __name__ == "__main__":
    reaction_wheel = ReactionWheel(
        RWParams(
            l_1=0.5,
            l_c1=0.25,
            m_1=1,
            m_2=5,
            r=0.1,
            tau=lambda Q: 0
        )
    )

    sim = Simulator(
        SimConfig(
            system=reaction_wheel,
            duration=10,
            fps=165
        )
    )

    sim.run(np.array([1.5, 0, 0, 0])).save()
