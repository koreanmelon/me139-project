from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
from matplotlib.pyplot import figure, show

from systems.system.system import StyledJointT, StyledLinkT


class Animator:
    """
    Animates a system with multiple links and joints.
    """

    def __init__(self, links: list[StyledLinkT], joints: list[StyledJointT], show: bool = False, duration: int = 5, fps: int = 30, speed: float = 1.0) -> None:
        self.links = links
        self.joints = joints

        self.show = show

        self.duration: int = duration       # (sec)
        self.fps: int = fps                 # (frames/sec)
        self.frames: int = fps * duration   # rendered frames (frames)
        self.dt: float = 1 / fps            # timestep (sec)

        self.t_range = np.linspace(0, duration, self.frames, endpoint=False)
        assert self.t_range[1] - self.t_range[0] == self.dt

        self.writer = FFMpegWriter(fps=int(self.fps * speed))

    def run(self, figsize: tuple[float, float] = (6, 6), dpi: float = 100):
        """
        Render the animation.
        """

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

    def save(self, filename: str):
        """
        Save the animation to a file.
        """

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
