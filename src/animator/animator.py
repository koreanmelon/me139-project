from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.pyplot import figure, show
from systems.system.link import StyledJointT, StyledLinkT


class Animator:
    """
    Animates a system with multiple links and joints.
    """

    def __init__(self, links: list[StyledLinkT], joints: list[StyledJointT], show: bool = False, duration: int = 5, fps: int = 30, speed: float = 1.0) -> None:
        self.links = links
        self.joints = joints

        self.window_x = sum([abs(max(link.end.x - link.start.x)) for link in links])
        self.window_y = sum([abs(max(link.end.y - link.start.y)) for link in links])

        self.window_x = max(self.window_x, self.window_y) + 0.1
        self.window_y = self.window_x

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
        self.artists: list[Artist] = []

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
        Path(f"outputs/videos/{filename}").mkdir(parents=True, exist_ok=True)

        self.animation.save(f"outputs/videos/{filename}/{filename}_{timestamp}.mp4", writer=self.writer)

        return self

    def init_frame(self):
        self.ax.clear()

        # Center the image on the fixed anchor point and ensure the axes are equal
        self.ax.set_xlim(-self.window_x, self.window_x)
        self.ax.set_ylim(-self.window_y, self.window_y)
        self.ax.set_aspect('equal', adjustable='box')

        for joint in self.joints:
            self.artists.append(self.ax.add_patch(
                Circle(
                    (joint.x[0], joint.y[0]),
                    radius=joint.radius,
                    edgecolor=joint.edgecolor,
                    facecolor=joint.facecolor,
                    zorder=joint.zorder
                )
            ))

        # Plot links
        for link in self.links:
            self.artists.append(self.ax.add_line(
                Line2D(
                    [link.start.x[0], link.end.x[0]],
                    [link.start.y[0], link.end.y[0]],
                    linewidth=link.linewidth,
                    color=link.color,
                    zorder=link.zorder
                )
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
            self.artists[a_i].set(
                data=[
                    [link.start.x[i], link.end.x[i]],
                    [link.start.y[i], link.end.y[i]]
                ]
            )
            a_i += 1

        return self.artists
