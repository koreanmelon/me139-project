import numpy as np
from scipy.integrate import solve_ivp
from systems.system.system import RoboticSystem as RS
from systems.system.system import Vec


class Simulator:
    """
    Simulates an arbitrary robotic system and renders the animation.
    """

    def __init__(self, system: RS, method: str = "DOP853", duration: int = 5, fps: int = 60) -> None:
        self.system = system
        self.method = method

        self.duration: int = duration       # (sec)
        self.fps: int = fps                 # (frames/sec)
        self.frames: int = fps * duration   # rendered frames (frames)
        self.dt: float = 1 / fps            # timestep (sec)

        self.t_range = np.linspace(0, self.duration, self.frames, endpoint=False)
        assert self.t_range[1] - self.t_range[0] == self.dt

    def run(self, Q_0: Vec) -> None:
        """
        Simulate the system and store the results. Note that the method used to
        solve the system of ODEs is crucial in determining the accuracy of the
        model/system.
        """

        Q = solve_ivp(
            fun=self.system.deriv,
            t_span=(0, self.duration),
            y0=Q_0,
            method=self.method,
            t_eval=self.t_range
        ).y

        theta_t_vec = Q[:self.system.n]

        self.links = self.system.link_positions(theta_t_vec)
        self.joints = self.system.joint_positions(theta_t_vec)
