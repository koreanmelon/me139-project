import argparse
from time import perf_counter

import numpy as np

from animator import Animator
from simulator import Simulator
from systems.double_pendulum import DoublePendulum, DPParams
from systems.reaction_wheel import ReactionWheel, RWParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robotic system.")
    parser.add_argument("--system", "-s", type=str, required=True, help="The robotic system to simulate.")
    parser.add_argument("--q0", "-q", type=float, nargs="+", required=True, help="Initial joint angles and velocities.")
    parser.add_argument("--duration", "-D", type=int, default=5, help="Duration of the animation.")
    parser.add_argument("--fps", "-F", type=int, default=30, help="Frames per second of the animation.")
    parser.add_argument("--speed", "-S", type=float, default=1.0, help="Speed of the animation.")
    parser.add_argument("--save", action="store_true", help="Save the animation.")
    parser.add_argument("--no-show", action="store_true", help="Do not show the animation.")

    args = parser.parse_args()

    if args.system.lower() == "reactionwheel":
        system = ReactionWheel(
            RWParams(
                l_1=0.5,
                l_c1=0.25,
                m_1=1,
                m_2=5,
                r=0.1
            )
        )
    elif args.system.lower() == "doublependulum":
        system = DoublePendulum(DPParams())
    else:
        print("Please specify a system to simulate.")
        exit(1)

    sim = Simulator(
        system=system,
        duration=args.duration,
        fps=args.fps

    )

    start_sim = perf_counter()
    sim.run(np.array(args.q0))
    end_sim = perf_counter()

    print(f"Simulation time: {end_sim - start_sim:.3f} sec")

    ani = Animator(
        links=sim.links,
        joints=sim.joints,
        show=not args.no_show,
        duration=args.duration,
        fps=args.fps,
        speed=args.speed

    )

    start_ani = perf_counter()
    ani.run()
    end_ani = perf_counter()

    print(f"Animation time: {end_ani - start_ani:.3f} sec")

    if args.save:
        start_save = perf_counter()
        ani.save(sim.system.__class__.__name__)
        end_save = perf_counter()

        print(f"Write time: {end_save - start_save:.3f} sec")
