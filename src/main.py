import argparse
from time import perf_counter

import numpy as np

from animator.animator import Animator
from simulator.simulator import Simulator
from systems.double_pendulum import DoublePendulum, DPParams
from systems.reaction_wheel import ReactionWheel, RWParams
from systems.three_link_reaction_wheel import TLRW, TLRWParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robotic system.")
    parser.add_argument("system", type=str, help="The robotic system to simulate.")
    parser.add_argument("--q0", "-q", type=float, nargs="+", required=True, help="Initial joint angles and velocities.")
    parser.add_argument("--duration", "-D", type=int, default=5, help="Duration of the animation.")
    parser.add_argument("--fps", "-F", type=int, default=30, help="Frames per second of the animation.")
    parser.add_argument("--speed", "-S", type=float, default=1.0, help="Speed of the animation.")
    parser.add_argument("--save", action="store_true", help="Save the animation.")
    parser.add_argument("--no-show", action="store_true", help="Do not show the animation.")

    args = parser.parse_args()

    print("Solving the system...")
    start_solve = perf_counter()

    try:
        if args.system.lower() == "reactionwheel":
            system = ReactionWheel(RWParams())
        elif args.system.lower() == "doublependulum":
            system = DoublePendulum(DPParams())
        elif args.system.lower() == "tlrw":
            system = TLRW(TLRWParams())
        else:
            print("Please specify a system to simulate.")
            exit(1)
    except Exception as e:
        end_solve = perf_counter()
        print(f"System solving failed after {end_solve - start_solve:.3f} seconds.\n")
        raise e

    end_solve = perf_counter()

    print(f"System solved in {end_solve - start_solve:.3f} seconds.\n")

    sim = Simulator(
        system=system,
        duration=args.duration,
        fps=args.fps

    )

    start_sim = perf_counter()
    try:
        sim.run(np.array(args.q0))
    except Exception as e:
        end_sim = perf_counter()
        print(f"Simulation failed after {end_sim - start_sim:.3f} seconds.\n")
        raise e

    end_sim = perf_counter()

    print(f"Simulation time: {end_sim - start_sim:.3f} sec\n")

    ani = Animator(
        links=sim.links,
        joints=sim.joints,
        show=not args.no_show,
        duration=args.duration,
        fps=args.fps,
        speed=args.speed

    )

    print("Animating...")
    start_ani = perf_counter()

    try:
        ani.run()
    except Exception as e:
        end_ani = perf_counter()
        print(f"Animation failed after {end_ani - start_ani:.3f} seconds.\n")
        raise e

    end_ani = perf_counter()

    print(f"Animation time: {end_ani - start_ani:.3f} sec\n")

    if args.save:
        print("Saving...")
        start_save = perf_counter()

        try:
            ani.save(sim.system.__class__.__name__)
        except Exception as e:
            end_save = perf_counter()
            print(f"Save failed after {end_save - start_save:.3f} seconds.\n")
            raise e

        end_save = perf_counter()

        print(f"Save time: {end_save - start_save:.3f} sec\n")
