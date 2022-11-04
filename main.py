import argparse
import pickle
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from src.animator.animator import Animator
from src.simulator.simulator import Simulator
from src.systems.double_pendulum import DoublePendulum, DPParams
from src.systems.reaction_wheel import ReactionWheel, RWParams
from src.systems.three_link_reaction_wheel import TLRW, TLRWParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robotic system.")
    parser.add_argument("system", type=str, help="The robotic system to simulate.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output.")
    parser.add_argument("--q0", type=float, nargs="+", required=True, help="Initial joint angles and velocities.")
    parser.add_argument("--duration", "-d", type=int, default=5, help="Duration of the animation.")
    parser.add_argument("--fps", "-f", type=int, default=30, help="Frames per second of the animation.")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speed of the animation.")
    parser.add_argument("--save", action="store_true", help="Save the animation.")
    parser.add_argument("--no-show", action="store_true", help="Do not show the animation.")

    args = parser.parse_args()

    def vprint(print_str):
        if args.verbose:
            print(print_str)

    vprint("Solving the system...")
    start_solve = perf_counter()

    try:
        if args.system.lower() == "reactionwheel":
            system = ReactionWheel(
                RWParams(
                    l_1=0.25,
                    l_c1=0.125,
                    m_1=1,
                    m_2=0.5,
                    r=0.076
                )
            )
        elif args.system.lower() == "doublependulum":
            system = DoublePendulum(DPParams())
        elif args.system.lower() == "tlrw":
            system = TLRW(TLRWParams())
        else:
            vprint("Please specify a system to simulate.")
            exit(1)
    except Exception as e:
        end_solve = perf_counter()
        vprint(f"System solving failed after {end_solve - start_solve:.3f} seconds.\n")
        raise e

    end_solve = perf_counter()

    vprint(f"System solved in {end_solve - start_solve:.3f} seconds.\n")

    # timestamp = datetime.now().isoformat(sep='T', timespec='seconds')
    # Path(f"cache/{system.__class__.__name__}_{timestamp}").touch(exist_ok=True)
    # with open(f"cache/{system.__class__.__name__}_{timestamp}", "wb") as outfile:
    #     start_serialize = perf_counter()
    #     pickle.dump(system, outfile)
    #     end_serialize = perf_counter()

    vprint(f"System serialized in {end_serialize - start_serialize:.3f} seconds.\n")

    sim = Simulator(
        system=system,
        duration=args.duration,
        fps=args.fps,
        method="DOP853"
    )

    start_sim = perf_counter()
    try:
        sim.run(np.array(args.q0))
    except Exception as e:
        end_sim = perf_counter()
        vprint(f"Simulation failed after {end_sim - start_sim:.3f} seconds.\n")
        raise e

    end_sim = perf_counter()

    vprint(f"Simulation time: {end_sim - start_sim:.3f} sec\n")

    ani = Animator(
        links=sim.links,
        joints=sim.joints,
        show=not args.no_show,
        duration=args.duration,
        fps=args.fps,
        speed=args.speed

    )

    vprint("Animating...")
    start_ani = perf_counter()

    try:
        ani.run()
    except Exception as e:
        end_ani = perf_counter()
        vprint(f"Animation failed after {end_ani - start_ani:.3f} seconds.\n")
        raise e

    end_ani = perf_counter()

    vprint(f"Animation time: {end_ani - start_ani:.3f} sec\n")

    if args.save:
        vprint("Saving...")
        start_save = perf_counter()

        try:
            ani.save(sim.system.__class__.__name__)
        except Exception as e:
            end_save = perf_counter()
            vprint(f"Save failed after {end_save - start_save:.3f} seconds.\n")
            raise e

        end_save = perf_counter()

        vprint(f"Save time: {end_save - start_save:.3f} sec\n")
