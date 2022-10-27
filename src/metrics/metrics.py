from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Optional


@dataclass
class MetricsConfig:
    progressive: bool = False


class Metrics:

    def __init__(self, config: MetricsConfig) -> None:
        self.perf_dict: dict[str, float] = {}

        self.progressive = config.progressive

    def measure_perf(self, f_list: list[Callable], args_list: Optional[list[Any]] = None) -> None:
        args_list = [None] * len(f_list) if args_list is None else args_list

        for f, args in zip(f_list, args_list):
            f_name = f.__name__

            if args is None:
                start = perf_counter()
                f()
                end = perf_counter()
            else:
                start = perf_counter()
                f(args)
                end = perf_counter()

            perf = end - start
            self.perf_dict[f_name] = perf

            if self.progressive:
                print(f"{f_name} exec time: {perf:.3f} sec")

    def print_perf(self) -> None:
        for f_name, perf in self.perf_dict.items():
            print(f"Exec time for {f_name}: {perf:.3f} sec")
