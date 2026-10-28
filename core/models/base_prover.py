"""Base model module providing abstract infrastructure for provers."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Iterable, Optional

from core.problem import Problem
from core import var


@dataclass
class BaseConfig:
    generation_timeout: Optional[float] = 60 * 60
    workers: int = 2
    name_modifier: str = ""


class BaseProver:
    """Abstract base for provers."""
    PROVER_NAME = "base"
    CONFIG_CLASS = BaseConfig

    def __init__(self, config: Optional[CONFIG_CLASS] = None, **kwargs) -> None:
        self.config = config or self.CONFIG_CLASS()

    def _solve(self, problem: Problem, **kwarg) -> None:
        problem.solution = ""

    def _solve_wrapper(self, problem: Problem, kwargs) -> Problem:
        logging.info(f"Problem {problem.info.id} starting")
        with problem:
            self._solve(problem=problem, **kwargs)
        problem.save(base_dir=var.DATA_DIRECTORY / self._get_name())
        return problem

    def solve(self, problems: Iterable[Problem], **kwargs: Any) -> None:
        executor = ProcessPoolExecutor(max_workers=self.config.workers)
        try:
            solutions = executor.map(self._solve_wrapper, problems, repeat(kwargs), )
            for prob in solutions:
                logging.info(f"Problem {prob.info.id} solved in {prob.timing:.2f} seconds")
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    def _get_name(self) -> str:
        return self.PROVER_NAME + self.config.name_modifier
