"""BaseAgentic model for agent-based Lean theorem proving.

Provides sandboxed temporary environment setup, problem statement
isolation, abstract agent execution, result reading, and handling
of negated theorems.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import subprocess

from dataclasses import dataclass, field
from core.problem import Problem
from core.models.base_prover import BaseProver, BaseConfig
from core.utils import get_git_status
from core import var
from typing import Any, Optional
from pathlib import Path


@dataclass
class AgenticConfig(BaseConfig):
    env_template: str = "clever"
    filename: str = "problem.lean"
    input: Optional[str] = None
    output: Optional[str] = None
    skills: dict[str, Any] = field(default_factory=dict)


class RetryGenerationException(Exception):
    """Custom exception to signal repeating generation in a new environment."""

    def __init__(self, message: str = "Repeat generation requested.", wait_seconds: int = 10) -> None:
        super().__init__(message)
        self.wait_seconds = wait_seconds


class BaseAgentic(BaseProver):
    """Base class for agent-based provers using sandboxed environments.

    Manages temporary directory setup to isolated environments, agent
    execution (via the abstract ``_agent`` method), reading results back.
    """
    PROVER_NAME = "agentic"
    CONFIG_CLASS = AgenticConfig

    def _base_setup(self, **kwargs: Any) -> dict[str, Any]:
        """Create the temporary directory and copy the environment template.

        Args:
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dict with 'paths' containing base and template directory paths.
        """
        shutil.rmtree(var.TEMPORARY_DIR, ignore_errors=True)
        logging.info(f"Setting up temporary directory {var.TEMPORARY_DIR}")
        paths = {'base': var.TEMPORARY_DIR / 'envs'}
        os.makedirs(paths['base'], exist_ok=True)

        env_src = var.ENVS_DIRECTORY / self.config.env_template
        paths['template'] = var.TEMPORARY_DIR / self.config.env_template
        shutil.copytree(
            env_src, paths['template'], dirs_exist_ok=True, symlinks=True,
        )  # TODO: Copy packages to avoid symlinks pointing outside TMP_DIR
        logging.debug(
            f"Environment template copied to directory {paths['template']}"
        )
        paths['skills'] = var.TEMPORARY_DIR / 'skills'
        os.makedirs(paths['skills'], exist_ok=True)
        for key, conf in self.config.skills.items():
            src = var.SKILLS_DIRECTORY / conf.root
            git = get_git_status(src)
            if not git['status']:
                logging.error(f"Please commit changes in Skill '{key}' before running!")
                raise RuntimeError(f"Uncommitted changes in Skill '{key}' ({src})")
            for field_name in ('branch', 'commit'):
                val = getattr(conf, field_name)
                if not git[field_name] == val:
                    logging.info(
                        f"Skill '{key}' on {field_name} {git[field_name]}, please checkout {field_name} {val}...")
                    raise RuntimeError(f"Skill '{key}' {field_name} mismatch: {git[field_name]} != {val}")
            dst = str(Path(paths['skills']) / conf.root)
            shutil.copytree(src, dst)
            logging.debug(f"Skill {key}@{conf.commit} copied to directory {dst}")
        return {'paths': paths}

    def solve(self, **kwargs: Any) -> None:
        """Run generation with base environment setup."""
        super().solve(**kwargs, **self._base_setup(**kwargs))

    def _env_setup(self, problem: Problem, paths: dict[str, Path | list[Path]]) -> dict:
        """Set up an isolated environment for a single attempt.

        Copies the template into the attempt's env directory and writes
        the problem statement file.
        """
        logging.debug(f"Starting setup in environment directory {paths['env']}")
        # Setup env
        env_path = var.TEMPORARY_DIR / self.config.env_template
        shutil.copytree(env_path, paths['env'], dirs_exist_ok=True, symlinks=True)
        if self.config.input is None:
            # Copy statement to env
            text = problem.text[0] if isinstance(problem.text, list) else problem.text
            paths['problems'] = [paths['env'] / self.config.filename, ]
            paths['problems'][0].write_text(text)
        else:
            paths['problems'] = list()
            for i, p in enumerate(problem.text):
                p_path = paths['env'] / self.config.input.format(i=i)
                p_path.parent.mkdir(parents=True, exist_ok=True)
                p_path.write_text(p)
                paths['problems'].append(p_path)
        logging.debug("Setup complete.")
        return dict()

    def _get_allowed_paths(self, paths: dict[str, Path | list[Path]]) -> dict[str, tuple[Path, ...]]:
        allowed_paths = {
            'read': (
                Path("./"), paths['skills'], paths['env'],
                Path(f"/var/cache/lean/{var.DEFAULT_LEAN_VERSION}"),
                Path(subprocess.run(['elan', 'which', 'lake'], cwd=paths['env'], capture_output=True,
                                    text=True).stdout.strip()).parent.parent
            ),
            'write': (Path("./"), paths['env']),
        }
        logging.debug(f"Allowed paths: {allowed_paths}")
        return allowed_paths


    def _agent(self, problem: Problem, **kwargs: Any) -> None:
        """Execute the agent logic for a proof attempt (abstract)."""
        pass

    def _solve(self, problem: Problem, paths: dict[str, Path | list[Path]] | None = None, **kwargs: Any) -> None:
        """Run the agent in a temporary env and read back the result."""
        while True:
            try:
                with tempfile.TemporaryDirectory(
                        prefix="", dir=paths['base']
                ) as env_dir:
                    env_path = Path(env_dir)
                    paths = {**paths, 'env': env_path}
                    try:
                        env_kwargs = self._env_setup(problem=problem, paths=paths)
                        paths['allowed'] = self._get_allowed_paths(paths)
                        self._agent(problem=problem, paths=paths, **env_kwargs)
                    finally:
                        if self.config.output is None:
                            matched = paths['problems']
                        else:
                            matched = sorted(paths['env'].glob(self.config.output))
                        problem.solution = [p.read_text() for p in matched if p.is_file()]
            except RetryGenerationException as e:
                logging.info(f"Problem {problem.info.id} generation failed. Waiting {e.wait_seconds} before retrying.")
                time.sleep(e.wait_seconds)
                logging.info(f"Problem {problem.info.id}, retrying now...")
            else:
                break
