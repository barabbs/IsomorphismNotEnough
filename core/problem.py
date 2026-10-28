from __future__ import annotations

import time
import json
import os

from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ProblemInfo:
    """Structured information about a problem instance."""
    dataset: str
    id: str | int
    subset: str = ''
    task: str = "PROOF"
    attempt: int = 0

    def get_str_id(self) -> str:
        id_str = f"{self.id:03d}" if isinstance(self.id, int) else str(self.id)
        return f"{self.subset}-{id_str}-{self.attempt}"

    def get_filepath(self) -> Path:
        return Path(self.dataset) / self.task / f"{self.get_str_id()}.json"


@dataclass
class Problem:
    """A single generation-and-evaluation cycle."""
    text: str | list[str]
    info: ProblemInfo
    solution: Optional[str | list[str]] = None
    timing: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    exception: Optional[dict] = None
    # Use wall-clock time by default (process_time underestimates async/subprocess workloads)
    _process: bool = False

    def _get_time(self) -> float:
        """Return the current timestamp from the configured clock."""
        return time.process_time() if self._process else time.perf_counter()

    def get_elapsed(self) -> float:
        """Return time elapsed since entering the context."""
        return self._get_time() - (self.timing or 0.0)

    def reset_timer(self) -> None:
        """Reset the timer to the current time."""
        self.timing = self._get_time()

    def __enter__(self) -> Problem:
        self.reset_timer()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> bool:
        self.timing = self.get_elapsed()
        if exc_type is not None and not issubclass(exc_type, KeyboardInterrupt):
            self.exception = {'type': exc_type.__name__, 'value': str(exc_val)}
            return True  # Suppress the exception
            # return False
        return False

    def save(self, base_dir: Path) -> None:
        """Serialize to JSON file."""
        filepath = base_dir / self.info.get_filepath()
        os.makedirs(filepath.parent, exist_ok=True)
        datadict = asdict(self)
        datadict.pop('_process', None)  # Exclude non-serializable field
        with open(filepath, 'w') as f:
            json.dump(datadict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, base_dir: Path, info: ProblemInfo) -> Problem:
        """Load from JSON file."""
        filepath = base_dir / info.get_filepath()
        with open(Path(filepath), 'r') as f:
            data = json.load(f)
        data.pop('info')
        return cls(info=info, **data)
