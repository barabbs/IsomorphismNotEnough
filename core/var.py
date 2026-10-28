"""Global configuration constants, paths, timeouts, flags, and Lean keywords."""

from pathlib import Path
import os
from datetime import datetime

DEFAULT_LEAN_VERSION = "v4.24.0"

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
LOGS_DATETIME_FORMAT = '%Y-%m-%d-%H-%M-%S'

HOSTNAME = os.uname().nodename
START_TIME = datetime.now()

################ PATHS & FILENAMES ################

#### ROOT
ROOT_DIR = Path(__file__).parent.parent.absolute()
AGENTS_DIRECTORY = ROOT_DIR / "agents"
SKILLS_DIRECTORY = AGENTS_DIRECTORY / "skills"
ENVS_DIRECTORY = AGENTS_DIRECTORY / "envs"
CONTEXTS_DIRECTORY = AGENTS_DIRECTORY / "contexts"
KEYS_DIRECTORY = AGENTS_DIRECTORY / "keys"
BENCHMARKS_DIRECTORY = ROOT_DIR / "benchmarks"
DATA_DIRECTORY = ROOT_DIR / "data"
# Scratch root for per-attempt envs/skills. Override with APPV_TMPDIR (or the
# clever.py --tmp-dir flag) to run several instances at once without them wiping
# each other's /tmp/appv (e.g. openhands and codex in parallel).
TEMPORARY_DIR = Path(os.environ.get("APPV_TMPDIR", "/tmp/appv"))