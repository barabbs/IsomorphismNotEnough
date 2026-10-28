"""Git status helper for querying repository state."""

import os
import subprocess


def get_git_status(path: str | os.PathLike[str]) -> dict[str, str | bool]:
    """Return a dict with clean status, branch name, and HEAD commit hash.

    Args:
        path: Path to the git repository.

    Returns:
        Dict with keys 'status' (bool, True if clean), 'branch', and 'commit'.
    """
    return {
        'status': len(
            subprocess.run(['git', 'status', '--porcelain'], cwd=path, capture_output=True,
                           text=True).stdout.strip()) == 0,
        'branch': subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=path, capture_output=True,
                                 text=True).stdout.strip(),
        'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=path, capture_output=True, text=True).stdout.strip()
    }
