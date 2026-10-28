import argparse
import random
import re
import sys
from pathlib import Path

from core import var


def load_ids(problems_dir: Path) -> list[int]:
    ids = []
    for path in problems_dir.glob("problem_*.lean"):
        m = re.fullmatch(r"problem_(\d+)", path.stem)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def sample_ids(n: int, seed: int, problems_dir: Path, legacy: bool = False) -> list[int]:
    """Sample ``n`` problem ids, reproducibly.

    Default (nested): first ``n`` ids of a seed-determined permutation — for
    the same seed, the ``n``-sample is always a subset of the ``m``-sample
    when ``n < m``, so a run set can be extended later without discarding
    completed work. ``legacy=True`` restores the original ``random.sample``
    draw (NOT nested), reproducing the ids of samples taken before the
    nested scheme was introduced.
    """
    ids = load_ids(problems_dir)
    print(f"{len(ids)} problem ids found in {problems_dir}.", file=sys.stderr)
    if n > len(ids):
        raise ValueError(f"Requested {n} ids but only {len(ids)} available.")
    rng = random.Random(seed)
    if legacy:
        return sorted(rng.sample(ids, n))
    order = ids.copy()
    rng.shuffle(order)
    return sorted(order[:n])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample problem ids from the clever dataset.")
    parser.add_argument("n", type=int, help="Number of ids to sample")
    parser.add_argument("--seed", type=int, default=50550, help="Random seed for reproducibility (default: 50550)")
    parser.add_argument("--legacy", action="store_true", default=False,
                        help="Use the original random.sample draw (not nested across sizes); "
                             "reproduces samples taken before the nested scheme")
    parser.add_argument("--dir", type=Path, help="Directory of problem_*.lean files",
                        default=var.BENCHMARKS_DIRECTORY / "clever" / "src" / "lean4" / "human_eval")
    args = parser.parse_args()

    print(" ".join(str(i) for i in sample_ids(args.n, args.seed, args.dir, legacy=args.legacy)))
