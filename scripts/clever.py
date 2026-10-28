import argparse
import dataclasses
import logging
import os
import re
import asyncio
from pathlib import Path
import csv
from datetime import date, datetime, timedelta
from time import sleep

from clever_bench.benchmark import Benchmark
from clever_bench.task import ProblemViewTask, ValidationResult

from core.benchmarks.clever import format_lean, TASKS, parse_view, MANUAL_PROMPTS
from core.problem import Problem, ProblemInfo
from core.models import claude
from core import var
import copy

PROVER_CLASSES = {
    'claude': {
        'model': claude.Claude,
        'config': claude.ClaudeConfig,
        'context': claude.ContextConfig
    },
}

# Optional backends: each pulls in a heavyweight, separately-installed SDK
# (openai-codex, openhands-sdk) or a module not shipped with every checkout
# (core.models.openai). Register them lazily so a missing dependency only
# disables its own -p option instead of breaking the whole CLI.
try:
    from core.models import openai
    PROVER_CLASSES['openai'] = {
        'model': openai.OpenAI,
        'config': openai.OpenAIConfig,
        'context': openai.ContextConfig
    }
except ImportError as exc:
    logging.getLogger(__name__).warning(f"OpenAI backend (-p openai) unavailable: {exc!r}")

try:
    from core.models import codex
    PROVER_CLASSES['codex'] = {
        'model': codex.CodexProver,
        'config': codex.CodexConfig,
        'context': codex.ContextConfig,
    }
    PROVER_CLASSES['codex-pre'] = {
        'model': codex.CodexPre,
        'config': codex.CodexPreConfig,
        'context': codex.ContextConfig,
    }
except ImportError as exc:
    logging.getLogger(__name__).warning(f"Codex backend (-p codex) unavailable: {exc!r}")

try:
    from core.models import openhands
    # Kimi / MiniMax / GLM are served by an OpenAI-compatible endpoint through the OpenHands SDK.
    PROVER_CLASSES['kimi'] = {
        'model': openhands.Kimi,
        'config': openhands.KimiConfig,
        'context': openhands.ContextConfig,
    }
    PROVER_CLASSES['minimax'] = {
        'model': openhands.MiniMax,
        'config': openhands.MiniMaxConfig,
        'context': openhands.ContextConfig,
    }
    PROVER_CLASSES['glm'] = {
        'model': openhands.Glm,
        'config': openhands.GlmConfig,
        'context': openhands.ContextConfig,
    }
    PROVER_CLASSES['qwen'] = {
        'model': openhands.Qwen,
        'config': openhands.QwenConfig,
        'context': openhands.ContextConfig,
    }
except ImportError as exc:
    logging.getLogger(__name__).warning(f"OpenHands backend (-p kimi/minimax/glm) unavailable: {exc!r}")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s\t%(message)s")


def wait_until_start_time(start_at: str) -> None:
    """Sleep until ``start_at``: 'HH:MM:SS' waits for the next occurrence
    (today or tomorrow); 'DD/MM/HH:MM:SS' waits for that day and month in the
    current year and must lie in the future."""
    try:
        if '/' in start_at:
            day, month, time_part = start_at.split('/', 2)
            target_time = datetime.strptime(time_part, "%H:%M:%S").time()
            target_date = date(datetime.now().year, int(month), int(day))
        else:
            target_time = datetime.strptime(start_at, "%H:%M:%S").time()
            target_date = None
    except ValueError as exc:
        raise ValueError("Invalid --start-at value. Use HH:MM:SS or DD/MM/HH:MM:SS (24-hour format).") from exc

    now = datetime.now()
    if target_date is None:
        target = datetime.combine(now.date(), target_time)
        if target <= now:
            target += timedelta(days=1)
    else:
        target = datetime.combine(target_date, target_time)
        if target <= now:
            raise ValueError(f"--start-at {start_at} is in the past")
    logging.info(f"Delaying start until {target.strftime("%Y-%m-%d %H:%M:%S")}")
    while True:
        now = datetime.now()
        delay_seconds = (target - now).total_seconds()
        if delay_seconds <= 0:
            logging.info(f"  > starting now!")
            break
        logging.info(f"  > {delay_seconds:.0f} second remaining...")
        sleep(min(delay_seconds, 600))


_TOP_DECL_RE = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:noncomputable\s+|private\s+|protected\s+|partial\s+)*"
    r"(?:def|theorem|lemma|abbrev|structure|inductive)\s+([A-Za-z_][A-Za-z0-9_.']*)",
    re.MULTILINE)


def _duplicate_declarations(lean_text) -> list[str]:
    """Top-level names declared more than once in the assembled input(s)."""
    texts = lean_text if isinstance(lean_text, list) else [lean_text]
    dupes = []
    for text in texts:
        names = _TOP_DECL_RE.findall(text)
        dupes += sorted({n for n in names if names.count(n) > 1})
    return sorted(set(dupes))


def construct_lean(prob_view, pid, dataset, attempt, task, task_component, blocks, fields, base_dir=None):
    problem = Problem.load(base_dir=base_dir, info=ProblemInfo(id=pid, task=task, **dataset, attempt=attempt))
    prev_views = parse_view(problem=problem, task_component=task_component, blocks=blocks)
    if not isinstance(prob_view, list):
        prob_view = [prob_view, ] * len(prev_views)
    return problem, list(copy.replace(dst, **{field: getattr(src, field) for field in fields}) for dst, src in
                         zip(prob_view, prev_views))


def launch(task, dataset, config, task_component, blocks, prover_name='claude', load_kwargs=None, previous=None,
           problem_ids=None, workers=2, attempt=0, load=True, base_dir=None, dry_run=None, modifiers=None):
    benchmark = Benchmark(is_sample=dataset['subset'] == "sample_examples")
    benchmark.load_all(**(load_kwargs or dict()))
    view_task = ProblemViewTask(benchmark, task_component)
    problems = list()
    if problem_ids is None:
        problem_ids = [p.problem_id for p in benchmark.problems]
    for pid in problem_ids:
        prob_view = view_task.get_view(pid, leave_test_cases_lean=True, leave_generated_spec=True)
        if previous is not None:
            _, prob_view = construct_lean(prob_view=prob_view, pid=pid, dataset=dataset, attempt=attempt, **previous,
                                          base_dir=base_dir)
        lean_text = format_lean(prob_view, blocks)
        # Construction guard: a malformed upstream record (e.g. a full
        # `def generated_spec ...` pasted into its body section) reassembles
        # into an input with duplicated top-level declarations that cannot
        # elaborate — skip it loudly instead of burning a run on garbage.
        dupes = _duplicate_declarations(lean_text)
        if dupes:
            logging.error(f"Problem {pid:03d}: assembled input re-declares {dupes} "
                          f"(malformed upstream record?) — skipping; fix or re-run the upstream entry")
            continue
        if dry_run is not None:
            if isinstance(lean_text, list):
                for i, lt in enumerate(lean_text):
                    (Path(dry_run) / f"{task}-{pid:03d}-{i}.lean").write_text(lt)
            else:
                (Path(dry_run) / f"{task}-{pid:03d}.lean").write_text(lean_text)
            continue

        problem_info = ProblemInfo(id=pid, task=task, **dataset, attempt=attempt)
        if load:
            try:
                problem = Problem.load(base_dir=base_dir, info=problem_info)
                logging.info(f"Loaded problem {problem_info.id:03d}")
            except FileNotFoundError:
                problem = Problem(text=lean_text, info=problem_info)
        else:
            problem = Problem(text=lean_text, info=problem_info)
        problems.append(problem)
    prover_classes = PROVER_CLASSES[prover_name]
    prover_config = prover_classes['config'](
        context=prover_classes['context'](**config.pop('context')),
        workers=workers, **config
    )
    modifiers = modifiers or dict()
    for k, v in modifiers.items():
        if not v:
            continue
        if k == 'native_skills':
            # Boolean toggle (codex/openhands backends): fall back to the
            # legacy inline SKILL.md + slash-command-expansion regime.
            prover_config.native_skills = False
        else:
            setattr(prover_config, k, dict())
        prover_config.name_modifier += f"-no_{k}"
    if modifiers.get('skills', False):
        key = "_".join(task.split("_")[:2])
        if key in MANUAL_PROMPTS:
            prover_config.prompt = MANUAL_PROMPTS[key]
    # In-run section validation (only OpenHandsConfig has the attribute): a
    # labeled block is 'editable' when hidden (agent-authored), else 'fixed'.
    if hasattr(prover_config, 'validation_sections'):
        rules = {b.label: 'editable' if b.hide else 'fixed' for b in blocks if b.label is not None}
        # Implementation-authoring tasks (IMPL_GEN*, END_TO_END) instruct
        # uncommenting the test cases, which changes the normalized skeleton —
        # so TEST_CASES follows IMPLEMENTATION's editability there. Proof tasks
        # keep it fixed (tests arrive uncommented and must stay untouched).
        if rules.get('IMPLEMENTATION') == 'editable' and 'TEST_CASES' in rules:
            rules['TEST_CASES'] = 'editable'
        prover_config.validation_sections = rules
    prover = prover_classes['model'](config=prover_config)
    prover.solve(problems=tuple(p for p in problems if (not load or p.solution is None)))


import difflib


def evaluate(task, dataset, task_component, blocks, previous=None, attempt=0, timeout=60000, problem_ids=None,
             load=False, erase=False, base_dir=None, dry_run=None, **kwargs):
    print(f"Evaluating {task}...")
    benchmark = Benchmark(is_sample=dataset['subset'] == "sample_examples")
    benchmark.load_all()
    view_task = ProblemViewTask(benchmark, task_component)
    if problem_ids is None:
        problem_ids = [p.problem_id for p in benchmark.problems]
    for pid in problem_ids:
        prob_view = view_task.get_view(pid)
        try:
            if previous is not None:
                _, prob_view = construct_lean(prob_view=prob_view, pid=pid, dataset=dataset, attempt=attempt,
                                              **previous,
                                              base_dir=base_dir)
            problem, prob_view = construct_lean(prob_view=prob_view, pid=pid, dataset=dataset, attempt=attempt,
                                                task=task, task_component=task_component, blocks=blocks,
                                                fields=tuple(b.field for b in blocks if b.hide), base_dir=base_dir)
            # <sanity_check>
            sol = format_lean(construct_lean(prob_view=prob_view, pid=pid, dataset=dataset, attempt=attempt,
                                             task=task, task_component=task_component, blocks=blocks,
                                             fields=("problem_spec_nl",), base_dir=base_dir)[1],
                              tuple(copy.replace(b, hide=False) for b in blocks))
            if dry_run is not None:
                if isinstance(sol, list):
                    for i, lt in enumerate(sol):
                        (Path(dry_run) / f"{task}-{pid:03d}-{i}.lean").write_text(lt)
                else:
                    (Path(dry_run) / f"{task}-{pid:03d}.lean").write_text(sol)
                continue
            diff = tuple(difflib.unified_diff(
                tuple(filter(lambda x: x.strip() != "", problem.solution[0].splitlines(keepends=True))),
                tuple(filter(lambda x: x.strip() != "", sol.splitlines(keepends=True))),
                fromfile="solution", tofile="expected"
            ))
            if len(diff) > 0:
                logging.warning(f"{task}-{pid:03d} solution mismatch:\n{''.join(diff)}")
            # </sanity_check>
        except FileNotFoundError:
            logging.warning(f"Problem {pid:03d} not found for evaluation.")
            continue
        print(f"  PROBLEM {problem.info.id:03} - {problem.metadata.get("status", "####")} {problem.metadata.get("comment", "")}")
        if 'validation' in problem.metadata:
            if erase:  # or problem.metadata['validation'][0]['error_message'] == "Timeout":
                problem.metadata.pop('validation')
                problem.save(base_dir=base_dir)
                continue
            elif load:
                valids = problem.metadata['validation']
                for valid in valids:
                    print(
                        f"    > COMP {int(valid['compilation_ok'])} | ISO {int(valid['isomorphism_ok'])}, CORR {int(valid['correctness_ok'])}    [LOADED]")
                continue
        problem.metadata['validation'] = list()
        for view in prob_view:
            valid = asyncio.run(view_task.submit_async(view, timeout_in_ms=timeout))
            problem.metadata['validation'].append(valid._asdict())
            print(
                f"    > COMP {int(valid.compilation_ok)} | ISO {int(valid.isomorphism_ok)}, CORR {int(valid.correctness_ok)}")
        problem.save(base_dir=base_dir)


def summary(tasks, dataset, subject, prover_name='claude', problem_ids=None, attempt=0, base_dir=None, out_path=None,
            transpose=True):
    if problem_ids is None:
        benchmark = Benchmark(is_sample=dataset['subset'] == "sample_examples")
        benchmark.load_all()
        problem_ids = [p.problem_id for p in benchmark.problems]

    headers = ["task"] + [f"{pid:03d}" for pid in problem_ids]

    rows = []
    for task in tasks:
        row = [task]
        pid_values = []
        for pid in problem_ids:
            try:
                problem = Problem.load(base_dir=base_dir,
                                       info=ProblemInfo(id=pid, task=task, **dataset, attempt=attempt))
                if subject == "status":
                    value = problem.metadata['status'] if problem.exception is None else problem.exception['type']
                elif subject == "timing":
                    value = f"{problem.timing:.2f}" if problem.timing is not None else "?"
                elif subject == "price":
                    result_msg = next(
                        (t for t in problem.metadata.get(prover_name, []) if t.get('type') == 'ResultMessage'), None)
                    # 'cost_usd' is the legacy key of older codex/openhands traces.
                    cost = None if result_msg is None else (
                        result_msg.get('total_cost_usd', result_msg.get('cost_usd')))
                    value = f"{cost:.4f}" if isinstance(cost, (int, float)) else "?"
                elif subject == "comment":
                    value = problem.metadata.get('comment', '')
                elif subject == "analysis":
                    value = problem.metadata.get('analysis', '')
                    if isinstance(value, list):
                        if task == "SPEC_ISO_GUIDED_MULT":
                            pass  # value = list(filter(lambda x: x is not None, value))
                        else:
                            value = (", " if task != "SPEC_ISO_GUIDED_MULT" else ";").join(map(str, value))

                elif subject == "full":
                    value = problem.metadata['status'] if problem.exception is None else problem.exception['type']
                    if isinstance(value, list):
                        for i in range(len(value)):
                            try:
                                comment = problem.metadata.get('comment', list())[i]
                                if comment != "":
                                    value[i] += f" {comment}"
                            except IndexError:
                                pass
                    else:
                        comment = problem.metadata.get('comment', '')
                        if comment != "":
                            value += f" {comment}"
                elif subject.startswith("val_"):
                    validation = problem.metadata.get('validation', list())
                    field = {"val_comp": "compilation_ok", "val_iso": "isomorphism_ok", "val_corr": "correctness_ok"}[
                        subject]
                    value = list(v[field] for v in validation)
            except FileNotFoundError:
                value = ""
            pid_values.append(value)

        n_subrows = max(len(p) for p in pid_values) if isinstance(pid_values[0], list) else 1
        for i in range(n_subrows):
            sub_row = [task if i == 0 else ""]
            for v in pid_values:
                try:
                    sub_row.append(v[i] if isinstance(v, list) else (v if i == 0 else ""))
                except IndexError:
                    sub_row.append("")
            rows.append(sub_row)

    all_rows = [headers] + rows
    out = open(out_path or base_dir / f"{dataset['dataset']}-{dataset['subset']}-{subject}.csv", "w", newline="")
    writer = csv.writer(out)
    if transpose:
        all_rows = list(map(list, zip(*all_rows)))
    writer.writerows(all_rows)


def main(args):
    if args.tmp_dir is not None:
        # Set both the env var (inherited by spawned workers) and the already
        # imported module global (used by forked workers) so a second concurrent
        # instance uses a separate scratch root and does not wipe this one's.
        os.environ['APPV_TMPDIR'] = str(args.tmp_dir)
        var.TEMPORARY_DIR = Path(args.tmp_dir)
    if args.start_at is not None:
        wait_until_start_time(args.start_at)

    if args.no_native and 'native_skills' not in {
            f.name for f in dataclasses.fields(PROVER_CLASSES[args.prover]['config'])}:
        raise SystemExit(f"-p {args.prover} has no native-skills regime; --no-native is not applicable")
    # Keys are prover-config attribute names; the "-no_<attr>" suffix must stay
    # identical between this base_dir and launch()'s name_modifier (save path).
    modifiers = {'mcp_server': args.no_mcp, 'skills': args.no_skills, 'native_skills': args.no_native}
    base_dir = args.base_dir or (
                var.DATA_DIRECTORY / (args.prover + ''.join(f"-no_{k}" for k, v in modifiers.items() if v)))

    if args.evaluate:
        for task in args.tasks:
            task = task.upper().replace('-', '_')
            task_data = TASKS[task].copy()
            ids = args.ids
            evaluate(
                task=task,
                **task_data,
                problem_ids=ids,
                attempt=args.attempt,
                base_dir=base_dir,
                dry_run=args.dry_run,
            )
    elif args.summary is not None:
        summary(
            tasks=args.tasks,
            dataset=TASKS[args.tasks[0]]['dataset'],
            subject=args.summary,
            prover_name=args.prover,
            problem_ids=args.ids,
            attempt=args.attempt,
            base_dir=base_dir
        )
    else:
        for task in args.tasks:
            task = task.upper().replace('-', '_')
            task_data = TASKS[task].copy()
            ids = args.ids
            launch(
                task=task,
                **task_data,
                prover_name=args.prover,
                problem_ids=ids,
                workers=args.workers,
                attempt=args.attempt,
                base_dir=base_dir,
                dry_run=args.dry_run,
                modifiers=modifiers
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prover', default='claude', type=str, choices=sorted(PROVER_CLASSES.keys()),
                        help='Prover backend to use')

    parser.add_argument('-e', '--evaluate', action='store_true', default=False, )
    parser.add_argument('-s', '--summary', type=str, default=None,
                        help='Write summary CSV on specified subject [status/timing/price]')

    parser.add_argument('-t', '--tasks', default=['SPEC_GEN', ], nargs='+', choices=sorted(TASKS.keys()),
                        help='Tasks name to run')
    parser.add_argument('-w', '--workers', default=2, type=int, help='Number of parallel workers to use')
    parser.add_argument('-i', '--ids', nargs='+', type=int, default=None, help='Problem IDs to run')

    # parser.add_argument("--dataset", default="clever",
    #                     help="Dataset name (default: clever)")
    # parser.add_argument("--subset", default="human_eval",
    #                     help="Subset name (default: human_eval)")
    parser.add_argument("--attempt", type=int, default=0,
                        help="Attempt number (default: 0)")
    parser.add_argument("--base-dir", type=Path, default=None,
                        help="Base directory for problem data (default: data/<prover>[-no_...])")
    parser.add_argument('--dry-run', type=str,
                        help='If provided, prints formatted Lean files to the specified directory instead of running the prover')
    parser.add_argument('--start-at', type=str,
                        help='Delay start until local time HH:MM:SS (if already passed today, waits until '
                             'tomorrow) or DD/MM/HH:MM:SS (explicit day and month, current year).')
    parser.add_argument('--no-mcp', action='store_true', default=False, help='Disable MCP server')
    parser.add_argument('--no-skills', action='store_true', default=False, help='Disable Skills')
    parser.add_argument('--no-native', action='store_true', default=False,
                        help='Disable native skills/commands loading (codex/openhands backends): inline '
                             'SKILL.md and expand /plugin:cmd prompts instead — the pre-2026-08 legacy regime')
    parser.add_argument('--tmp-dir', type=Path, default=None,
                        help='Scratch root for per-attempt envs (default: /tmp/appv or $APPV_TMPDIR). '
                             'Set a unique path per instance to run several provers concurrently.')

    args = parser.parse_args()

    main(args)
